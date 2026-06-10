from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pysiglib.jax_api as pysiglib
import pytest
from georax import CG2, SO, Euclidean, Manifold

from roughrax import (
    CommutatorFreeLogODE2,
    LogODE,
    RoughTerm,
    SignatureInterpolation,
    VirtualPathInterpolation,
)
from roughrax._bases import (
    PrimitiveBasis,
    make_lyndon_basis,
    make_planar_tree_basis,
    make_tree_basis,
)
from roughrax._pseudo_bialgebra_map import form_pseudo_bialgebra_map


@dataclass(frozen=True)
class BenchmarkCase:
    dim: int
    depth: int
    solution: Literal["ito", "stratonovich"]
    geometry: Manifold
    vector_field: Callable
    ts: np.ndarray
    coarse_ts: np.ndarray
    ys: np.ndarray


@dataclass(frozen=True)
class CommutatorFreeBenchmarkCase:
    state_dim: int
    vector_field: Callable
    ts: np.ndarray
    coarse_ts: np.ndarray
    ys: np.ndarray


class MLPVectorField(eqx.Module):
    """MLP with dense weight matrices, so each evaluation costs real matmuls.

    This compute-bound regime is where the commutator-free solver's O(d)
    evaluations should beat the log-ODE's O(d^2) bracket JVPs.
    """

    config: tuple = eqx.field(static=True)

    def __call__(self, y):
        dim, state_dim, width, num_layers, seed = self.config
        rng = np.random.default_rng(seed)
        sizes = (state_dim, *(width,) * num_layers)
        hidden = y
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            weight = rng.normal(scale=fan_in**-0.5, size=(fan_in, fan_out))
            hidden = jnp.tanh(hidden @ jnp.asarray(weight, dtype=y.dtype))
        readout = rng.normal(scale=width**-0.5, size=(width, dim * state_dim))
        return jnp.tanh(hidden @ jnp.asarray(readout, dtype=y.dtype)).reshape(
            dim, y.shape[0]
        )


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def so3_vector_field(y):
    return jnp.eye(3, dtype=y.dtype)


def so3_state_dependent_vector_field(y):
    return jnp.stack(
        [
            jnp.stack([1.0 + 0.1 * y[0, 0], 0.05 * y[1, 0], 0.03 * y[2, 0]]),
            jnp.stack([0.04 * y[0, 1], 1.0 + 0.1 * y[1, 1], 0.05 * y[2, 1]]),
            jnp.stack([0.03 * y[0, 2], 0.04 * y[1, 2], 1.0 + 0.1 * y[2, 2]]),
        ]
    ).astype(y.dtype)


def brownian_like_path(*, dim: int, num_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / num_steps
    increments = rng.normal(scale=np.sqrt(dt), size=(num_steps, dim))
    return np.concatenate([np.zeros((1, dim)), np.cumsum(increments, axis=0)])


def make_case(
    *,
    dim: int,
    depth: int,
    solution: Literal["ito", "stratonovich"],
    geometry: Manifold,
    vector_field: Callable,
    seed: int,
) -> BenchmarkCase:
    num_fine_steps = 1024
    signature_window_size = 16
    ts = np.linspace(0.0, 1.0, num_fine_steps + 1)
    return BenchmarkCase(
        dim=dim,
        depth=depth,
        solution=solution,
        geometry=geometry,
        vector_field=vector_field,
        ts=ts,
        coarse_ts=ts[::signature_window_size],
        ys=brownian_like_path(dim=dim, num_steps=num_fine_steps, seed=seed),
    )


CASES = [
    pytest.param(
        make_case(
            dim=2,
            depth=3,
            solution="stratonovich",
            geometry=Euclidean(),
            vector_field=rough_vector_field,
            seed=0,
        ),
        id="euclidean-stratonovich",
    ),
    pytest.param(
        make_case(
            dim=2,
            depth=3,
            solution="ito",
            geometry=Euclidean(),
            vector_field=rough_vector_field,
            seed=0,
        ),
        id="euclidean-ito",
    ),
    pytest.param(
        make_case(
            dim=3,
            depth=2,
            solution="stratonovich",
            geometry=SO(3),
            vector_field=so3_vector_field,
            seed=1,
        ),
        id="so3-strat",
    ),
    pytest.param(
        make_case(
            dim=3,
            depth=2,
            solution="ito",
            geometry=SO(3),
            vector_field=so3_vector_field,
            seed=1,
        ),
        id="so3-ito",
    ),
    pytest.param(
        make_case(
            dim=3,
            depth=2,
            solution="stratonovich",
            geometry=SO(3),
            vector_field=so3_state_dependent_vector_field,
            seed=2,
        ),
        id="so3-state-strat",
    ),
    pytest.param(
        make_case(
            dim=3,
            depth=2,
            solution="ito",
            geometry=SO(3),
            vector_field=so3_state_dependent_vector_field,
            seed=2,
        ),
        id="so3-state-ito",
    ),
]


def make_commutator_free_case(
    *,
    dim: int,
    state_dim: int,
    seed: int,
) -> CommutatorFreeBenchmarkCase:
    num_fine_steps = 256
    signature_window_size = 16
    ts = np.linspace(0.0, 1.0, num_fine_steps + 1)
    return CommutatorFreeBenchmarkCase(
        state_dim=state_dim,
        vector_field=MLPVectorField(config=(dim, state_dim, 512, 2, seed)),
        ts=ts,
        coarse_ts=ts[::signature_window_size],
        ys=brownian_like_path(dim=dim, num_steps=num_fine_steps, seed=seed + 100),
    )


COMMUTATOR_FREE_CASES = [
    pytest.param(
        make_commutator_free_case(dim=dim, state_dim=128, seed=7 * dim),
        id=f"d{dim}-h128",
    )
    for dim in (4, 8, 16)
]


def make_basis(case: BenchmarkCase) -> PrimitiveBasis:
    if case.solution == "stratonovich":
        return make_lyndon_basis(case.depth, case.dim)
    if isinstance(case.geometry, Euclidean):
        return make_tree_basis(case.depth, case.dim)
    return make_planar_tree_basis(case.depth, case.dim)


def signature_coefficients(case: BenchmarkCase):
    indices = np.searchsorted(case.ts, case.coarse_ts)
    if case.solution == "stratonovich":
        pysiglib.prepare_log_sig(case.dim, case.depth, 1)
        return tuple(
            pysiglib.log_sig(case.ys[indices[j] : indices[j + 1] + 1], case.depth)
            for j in range(len(indices) - 1)
        )

    planar = not isinstance(case.geometry, Euclidean)
    pysiglib.prepare_branched_sig(case.dim, case.depth, planar=planar)
    return tuple(
        pysiglib.branched_log_sig(
            case.ys[indices[j] : indices[j + 1] + 1],
            case.depth,
            tree_order="canonical",
            planar=planar,
        )
        for j in range(len(indices) - 1)
    )


def prepare_signature_backend(case: BenchmarkCase):
    if case.solution == "stratonovich":
        pysiglib.prepare_log_sig(case.dim, case.depth, 1)
    else:
        planar = not isinstance(case.geometry, Euclidean)
        pysiglib.prepare_branched_sig(case.dim, case.depth, planar=planar)


@eqx.filter_jit
def _signature_coefficients_jit(
    ts,
    ys,
    signature_knots,
    depth: int,
    solution: Literal["ito", "stratonovich"],
    geometry: Manifold,
):
    driver = diffrax.LinearInterpolation(
        ts=ts,
        ys=ys,
    )
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth,
        solution,
    )
    return control.materialise(geometry).coeffs


def signature_coefficients_jit(case: BenchmarkCase):
    coeffs = _signature_coefficients_jit(
        jnp.asarray(case.ts),
        jnp.asarray(case.ys),
        jnp.asarray(case.coarse_ts),
        case.depth,
        case.solution,
        case.geometry,
    )
    return jax.block_until_ready(coeffs)


def evaluation_state(case: BenchmarkCase):
    if isinstance(case.geometry, Euclidean):
        return jnp.asarray(0.25)
    return jnp.eye(case.dim)


@eqx.filter_jit
def evaluate_lifted_fields(lifted_fields, y):
    values = jnp.stack([field(y) for field in lifted_fields])
    return jax.block_until_ready(values)


def make_rough_term_coeffs(case: BenchmarkCase):
    coeffs = _make_rough_term_coeffs(
        jnp.asarray(case.ts),
        jnp.asarray(case.ys),
        jnp.asarray(case.coarse_ts),
        case.depth,
        case.solution,
        case.geometry,
        case.vector_field,
    )
    return jax.block_until_ready(coeffs)


@eqx.filter_jit
def _make_rough_term_coeffs(
    ts,
    ys,
    signature_knots,
    depth: int,
    solution: Literal["ito", "stratonovich"],
    geometry: Manifold,
    vector_field: Callable,
) -> RoughTerm:
    driver = diffrax.LinearInterpolation(
        ts=ts,
        ys=ys,
    )
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth,
        solution,
    )
    return RoughTerm(
        vector_field,
        control,
        geometry,
    ).control.coeffs


def solve_log_ode(case: BenchmarkCase):
    y1 = _solve_log_ode(
        jnp.asarray(case.ts),
        jnp.asarray(case.ys),
        jnp.asarray(case.coarse_ts),
        case.depth,
        case.solution,
        case.geometry,
        case.vector_field,
        evaluation_state(case),
    )
    return jax.block_until_ready(y1)


def commutator_free_evaluation_state(case: CommutatorFreeBenchmarkCase):
    return jnp.linspace(-0.1, 0.1, case.state_dim)


def make_depth2_stratonovich_term(
    case: CommutatorFreeBenchmarkCase,
    solver_name: Literal["log-ode", "commutator-free"],
) -> RoughTerm:
    driver = diffrax.LinearInterpolation(
        ts=jnp.asarray(case.ts),
        ys=jnp.asarray(case.ys),
    )
    signature = SignatureInterpolation(
        driver,
        jnp.asarray(case.coarse_ts),
        depth=2,
        solution="stratonovich",
    )
    control = (
        VirtualPathInterpolation(signature).materialise(Euclidean())
        if solver_name == "commutator-free"
        else signature.materialise(Euclidean())
    )
    return RoughTerm(case.vector_field, control, Euclidean())


@eqx.filter_jit
def _solve_log_ode(
    ts,
    ys,
    signature_knots,
    depth: int,
    solution: Literal["ito", "stratonovich"],
    geometry: Manifold,
    vector_field: Callable,
    y0,
):
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth,
        solution,
    )
    term = RoughTerm(vector_field, control, geometry)
    solver = diffrax.Heun() if isinstance(geometry, Euclidean) else CG2()
    sol = diffrax.diffeqsolve(
        term,
        LogODE(solver),
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=signature_knots.shape[0] + 4,
    )
    return sol.ys[-1]


@eqx.filter_jit
def _solve_prepared_depth2_stratonovich_case(
    term,
    signature_knots,
    y0,
    solver_name: Literal["log-ode", "commutator-free"],
):
    inner_solver = diffrax.Heun()
    solver = (
        CommutatorFreeLogODE2(inner_solver)
        if solver_name == "commutator-free"
        else LogODE(inner_solver)
    )
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=signature_knots.shape[0] + 4,
    )
    return sol.ys[-1]


@pytest.mark.benchmark(group="signature")
@pytest.mark.parametrize("case", CASES)
def test_benchmark_signature_coefficients(benchmark, case: BenchmarkCase):
    prepare_signature_backend(case)
    signature_coefficients_jit(case)
    coeffs = benchmark(signature_coefficients_jit, case)
    assert coeffs.shape[0] == len(case.coarse_ts) - 1


@pytest.mark.benchmark(group="pseudo-bialgebra-map-eval")
@pytest.mark.parametrize("case", CASES)
def test_benchmark_pseudo_bialgebra_map_eval(benchmark, case: BenchmarkCase):
    basis = make_basis(case)
    lifted_fields = form_pseudo_bialgebra_map(case.vector_field, basis, case.geometry)
    y = evaluation_state(case)
    evaluate_lifted_fields(lifted_fields, y)
    values = benchmark(evaluate_lifted_fields, lifted_fields, y)
    assert values.shape[0] == len(basis.keys)


@pytest.mark.benchmark(group="log-ode")
@pytest.mark.parametrize("case", CASES)
def test_benchmark_log_ode_solve(benchmark, case: BenchmarkCase):
    solve_log_ode(case)
    y1 = benchmark(solve_log_ode, case)
    assert y1.shape == evaluation_state(case).shape


@pytest.mark.benchmark(group="commutator-free-log-ode-2")
@pytest.mark.parametrize("case", COMMUTATOR_FREE_CASES)
@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param("log-ode", id="log-ode"),
        pytest.param("commutator-free", id="commutator-free"),
    ],
)
def test_benchmark_commutator_free_log_ode_2(
    benchmark,
    case: CommutatorFreeBenchmarkCase,
    solver_name: Literal["log-ode", "commutator-free"],
):
    term = make_depth2_stratonovich_term(case, solver_name)
    signature_knots = jnp.asarray(case.coarse_ts)
    y0 = commutator_free_evaluation_state(case)

    _solve_prepared_depth2_stratonovich_case(
        term,
        signature_knots,
        y0,
        solver_name,
    )
    y1 = benchmark(
        _solve_prepared_depth2_stratonovich_case,
        term,
        signature_knots,
        y0,
        solver_name,
    )
    assert y1.shape == commutator_free_evaluation_state(case).shape


@pytest.mark.benchmark(group="rough-term")
@pytest.mark.parametrize("case", CASES)
def test_benchmark_make_rough_term(benchmark, case: BenchmarkCase):
    make_rough_term_coeffs(case)
    coeffs = benchmark(make_rough_term_coeffs, case)
    assert coeffs.shape[0] == len(case.coarse_ts) - 1
