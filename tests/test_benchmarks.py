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

from roughrax import LogODE, RoughTerm, SignatureInterpolation
from roughrax._bases import (
    PrimitiveBasis,
    make_lyndon_basis,
    make_planar_tree_basis,
    make_tree_basis,
)
from roughrax._pseudo_bialgebra_map import (
    form_pseudo_bialgebra_evaluator,
    form_pseudo_bialgebra_map,
)


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
class PseudoBialgebraEvalCase:
    name: str
    basis: PrimitiveBasis
    geometry: Manifold
    vector_field: Callable
    y: np.ndarray


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def so3_vector_field(y):
    return jnp.eye(3, dtype=y.dtype)


def sizeable_vector_field(y):
    dtype = y.dtype
    n = y.shape[0]
    rows = jnp.arange(1, 5, dtype=dtype)[:, None]
    cols = jnp.arange(1, n + 1, dtype=dtype)[None, :]
    phases = rows * cols / n
    mixed = jnp.tanh(jnp.sum(jnp.sin(y[None, :] + phases), axis=-1))
    return jnp.stack(
        [
            jnp.sin(y + mixed[0]) + 0.10 * y**2,
            jnp.cos(1.3 * y - mixed[1]) + 0.05 * jnp.sin(y) * y,
            jnp.tanh(y + mixed[2]) + 0.03 * jnp.cos(2.0 * y),
            jnp.sin(y * y + mixed[3]) - 0.02 * y,
        ]
    )


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
]


PSEUDO_BIALGEBRA_EVAL_CASES = [
    pytest.param(
        PseudoBialgebraEvalCase(
            name="lyndon-dim4-depth4-state64",
            basis=make_lyndon_basis(4, 4),
            geometry=Euclidean(),
            vector_field=sizeable_vector_field,
            y=np.linspace(-0.5, 0.5, 64),
        ),
        id="lyndon-dim4-depth4-state64",
    ),
    pytest.param(
        PseudoBialgebraEvalCase(
            name="tree-dim4-depth3-state64",
            basis=make_tree_basis(3, 4),
            geometry=Euclidean(),
            vector_field=sizeable_vector_field,
            y=np.linspace(-0.5, 0.5, 64),
        ),
        id="tree-dim4-depth3-state64",
    ),
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


@eqx.filter_jit
def _evaluate_lifted_fields_old(lifted_fields, y):
    return jnp.stack([field(y) for field in lifted_fields])


def evaluate_lifted_fields_old(lifted_fields, y):
    return jax.block_until_ready(_evaluate_lifted_fields_old(lifted_fields, y))


@eqx.filter_jit
def _evaluate_lifted_fields_new(evaluator, y):
    return evaluator(y)


def evaluate_lifted_fields_new(evaluator, y):
    return jax.block_until_ready(_evaluate_lifted_fields_new(evaluator, y))


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


@pytest.mark.benchmark(group="pseudo-bialgebra-map-eval-old-vs-new")
@pytest.mark.parametrize("case", PSEUDO_BIALGEBRA_EVAL_CASES)
@pytest.mark.parametrize("implementation", ["old", "new"])
def test_benchmark_pseudo_bialgebra_map_eval_old_vs_new(
    benchmark, case: PseudoBialgebraEvalCase, implementation: str
):
    y = jnp.asarray(case.y)
    if implementation == "old":
        lifted_fields = form_pseudo_bialgebra_map(
            case.vector_field, case.basis, case.geometry
        )
        evaluate_lifted_fields_old(lifted_fields, y)
        values = benchmark(evaluate_lifted_fields_old, lifted_fields, y)
    else:
        evaluator = form_pseudo_bialgebra_evaluator(
            case.vector_field, case.basis, case.geometry
        )
        evaluate_lifted_fields_new(evaluator, y)
        values = benchmark(evaluate_lifted_fields_new, evaluator, y)

    assert values.shape[0] == len(case.basis.keys)
    assert values.shape[1:] == y.shape


@pytest.mark.benchmark(group="log-ode")
@pytest.mark.parametrize("case", CASES)
def test_benchmark_log_ode_solve(benchmark, case: BenchmarkCase):
    solve_log_ode(case)
    y1 = benchmark(solve_log_ode, case)
    assert y1.shape == evaluation_state(case).shape


@pytest.mark.benchmark(group="rough-term")
@pytest.mark.parametrize("case", CASES)
def test_benchmark_make_rough_term(benchmark, case: BenchmarkCase):
    make_rough_term_coeffs(case)
    coeffs = benchmark(make_rough_term_coeffs, case)
    assert coeffs.shape[0] == len(case.coarse_ts) - 1
