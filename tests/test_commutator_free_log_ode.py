from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from georax import RKMK, SO, Euclidean

import roughrax._pseudo_bialgebra_map as pseudo_bialgebra_module
import roughrax._term as term_module
from roughrax import (
    CommutatorFreeLogODE2,
    LogODE,
    RoughTerm,
    SignatureInterpolation,
    VirtualPathInterpolation,
)
from roughrax._bases import make_lyndon_basis
from roughrax._virtual_path import (
    depth2_first_area_from_increments,
    depth2_logsig_to_first_area,
    pairwise_virtual_increments_depth2,
    spectral_virtual_increments_depth2,
    virtual_increments_depth2,
)


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def so3_vector_field(y):
    return jnp.eye(3, dtype=y.dtype)


def coupled_euclidean_vector_field(y):
    return jnp.stack(
        [
            jnp.asarray([0.3 + 0.2 * jnp.sin(y[0]), -0.1 + 0.1 * y[1] ** 2]),
            jnp.asarray([0.2 * jnp.cos(y[1]), 0.4 + 0.1 * y[0]]),
        ]
    )


def _deterministic_coeffs(basis):
    return jnp.linspace(-0.3, 0.4, len(basis.keys))


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 8])
def test_pairwise_factorisation_reconstructs_first_level_and_area(dim):
    basis = make_lyndon_basis(2, dim)
    coeffs = _deterministic_coeffs(basis)

    increments = pairwise_virtual_increments_depth2(coeffs, basis)
    actual_first, actual_area = depth2_first_area_from_increments(increments)
    expected_first, expected_area = depth2_logsig_to_first_area(coeffs, basis)

    assert jnp.allclose(actual_first, expected_first, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(actual_area, expected_area, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 8])
def test_spectral_factorisation_reconstructs_first_level_and_area(dim):
    basis = make_lyndon_basis(2, dim)
    coeffs = _deterministic_coeffs(basis)

    increments = spectral_virtual_increments_depth2(coeffs, basis)
    actual_first, actual_area = depth2_first_area_from_increments(increments)
    expected_first, expected_area = depth2_logsig_to_first_area(coeffs, basis)

    assert jnp.allclose(actual_first, expected_first, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(actual_area, expected_area, atol=1e-5, rtol=1e-5)


def test_virtual_factorisation_is_filter_jit_safe():
    dim = 5
    basis = make_lyndon_basis(2, dim)

    @eqx.filter_jit
    def factorise(coeffs):
        return virtual_increments_depth2(coeffs, basis)

    increments = factorise(_deterministic_coeffs(basis))
    assert increments.shape == (2 + 2 * (dim // 2), dim)


def test_commutator_free_solver_runs_under_filter_jit():
    @eqx.filter_jit
    def solve(ts, ys, signature_knots, y0):
        driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
        signature = SignatureInterpolation(
            driver,
            signature_knots,
            depth=2,
            solution="stratonovich",
        )
        control = VirtualPathInterpolation(signature)
        term = RoughTerm(rough_vector_field, control, Euclidean())
        sol = diffrax.diffeqsolve(
            term,
            CommutatorFreeLogODE2(diffrax.Heun()),
            t0=signature_knots[0],
            t1=signature_knots[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(signature_knots),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=signature_knots.shape[0] + 4,
        )
        return sol.ys[-1]

    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    y0 = jnp.asarray(0.25)

    y1 = solve(ts, ys, ts[::2], y0)
    assert y1.shape == y0.shape


def test_virtual_path_interpolation_precomputes_virtual_increments():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    signature = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    ).materialise(Euclidean())

    control = VirtualPathInterpolation(signature).materialise(Euclidean())
    assert control.virtual_increments_array is not None
    assert signature.coeffs is not None
    assert signature.basis is not None

    expected = jnp.stack(
        [
            spectral_virtual_increments_depth2(coeffs, signature.basis)
            for coeffs in signature.coeffs
        ]
    )
    assert control.virtual_increments_array.shape == (2, 4, 2)
    assert jnp.allclose(control.virtual_increments_array, expected)
    assert jnp.allclose(
        control.virtual_increments(signature_knots[0], signature_knots[1]),
        expected[0],
    )


def test_virtual_path_interpolation_rejects_off_grid_intervals():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    signature = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    )
    control = VirtualPathInterpolation(signature).materialise(Euclidean())

    with pytest.raises(Exception, match="exact knot-to-knot"):
        jax.block_until_ready(
            control.virtual_increments(
                signature_knots[0],
                0.5 * (signature_knots[0] + signature_knots[1]),
            )
        )


def test_commutator_free_solver_uses_precomputed_virtual_path_under_filter_jit():
    @eqx.filter_jit
    def solve(ts, ys, signature_knots, y0):
        driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
        signature = SignatureInterpolation(
            driver,
            signature_knots,
            depth=2,
            solution="stratonovich",
        )
        control = VirtualPathInterpolation(signature)
        term = RoughTerm(rough_vector_field, control, Euclidean())
        sol = diffrax.diffeqsolve(
            term,
            CommutatorFreeLogODE2(diffrax.Heun()),
            t0=signature_knots[0],
            t1=signature_knots[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(signature_knots),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=signature_knots.shape[0] + 4,
        )
        return sol.ys[-1]

    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    y0 = jnp.asarray(0.25)

    y1 = solve(ts, ys, ts[::2], y0)
    assert y1.shape == y0.shape


def test_commutator_free_solver_is_close_to_log_ode_on_euclidean_problem():
    ts = jnp.linspace(0.0, 1.0, 65)
    ys = jnp.stack(
        [
            0.4 * jnp.sin(2.0 * ts),
            0.3 * jnp.cos(3.0 * ts) + 0.1 * ts,
        ],
        axis=-1,
    )
    signature_knots = ts[::8]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    signature = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    )
    log_ode_term = RoughTerm(coupled_euclidean_vector_field, signature, Euclidean())
    commutator_free_term = RoughTerm(
        coupled_euclidean_vector_field,
        VirtualPathInterpolation(signature),
        Euclidean(),
    )

    def solve(term, solver):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=signature_knots[0],
            t1=signature_knots[-1],
            dt0=None,
            y0=jnp.asarray([0.25, -0.15]),
            stepsize_controller=diffrax.StepTo(signature_knots),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=signature_knots.shape[0] + 4,
        )
        return sol.ys[-1]

    log_ode_y1 = solve(log_ode_term, LogODE(diffrax.Heun()))
    commutator_free_y1 = solve(
        commutator_free_term,
        CommutatorFreeLogODE2(diffrax.Heun()),
    )

    assert jnp.linalg.norm(commutator_free_y1 - log_ode_y1) < 1e-4


def test_commutator_free_solver_runs_with_so3_frame_vector_field():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack(
        [
            0.2 * jnp.sin(2.0 * ts),
            0.15 * jnp.cos(3.0 * ts),
            0.1 * jnp.sin(5.0 * ts) + 0.05 * ts,
        ],
        axis=-1,
    )
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    signature = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    )
    log_ode_term = RoughTerm(so3_vector_field, signature, SO(3))
    commutator_free_term = RoughTerm(
        so3_vector_field,
        VirtualPathInterpolation(signature),
        SO(3),
    )

    def solve(term, solver):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=signature_knots[0],
            t1=signature_knots[-1],
            dt0=None,
            y0=jnp.eye(3),
            stepsize_controller=diffrax.StepTo(signature_knots),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=signature_knots.shape[0] + 4,
        )
        return sol.ys[-1]

    log_ode_y1 = solve(log_ode_term, LogODE(RKMK(diffrax.Heun())))
    y1 = solve(
        commutator_free_term,
        CommutatorFreeLogODE2(RKMK(diffrax.Heun())),
    )
    assert y1.shape == (3, 3)
    assert jnp.allclose(y1.T @ y1, jnp.eye(3), atol=1e-5, rtol=1e-5)
    assert jnp.linalg.norm(y1 - log_ode_y1) < 1e-3


def test_commutator_free_solver_matches_step2_nilpotent_sign_convention():
    basis = make_lyndon_basis(2, 2)
    a = jnp.asarray([0.3, -0.2])
    area_01 = jnp.asarray(0.17)
    coeffs = []
    for key in basis.keys:
        if len(key) == 1:
            coeffs.append(a[key[0]])
        elif key == (0, 1):
            coeffs.append(area_01)
        else:
            raise AssertionError(f"unexpected depth-2 key {key!r}")
    coeffs = jnp.stack(coeffs)

    def apply_segment(y, increment):
        x, z = y
        v1, v2 = increment
        return jnp.asarray([x + v1, z + v2 * x + 0.5 * v1 * v2])

    increments = spectral_virtual_increments_depth2(coeffs, basis)
    y0 = jnp.asarray([0.4, -0.1])
    y = y0
    for increment in increments:
        y = apply_segment(y, increment)

    expected = jnp.asarray(
        [
            y0[0] + a[0],
            y0[1] + a[1] * y0[0] + 0.5 * a[0] * a[1] + area_01,
        ]
    )
    assert jnp.allclose(y, expected, atol=1e-5, rtol=1e-5)


def test_commutator_free_solver_does_not_use_lifted_vf_during_step(monkeypatch):
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    signature = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    )
    control = VirtualPathInterpolation(signature)
    term = RoughTerm(rough_vector_field, control, Euclidean())

    def fail(*args, **kwargs):
        raise AssertionError("lifted vector-field route was called")

    monkeypatch.setattr(RoughTerm, "vf", fail)
    monkeypatch.setattr(RoughTerm, "vf_prod", fail)
    monkeypatch.setattr(term_module, "form_pseudo_bialgebra_map", fail)
    monkeypatch.setattr(pseudo_bialgebra_module, "form_pseudo_bialgebra_map", fail)

    y1, *_ = CommutatorFreeLogODE2(diffrax.Heun()).step(
        term,
        signature_knots[0],
        signature_knots[1],
        jnp.asarray(0.25),
        None,
        None,
        False,
    )
    assert y1.shape == ()


def test_commutator_free_solver_requires_virtual_path_interpolation():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    )
    term = RoughTerm(rough_vector_field, control, Euclidean())
    solver = CommutatorFreeLogODE2(diffrax.Heun())

    with pytest.raises(ValueError, match="VirtualPathInterpolation"):
        solver.step(
            term,
            signature_knots[0],
            signature_knots[1],
            jnp.asarray(0.25),
            None,
            None,
            False,
        )


@pytest.mark.parametrize(
    ("depth", "solution", "match"),
    [
        (2, "ito", "solution='stratonovich'"),
        (3, "stratonovich", "depth=2"),
    ],
)
def test_virtual_path_interpolation_unsupported_cases_raise(depth, solution, match):
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, 0.5 * ts], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    signature = SignatureInterpolation(
        driver,
        signature_knots,
        depth=depth,
        solution=solution,
    )

    with pytest.raises(ValueError, match=match):
        VirtualPathInterpolation(signature).materialise(Euclidean())


def test_commutator_free_solver_public_import():
    from roughrax import CommutatorFreeLogODE2

    assert CommutatorFreeLogODE2.__name__ == "CommutatorFreeLogODE2"
