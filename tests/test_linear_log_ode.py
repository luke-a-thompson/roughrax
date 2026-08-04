from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest
from georax import Euclidean

from roughrax import LinearFer, LinearMagnus, RoughTerm
import roughrax._term as term_module
from roughrax._term import SignatureInterpolation

A0 = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
A1 = jnp.asarray([[0.2, -0.3], [0.4, 0.1]])
MATRICES = jnp.stack([A0, A1])


@pytest.fixture(autouse=True)
def force_pysiglib_cpu(monkeypatch):
    prepare_log_sig = term_module.pysiglib.prepare_log_sig

    def wrapped_prepare_log_sig(*args, **kwargs):
        kwargs["device"] = "cpu"
        return prepare_log_sig(*args, **kwargs)

    monkeypatch.setattr(
        term_module.pysiglib,
        "prepare_log_sig",
        wrapped_prepare_log_sig,
    )


class RightLinearVectorField:
    matrix_basis = MATRICES

    def __call__(self, y):
        return jnp.stack([y @ matrix for matrix in self.matrix_basis])


class LeftLinearVectorField:
    matrix_basis = MATRICES

    def __call__(self, y):
        return jnp.stack([matrix @ y for matrix in self.matrix_basis])


right_linear_vector_field = RightLinearVectorField()
left_linear_vector_field = LeftLinearVectorField()


def _driver(depth):
    ts = jnp.linspace(0.0, 1.0, 5)
    xs = jnp.asarray(
        [
            [0.0, 0.0],
            [0.3, 0.1],
            [0.1, 0.5],
            [0.7, -0.2],
            [0.4, 0.3],
        ]
    )
    signature_knots = jnp.asarray([0.0, 1.0])
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=xs),
        signature_knots,
        depth=depth,
        solution="stratonovich",
    )
    return control, signature_knots


def _solve(term, solver, signature_knots, y0):
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=4,
    )
    return sol.ys[-1]


def _commutator(left, right):
    return left @ right - right @ left


def _lyndon_matrix_basis(level_one, basis, side):
    matrices = [None] * len(basis.keys)

    def build(index):
        if matrices[index] is not None:
            return matrices[index]

        child_ids = basis.children[index]
        if not child_ids:
            matrix = level_one[basis.root_colour[index]]
        else:
            left = build(child_ids[0])
            right = build(child_ids[1])
            matrix = (
                _commutator(left, right)
                if side == "right"
                else _commutator(right, left)
            )
        matrices[index] = matrix
        return matrix

    return jnp.stack([build(index) for index in range(len(basis.keys))])


def _omega_components(coeffs, matrices, basis):
    degrees = jnp.asarray(basis.degree)
    return [
        jnp.tensordot(jnp.where(degrees == degree, coeffs, 0.0), matrices, axes=1)
        for degree in range(1, basis.depth + 1)
    ]


def test_linear_magnus_log_ode_matches_right_matrix_exponential():
    control, signature_knots = _driver(depth=3)
    term = RoughTerm(right_linear_vector_field, control, Euclidean())
    y0 = jnp.asarray([[1.0, 0.2], [-0.1, 0.8]])

    actual = _solve(term, LinearMagnus(side="right"), signature_knots, y0)

    matrices = _lyndon_matrix_basis(MATRICES, term.basis, "right")
    coeffs = term.contr(signature_knots[0], signature_knots[-1])
    omega = jnp.tensordot(coeffs, matrices, axes=1)
    expected = y0 @ jsl.expm(omega)

    assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_linear_magnus_log_ode_matches_left_matrix_exponential():
    control, signature_knots = _driver(depth=3)
    term = RoughTerm(left_linear_vector_field, control, Euclidean())
    y0 = jnp.asarray([[1.0, 0.2], [-0.1, 0.8]])

    actual = _solve(term, LinearMagnus(side="left"), signature_knots, y0)

    matrices = _lyndon_matrix_basis(MATRICES, term.basis, "left")
    coeffs = term.contr(signature_knots[0], signature_knots[-1])
    omega = jnp.tensordot(coeffs, matrices, axes=1)
    expected = jsl.expm(omega) @ y0

    assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_linear_fer_log_ode_matches_depth3_product():
    control, signature_knots = _driver(depth=3)
    term = RoughTerm(right_linear_vector_field, control, Euclidean())
    y0 = jnp.asarray([[0.8, -0.2], [0.3, 1.1]])

    actual = _solve(term, LinearFer(side="right"), signature_knots, y0)

    matrices = _lyndon_matrix_basis(MATRICES, term.basis, "right")
    components = _omega_components(
        term.contr(signature_knots[0], signature_knots[-1]), matrices, term.basis
    )
    f1 = components[0]
    f2 = components[1]
    f3 = components[2] - 0.5 * _commutator(components[0], components[1])
    expected = y0 @ jsl.expm(f1) @ jsl.expm(f2) @ jsl.expm(f3)

    assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_linear_solvers_require_level_one_matrix_basis():
    class FullBasisVectorField:
        matrix_basis = jnp.zeros((3, 2, 2))

        def __call__(self, y):
            return jnp.stack([y @ matrix for matrix in MATRICES])

    def missing_basis(y):
        return jnp.stack([y @ matrix for matrix in MATRICES])

    control, signature_knots = _driver(depth=2)
    for vector_field in (missing_basis, FullBasisVectorField()):
        term = RoughTerm(vector_field, control, Euclidean())
        with pytest.raises(ValueError, match="matrix_basis"):
            _solve(term, LinearMagnus(), signature_knots, jnp.eye(2))


@pytest.mark.parametrize(
    "solver",
    [LinearMagnus(side="right"), LinearFer(side="right")],
)
def test_precomputed_batched_logsignatures_match_materialised_controls(solver):
    sample_ts = jnp.linspace(0.0, 1.0, 5)
    signature_knots = sample_ts[::2]
    paths = jnp.asarray(
        [
            [
                [0.0, 0.0],
                [0.3, 0.1],
                [0.1, 0.5],
                [0.7, -0.2],
                [0.4, 0.3],
            ],
            [
                [0.0, 0.0],
                [-0.2, 0.4],
                [0.5, 0.2],
                [0.1, 0.8],
                [0.6, -0.1],
            ],
        ]
    )
    materialised_controls = [
        SignatureInterpolation(
            diffrax.LinearInterpolation(ts=sample_ts, ys=path),
            signature_knots,
            depth=3,
            solution="stratonovich",
        ).materialise(Euclidean())
        for path in paths
    ]
    assert all(control.coeffs is not None for control in materialised_controls)

    interval_batched_coeffs = jnp.stack(
        [control.coeffs for control in materialised_controls],
        axis=1,
    )
    precomputed_control = SignatureInterpolation.from_logsignatures(
        signature_knots,
        interval_batched_coeffs,
        input_dim=2,
        depth=3,
    )

    assert precomputed_control.basis is not None
    assert materialised_controls[0].basis is not None
    assert precomputed_control.basis.keys == materialised_controls[0].basis.keys

    vector_field = right_linear_vector_field
    y0 = jnp.asarray(
        [
            [[1.0, 0.2], [-0.1, 0.8]],
            [[0.8, -0.2], [0.3, 1.1]],
        ]
    )
    actual = _solve(
        RoughTerm(vector_field, precomputed_control, Euclidean()),
        solver,
        signature_knots,
        y0,
    )
    expected = jnp.stack(
        [
            _solve(
                RoughTerm(vector_field, control, Euclidean()),
                solver,
                signature_knots,
                initial,
            )
            for control, initial in zip(materialised_controls, y0, strict=True)
        ]
    )

    assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_linear_magnus_log_ode_runs_under_filter_jit():
    @eqx.filter_jit
    def solve(ts, xs, signature_knots, y0):
        control = SignatureInterpolation(
            diffrax.LinearInterpolation(ts=ts, ys=xs),
            signature_knots,
            depth=2,
            solution="stratonovich",
        )
        term = RoughTerm(right_linear_vector_field, control, Euclidean())
        return _solve(term, LinearMagnus(side="right"), signature_knots, y0)

    ts = jnp.linspace(0.0, 1.0, 5)
    xs = jnp.stack([jnp.sin(ts), jnp.cos(ts)], axis=-1)
    signature_knots = jnp.asarray([0.0, 1.0])
    y0 = jnp.eye(2)

    y1 = jax.block_until_ready(solve(ts, xs, signature_knots, y0))
    assert y1.shape == y0.shape


def test_linear_magnus_saveat_samples_exact_fake_time():
    control, signature_knots = _driver(depth=3)
    term = RoughTerm(right_linear_vector_field, control, Euclidean())
    y0 = jnp.asarray([[1.0, -0.1], [0.2, 0.9]])
    save_ts = jnp.linspace(signature_knots[0], signature_knots[-1], 5)

    sol = diffrax.diffeqsolve(
        term,
        LinearMagnus(side="right"),
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(ts=save_ts),
        max_steps=4,
    )

    matrices = _lyndon_matrix_basis(MATRICES, term.basis, "right")
    coeffs = term.contr(signature_knots[0], signature_knots[-1])
    omega = jnp.tensordot(coeffs, matrices, axes=1)
    fake_u = (save_ts - signature_knots[0]) / (signature_knots[-1] - signature_knots[0])
    expected = jnp.stack([y0 @ jsl.expm(u * omega) for u in fake_u])

    assert jnp.allclose(sol.ys, expected, atol=1e-6, rtol=1e-6)


def test_linear_fer_saveat_scales_the_full_generator():
    control, signature_knots = _driver(depth=3)
    term = RoughTerm(right_linear_vector_field, control, Euclidean())
    y0 = jnp.asarray([[0.7, 0.1], [-0.2, 1.2]])
    save_ts = jnp.linspace(signature_knots[0], signature_knots[-1], 4)

    sol = diffrax.diffeqsolve(
        term,
        LinearFer(side="right"),
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(ts=save_ts),
        max_steps=4,
    )

    matrices = _lyndon_matrix_basis(MATRICES, term.basis, "right")
    components = _omega_components(
        term.contr(signature_knots[0], signature_knots[-1]), matrices, term.basis
    )
    fake_u = (save_ts - signature_knots[0]) / (signature_knots[-1] - signature_knots[0])
    expected = []
    for u in fake_u:
        scaled = [u * component for component in components]
        factors = jnp.stack(
            [
                scaled[0],
                scaled[1],
                scaled[2] - 0.5 * _commutator(scaled[0], scaled[1]),
            ]
        )
        product = jnp.eye(factors.shape[-1], dtype=factors.dtype)
        for factor in factors:
            product = product @ jsl.expm(factor)
        expected.append(y0 @ product)
    expected = jnp.stack(expected)

    assert jnp.allclose(sol.ys, expected, atol=1e-6, rtol=1e-6)
