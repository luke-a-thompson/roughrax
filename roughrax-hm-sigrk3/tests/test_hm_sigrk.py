from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from georax import Euclidean

from roughrax import HMSigRK3, RoughTerm, SignatureInterpolation


jax.config.update("jax_enable_x64", True)


_BASE_INCREMENTS = jnp.asarray(
    [
        [0.7, -0.2],
        [-0.4, 0.6],
        [0.2, 0.1],
    ],
    dtype=jnp.float64,
)


def _rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def _control_vector_field(t, y, args):
    del t, args
    return _rough_vector_field(y)


def _scaled_path(scale):
    increments = scale * _BASE_INCREMENTS
    return jnp.concatenate(
        [jnp.zeros((1, 2), dtype=increments.dtype), jnp.cumsum(increments, axis=0)]
    )


def _solve_hm(path, y0):
    fine_ts = jnp.linspace(0.0, 1.0, path.shape[0])
    signature_knots = jnp.asarray([0.0, 1.0], dtype=fine_ts.dtype)
    driver = diffrax.LinearInterpolation(ts=fine_ts, ys=path)
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth=3,
        solution="stratonovich",
    )
    term = RoughTerm(_rough_vector_field, control, Euclidean())
    sol = diffrax.diffeqsolve(
        term,
        HMSigRK3(),
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=4,
    )
    return sol.ys[-1]


def _solve_reference(path, y0):
    ts = jnp.linspace(0.0, 1.0, path.shape[0])
    term = diffrax.ControlTerm(
        _control_vector_field,
        diffrax.LinearInterpolation(ts=ts, ys=path),
    )
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Dopri8(),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(ts),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=ts.shape[0] + 4,
    )
    return sol.ys[-1]


def test_hm_sigrk3_has_fourth_order_local_remainder():
    y0 = jnp.asarray(0.25, dtype=jnp.float64)
    scales = (0.5, 0.25, 0.125)
    errors = []
    for scale in scales:
        path = _scaled_path(scale)
        errors.append(
            float(jnp.abs(_solve_hm(path, y0) - _solve_reference(path, y0)))
        )

    assert errors[0] / errors[1] > 10.0
    assert errors[1] / errors[2] > 10.0


def test_hm_sigrk3_is_filter_jit_safe():
    @eqx.filter_jit
    def solve(path, y0):
        return _solve_hm(path, y0)

    y1 = solve(_scaled_path(0.25), jnp.asarray(0.25, dtype=jnp.float64))
    assert y1.shape == ()
    assert jnp.isfinite(y1)


def test_hm_sigrk3_is_exact_for_constant_vector_fields():
    constant_fields = jnp.asarray([1.2, -0.7], dtype=jnp.float64)

    def vector_field(y):
        del y
        return constant_fields

    path = _scaled_path(0.5)
    fine_ts = jnp.linspace(0.0, 1.0, path.shape[0])
    signature_knots = jnp.asarray([0.0, 1.0], dtype=fine_ts.dtype)
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=fine_ts, ys=path),
        signature_knots,
        depth=3,
        solution="stratonovich",
    )
    term = RoughTerm(vector_field, control, Euclidean())
    y0 = jnp.asarray(-0.3, dtype=jnp.float64)
    sol = diffrax.diffeqsolve(
        term,
        HMSigRK3(),
        t0=signature_knots[0],
        t1=signature_knots[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=diffrax.StepTo(signature_knots),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=4,
    )
    expected = y0 + jnp.dot(path[-1] - path[0], constant_fields)
    assert jnp.allclose(sol.ys[-1], expected, rtol=1e-11, atol=1e-11)


def test_hm_sigrk3_rejects_insufficient_depth():
    path = _scaled_path(0.25)
    ts = jnp.linspace(0.0, 1.0, path.shape[0])
    knots = jnp.asarray([0.0, 1.0], dtype=ts.dtype)
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=path),
        knots,
        depth=2,
        solution="stratonovich",
    )
    term = RoughTerm(_rough_vector_field, control, Euclidean())
    with pytest.raises(ValueError, match="depth >= 3"):
        HMSigRK3().init(term, knots[0], knots[-1], jnp.asarray(0.25), None)


def test_hm_sigrk3_stage_count():
    assert HMSigRK3.num_stages(1) == 4
    assert HMSigRK3.num_stages(2) == 8
    assert HMSigRK3.num_stages(3) == 13
    with pytest.raises(ValueError, match="positive"):
        HMSigRK3.num_stages(0)


def test_method_one_log_coordinates_reconstruct_tensor_signature():
    import pysiglib.jax_api as pysiglib

    path = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.4],
            [-0.1, 0.5, 0.2],
            [0.4, 0.1, -0.3],
        ],
        dtype=jnp.float64,
    )
    ts = jnp.linspace(0.0, 1.0, path.shape[0])
    knots = jnp.asarray([0.0, 1.0], dtype=path.dtype)

    def vector_field(y):
        return jnp.stack(
            [
                jnp.cos(y),
                jnp.sin(y),
                1.0 + 0.25 * jnp.tanh(y),
            ]
        )

    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=path),
        knots,
        depth=3,
        solution="stratonovich",
    )
    term = RoughTerm(vector_field, control, Euclidean())

    from roughrax._solver.hm_sigrk import _tensor_signature_level_three

    levels = _tensor_signature_level_three(term, term.control.coeffs[0])
    reconstructed = jnp.concatenate(
        [levels[0].reshape(-1), levels[1].reshape(-1), levels[2].reshape(-1)]
    )
    direct = pysiglib.sig(path, degree=3, scalar_term=False)
    assert jnp.allclose(reconstructed, direct, rtol=1e-11, atol=1e-11)


def test_moment_completion_satisfies_order_three_conditions():
    from roughrax._solver.hm_sigrk import _moment_weights

    level_one = jnp.asarray([0.2, -0.3, 0.1], dtype=jnp.float64)
    level_two = jnp.asarray(
        [
            [0.04, 0.05, -0.02],
            [-0.01, 0.09, 0.03],
            [0.02, -0.04, 0.01],
        ],
        dtype=jnp.float64,
    )
    level_three = jnp.arange(27, dtype=jnp.float64).reshape(3, 3, 3)
    level_three = 0.001 * (level_three - 13.0)

    cloud, omega, rho, safe_rho, beta, safe_beta = _moment_weights(
        level_one, level_two, level_three
    )
    zeta = jnp.einsum("qlk->kl", level_three) / (
        safe_beta[:, None] * safe_rho**2
    )

    order_one = rho * jnp.sum(omega, axis=0) + beta
    order_two = rho**2 * jnp.einsum("ik,il->kl", omega, cloud)
    order_two = order_two + beta[:, None] * rho * zeta
    order_three_cherry = rho**3 * jnp.einsum(
        "ik,il,im->klm", omega, cloud, cloud
    )
    order_three_cherry = order_three_cherry + (
        beta[:, None, None]
        * rho**2
        * jnp.einsum("kl,km->klm", zeta, zeta)
    )
    carrier_coefficients = jnp.transpose(level_three, (2, 0, 1)) / (
        safe_beta[:, None, None] * safe_rho
    )
    order_three_chain = (
        beta[:, None, None]
        * rho
        * jnp.transpose(carrier_coefficients, (0, 2, 1))
    )

    expected_order_two = jnp.swapaxes(level_two, 0, 1)
    expected_chain = jnp.transpose(level_three, (2, 1, 0))
    expected_cherry = jnp.transpose(level_three, (2, 0, 1))
    expected_cherry = expected_cherry + jnp.transpose(
        level_three, (2, 1, 0)
    )

    assert jnp.allclose(order_one, level_one, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        order_two, expected_order_two, rtol=1e-12, atol=1e-12
    )
    assert jnp.allclose(
        order_three_chain, expected_chain, rtol=1e-12, atol=1e-12
    )
    assert jnp.allclose(
        order_three_cherry, expected_cherry, rtol=1e-12, atol=1e-12
    )
