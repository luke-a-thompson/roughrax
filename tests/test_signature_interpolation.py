from __future__ import annotations

import diffrax
import equinox as eqx
import jax.numpy as jnp
from georax import Euclidean

from roughrax import LogODE, RoughTerm, SignatureInterpolation


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def test_rough_term_accepts_direct_logsig_columns():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, ts * 0.5], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    ).materialise(Euclidean())

    def direct_columns(y):
        logsig_size = 3
        return jnp.arange(y.size * logsig_size, dtype=y.dtype).reshape(
            y.shape + (logsig_size,)
        )

    y = jnp.asarray([0.25, 0.5])
    coeffs = control.evaluate(signature_knots[0], signature_knots[1])
    columns = direct_columns(y)

    for vector_field in (direct_columns, lambda y: direct_columns(y).reshape(-1)):
        term = RoughTerm(vector_field, control, Euclidean())
        assert term.vf(0.0, y, None).shape == (coeffs.shape[0],) + y.shape
        assert jnp.allclose(
            term.prod(term.vf(0.0, y, None), coeffs),
            jnp.tensordot(columns, coeffs, axes=1),
        )


def test_signature_interpolation_construction_is_filter_jit_safe():
    @eqx.filter_jit
    def solve(ts, ys, signature_knots, y0):
        driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
        control = SignatureInterpolation(
            driver,
            signature_knots,
            depth=2,
            solution="stratonovich",
        )
        term = RoughTerm(rough_vector_field, control, Euclidean())
        sol = diffrax.diffeqsolve(
            term,
            LogODE(diffrax.Heun()),
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
    ys = jnp.stack([ts, ts * 0.5], axis=-1)
    y1 = solve(ts, ys, ts[::2], jnp.asarray(0.25))
    assert y1.shape == ()


def test_signature_interpolation_evaluates_linearly():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([ts, ts * 0.5], axis=-1)
    signature_knots = ts[::2]
    driver = diffrax.LinearInterpolation(ts=ts, ys=ys)
    control = SignatureInterpolation(
        driver,
        signature_knots,
        depth=2,
        solution="stratonovich",
    ).materialise(Euclidean())

    assert jnp.allclose(
        control.evaluate(signature_knots[0], signature_knots[1]), control.coeffs[0]
    )
    assert jnp.allclose(
        control.evaluate(
            signature_knots[0], 0.5 * (signature_knots[0] + signature_knots[1])
        ),
        0.5 * control.coeffs[0],
    )
