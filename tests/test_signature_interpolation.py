from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pysiglib.jax_api as pysiglib
import pytest
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
        term = RoughTerm.from_lifted_vector_field(vector_field, control, Euclidean())
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


def test_signature_knots_must_match_regular_control_stride():
    ts = jnp.linspace(0.0, 1.0, 5)
    driver = diffrax.LinearInterpolation(ts=ts, ys=(ts**2)[:, None])
    control = SignatureInterpolation(
        driver,
        jnp.asarray([0.0, 0.75, 1.0]),
        depth=1,
        solution="stratonovich",
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match=r"control\.ts\[::stride\]"):
        control.materialise(Euclidean())


def test_signature_intervals_may_not_cross_knots():
    ts = jnp.asarray([0.0, 1.0, 2.0])
    control = SignatureInterpolation.from_logsignatures(
        ts,
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        input_dim=2,
        depth=2,
    )

    assert jnp.array_equal(control.evaluate(0.0, 1.0), control.coeffs[0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="may not cross"):
        control.evaluate(0.0, 2.0)


def test_generic_rough_term_rejects_batched_controls():
    control = SignatureInterpolation.from_logsignatures(
        jnp.asarray([0.0, 1.0]),
        jnp.ones((1, 2, 2)),
        input_dim=2,
        depth=1,
    )

    def vector_field(y):
        return jnp.stack([jnp.ones_like(y), 2 * jnp.ones_like(y)])

    term = RoughTerm(vector_field, control, Euclidean())
    y = jnp.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="jax.vmap"):
        term.prod(term.vf(0.0, y, None), term.contr(0.0, 1.0))


def test_ito_correction_is_forwarded_to_pysiglib():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.asarray([[0.0], [0.2], [-0.1], [0.4], [0.3]])
    signature_knots = ts[::2]
    correction = jnp.asarray([0.25], dtype=ys.dtype)
    windows = jnp.stack([ys[:3], ys[2:]])

    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=ys),
        signature_knots,
        depth=2,
        solution="ito",
        correction=correction,
    ).materialise(Euclidean())

    pysiglib.prepare_branched_sig(1, 2, planar=False)
    expected = pysiglib.branched_log_sig(
        windows,
        2,
        planar=False,
        correction=correction,
    )
    assert jnp.array_equal(control.correction, correction)
    assert jnp.allclose(control.coeffs, expected)


def test_signature_interpolation_rejects_stratonovich_correction():
    ts = jnp.asarray([0.0, 1.0])
    driver = diffrax.LinearInterpolation(ts=ts, ys=ts[:, None])

    with pytest.raises(ValueError, match="requires solution='ito'"):
        SignatureInterpolation(
            driver,
            ts,
            depth=2,
            solution="stratonovich",
            correction=jnp.asarray([1.0]),
        )


def test_from_logsignatures_matches_materialised_control():
    sample_ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.stack([sample_ts, sample_ts * 0.5], axis=-1)
    signature_knots = sample_ts[::2]
    materialised = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=sample_ts, ys=ys),
        signature_knots,
        depth=3,
        solution="stratonovich",
    ).materialise(Euclidean())
    assert materialised.coeffs is not None

    control = SignatureInterpolation.from_logsignatures(
        signature_knots,
        materialised.coeffs,
        input_dim=2,
        depth=3,
    )

    assert control.basis is not None
    assert materialised.basis is not None
    assert control.basis.keys == materialised.basis.keys
    for t0, t1 in zip(signature_knots[:-1], signature_knots[1:], strict=True):
        assert jnp.allclose(
            control.evaluate(t0, t1),
            materialised.evaluate(t0, t1),
        )


@pytest.mark.parametrize(
    ("ts", "coeffs", "message"),
    [
        (jnp.ones((2, 2)), jnp.ones((1, 3)), "ts must have shape"),
        (jnp.ones((1,)), jnp.ones((0, 3)), "at least two points"),
        (jnp.ones((3,)), jnp.ones((6,)), "coeffs must have shape"),
        (jnp.ones((3,)), jnp.ones((1, 3)), "first axis"),
        (jnp.ones((3,)), jnp.ones((2, 4)), "last axis"),
    ],
)
def test_from_logsignatures_validates_array_dimensions(ts, coeffs, message):
    with pytest.raises(ValueError, match=message):
        SignatureInterpolation.from_logsignatures(
            ts,
            coeffs,
            input_dim=2,
            depth=2,
        )


@pytest.mark.parametrize(("name", "value"), [("input_dim", 0), ("depth", 0)])
def test_from_logsignatures_requires_positive_integer_dimensions(name, value):
    kwargs = dict(input_dim=2, depth=2)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        SignatureInterpolation.from_logsignatures(
            jnp.asarray([0.0, 1.0]),
            jnp.ones((1, 3)),
            **kwargs,
        )


def test_from_logsignatures_rejects_ito_coefficients():
    with pytest.raises(ValueError, match="branched log-signatures"):
        SignatureInterpolation.from_logsignatures(
            jnp.asarray([0.0, 1.0]),
            jnp.ones((1, 3)),
            input_dim=2,
            depth=2,
            solution="ito",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "ts",
    [jnp.asarray([0.0, 0.5, 0.5]), jnp.asarray([0.0, 1.0, 0.5])],
)
def test_from_logsignatures_requires_strictly_increasing_ts(ts):
    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly increasing"):
        SignatureInterpolation.from_logsignatures(
            ts,
            jnp.ones((2, 3)),
            input_dim=2,
            depth=2,
        )


def test_from_logsignatures_is_filter_jit_and_vmap_safe():
    ts = jnp.asarray([0.0, 0.4, 1.0])
    coeffs_batch = jnp.asarray(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ]
    )

    @eqx.filter_jit
    def evaluate_population(ts, coeffs_batch):
        def evaluate_one(coeffs):
            control = SignatureInterpolation.from_logsignatures(
                ts,
                coeffs,
                input_dim=2,
                depth=2,
            )
            return jnp.stack(
                [control.evaluate(ts[i], ts[i + 1]) for i in range(ts.shape[0] - 1)]
            )

        return jax.vmap(evaluate_one)(coeffs_batch)

    @eqx.filter_jit
    def evaluate_batched_control(ts, coeffs):
        control = SignatureInterpolation.from_logsignatures(
            ts,
            coeffs,
            input_dim=2,
            depth=2,
        )
        return jnp.stack(
            [control.evaluate(ts[i], ts[i + 1]) for i in range(ts.shape[0] - 1)]
        )

    assert jnp.allclose(evaluate_population(ts, coeffs_batch), coeffs_batch)
    interval_batched_coeffs = jnp.swapaxes(coeffs_batch, 0, 1)
    assert jnp.allclose(
        evaluate_batched_control(ts, interval_batched_coeffs),
        interval_batched_coeffs,
    )
