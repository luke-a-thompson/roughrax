from __future__ import annotations

import diffrax
import equinox as eqx
import jax.numpy as jnp
import pysiglib.jax_api as pysiglib
import pytest
from georax import Euclidean

from roughrax import LogSignatureInterpolation, RoughTerm


def test_rough_term_accepts_direct_logsig_columns():
    control = LogSignatureInterpolation.from_logsignatures(
        jnp.array([0.0, 1.0]), jnp.array([[0.3, -0.2, 0.4]]), input_dim=2, depth=2
    )

    def direct_columns(y):
        logsig_size = 3
        return jnp.arange(y.size * logsig_size, dtype=y.dtype).reshape(
            y.shape + (logsig_size,)
        )

    y = jnp.asarray([0.25, 0.5])
    coeffs = control.evaluate(0.0, 1.0)
    columns = direct_columns(y)
    expected = 0.3 * columns[:, 0] - 0.2 * columns[:, 1] + 0.4 * columns[:, 2]

    for vector_field in (direct_columns, lambda y: direct_columns(y).reshape(-1)):
        term = RoughTerm.from_lifted_vector_field(vector_field, control, Euclidean())
        assert jnp.allclose(
            term.prod(term.vf(0.0, y, None), coeffs),
            expected,
        )


def test_signature_interpolation_evaluates_linearly():
    control = LogSignatureInterpolation.from_logsignatures(
        jnp.array([0.0, 2.0, 5.0]),
        jnp.array([[1.0, -2.0, 0.3], [4.0, 1.0, -0.5]]),
        input_dim=2,
        depth=2,
    )
    assert jnp.allclose(control.evaluate(0.5, 1.5), jnp.array([0.5, -1.0, 0.15]))
    assert jnp.allclose(control.evaluate(2.75, 4.25), jnp.array([2.0, 0.5, -0.25]))
    assert jnp.allclose(control.evaluate(4.25, 2.75), jnp.array([-2.0, -0.5, 0.25]))
    assert jnp.allclose(control.evaluate(3.5), jnp.array([3.0, -1.5, 0.05]))


def test_signature_knots_must_match_regular_control_stride():
    ts = jnp.linspace(0.0, 1.0, 5)
    driver = diffrax.LinearInterpolation(ts=ts, ys=(ts**2)[:, None])
    control = LogSignatureInterpolation(
        driver,
        jnp.asarray([0.0, 0.75, 1.0]),
        depth=1,
        solution="stratonovich",
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match=r"control\.ts\[::stride\]"):
        control.materialise(Euclidean())


def test_signature_intervals_may_not_cross_knots():
    ts = jnp.asarray([0.0, 1.0, 2.0])
    control = LogSignatureInterpolation.from_logsignatures(
        ts,
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        input_dim=2,
        depth=2,
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="may not cross"):
        control.evaluate(0.0, 2.0)


@pytest.mark.parametrize(
    "times",
    [
        (-0.1,),
        (2.1,),
        (-0.1, 0.5),
        (0.5, -0.1),
        (1.5, 2.1),
        (2.1, 1.5),
        (-0.1, -0.1),
        (2.1, 2.1),
    ],
)
def test_log_signature_interpolation_rejects_out_of_range_times(times):
    control = LogSignatureInterpolation.from_logsignatures(
        jnp.array([0.0, 1.0, 2.0]), jnp.array([[1.0], [2.0]]), 1, 1
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="signature knot range"):
        control.evaluate(*times)


def test_log_signature_interpolation_accepts_boundary_times():
    control = LogSignatureInterpolation.from_logsignatures(
        jnp.array([0.0, 1.0, 2.0]), jnp.array([[1.0], [2.0]]), 1, 1
    )
    assert jnp.array_equal(control.evaluate(0.0), jnp.array([0.0]))
    assert jnp.array_equal(control.evaluate(2.0), jnp.array([3.0]))
    assert jnp.array_equal(control.evaluate(0.0, 1.0), jnp.array([1.0]))
    assert jnp.array_equal(control.evaluate(2.0, 1.0), jnp.array([-2.0]))
    for t in (0.0, 2.0):
        assert jnp.array_equal(control.evaluate(t, t), jnp.array([0.0]))


def test_ito_correction_is_forwarded_to_pysiglib():
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.asarray([[0.0], [0.2], [-0.1], [0.4], [0.3]])
    signature_knots = ts[::2]
    correction = jnp.asarray([0.25], dtype=ys.dtype)
    windows = jnp.stack([ys[:3], ys[2:]])

    control = LogSignatureInterpolation(
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
    assert jnp.allclose(control.coeffs, expected)


def test_signature_interpolation_rejects_stratonovich_correction():
    ts = jnp.asarray([0.0, 1.0])
    driver = diffrax.LinearInterpolation(ts=ts, ys=ts[:, None])

    with pytest.raises(ValueError, match="requires solution='ito'"):
        LogSignatureInterpolation(
            driver,
            ts,
            depth=2,
            solution="stratonovich",
            correction=jnp.asarray([1.0]),
        )


@pytest.mark.parametrize(
    "ts",
    [jnp.asarray([0.0, 0.5, 0.5]), jnp.asarray([0.0, 1.0, 0.5])],
)
def test_from_logsignatures_requires_strictly_increasing_ts(ts):
    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly increasing"):
        LogSignatureInterpolation.from_logsignatures(
            ts,
            jnp.ones((2, 3)),
            input_dim=2,
            depth=2,
        )


@pytest.mark.parametrize(
    "solution,planar", [("stratonovich", False), ("ito", False), ("ito", True)]
)
def test_full_signature_coefficients_and_inverse(solution, planar):
    import numpy as np
    from georax import SO

    from roughrax import SignatureInterpolation

    ts = jnp.array([0.0, 0.5, 1.0])
    xs = jnp.array([[0.0, 0.0], [0.2, -0.1], [0.1, 0.3]])
    knots = ts[jnp.array([0, 2])]
    geometry = SO(3) if planar else Euclidean()
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=xs), knots, 3, solution
    ).materialise(geometry)
    if solution == "stratonovich":
        expected = pysiglib.sig(xs, 3)

        def combine(a, b):
            return pysiglib.sig_combine(a, b, 2, 3)
    else:
        expected = pysiglib.branched_sig(xs, 3, planar=planar)

        def combine(a, b):
            return pysiglib.branched_sig_combine(a, b, 2, 3, planar=planar)

    forward = control.evaluate(0.0, 1.0)
    reverse = control.evaluate(1.0, 0.0)
    np.testing.assert_allclose(forward, expected, atol=1e-7)
    np.testing.assert_allclose(combine(forward, reverse), 0, atol=1e-7)
    np.testing.assert_allclose(combine(reverse, forward), 0, atol=1e-7)
    np.testing.assert_array_equal(control.evaluate(0.0, 0.0), jnp.zeros_like(forward))
    with pytest.raises(eqx.EquinoxRuntimeError, match="adjacent signature knots"):
        control.evaluate(0.0, 0.5)
