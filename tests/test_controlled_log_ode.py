from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from georax import Euclidean, RKMK, SO

from roughrax import ControlledVectorField, LogODE, RoughTerm, SignatureInterpolation


def solve(term, ts, y0, *, solver=None, args=None):
    return diffrax.diffeqsolve(
        term,
        LogODE(diffrax.Heun() if solver is None else solver),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.StepTo(ts),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=len(ts) + 4,
    ).ys


def test_scalar_controlled_integral_and_missing_correction():
    stride = 2
    ts = jnp.linspace(0, 1, 9)
    xs = jnp.array([0.7, 1.2, 0.2, 0.9, -0.4, 0.3, 0.1, 0.8, 0.4])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs[:, None])
    knots = ts[::stride]
    control = SignatureInterpolation.from_logsignatures(
        knots, jnp.diff(xs[::stride])[:, None], input_dim=1, depth=2
    )

    def vf(t, y, args):
        return driver.evaluate(t)

    def vf_prime(t, y, args):
        return jnp.ones((1, 1))

    exact = 0.5 * (xs[::stride] ** 2 - xs[0] ** 2)
    corrected = solve(
        RoughTerm(ControlledVectorField(vf, vf_prime), control), knots, jnp.array(0.0)
    )
    frozen = solve(
        RoughTerm(ControlledVectorField(vf, None), control), knots, jnp.array(0.0)
    )
    quadratic_variation = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(jnp.diff(xs[::stride]) ** 2)]
    )
    np.testing.assert_allclose(corrected, exact, atol=3e-7)
    np.testing.assert_allclose(frozen, exact - quadratic_variation / 2, atol=3e-7)


def test_higher_controlled_derivatives_with_jit_vmap_and_grad():
    depth = 4
    ts = jnp.array([0.0, 0.5, 1.0])
    xs = jnp.array([[0.4], [0.5], [0.7]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    coefficient = ControlledVectorField.from_driver(
        lambda x, y, scale: scale * x[:, None, None] ** (depth - 1) * jnp.ones_like(y),
        driver,
        depth=depth,
    )
    control = SignatureInterpolation.from_logsignatures(
        ts, jnp.diff(xs, axis=0), input_dim=1, depth=depth
    )
    term = RoughTerm(coefficient, control)

    @eqx.filter_jit
    def evaluate(term, scale):
        return solve(term, ts, jnp.zeros((2, 2)), solver=diffrax.Dopri5(), args=scale)[
            -1
        ]

    exact = (xs[-1, 0] ** depth - xs[0, 0] ** depth) / depth
    actual = jax.vmap(lambda scale: evaluate(term, scale))(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(
        actual,
        jnp.broadcast_to(jnp.array([exact, 2 * exact])[:, None, None], (2, 2, 2)),
        atol=2e-7,
    )
    gradient = jax.grad(lambda scale: evaluate(term, scale).sum())(jnp.array(1.0))
    np.testing.assert_allclose(gradient, 4 * exact, atol=3e-7)


def test_depth_three_keeps_nonsymmetric_area_dependence():
    # Z = integral X^1 dX^2 is not a function of the current X alone.
    # Around this rectangle, integral Z dX^1 = -a^2 b.
    a, b = 0.7, 0.4
    ts = jnp.linspace(0, 1, 5)
    xs = jnp.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    area = diffrax.LinearInterpolation(ts=ts, ys=jnp.array([0, 0, a * b, a * b, a * b]))

    def vf(t, y, args):
        return jnp.array([area.evaluate(t), 0.0])

    def first(t, y, args):
        return jnp.zeros((2, 2)).at[1, 0].set(driver.evaluate(t)[0])

    def second(t, y, args):
        return jnp.zeros((2, 2, 2)).at[0, 1, 0].set(1.0)

    knots = ts[jnp.array([0, 4])]
    control = SignatureInterpolation(driver, knots, depth=3, solution="stratonovich")
    term = RoughTerm(ControlledVectorField(vf, first, second), control)
    actual = solve(term, knots, jnp.array(0.0))[-1]
    np.testing.assert_allclose(actual, -(a**2) * b, atol=2e-7)


def test_from_driver_keeps_cross_channel_area():
    depth = 2
    ts = jnp.linspace(0, 1, 5)
    a, b = 0.7, 0.4
    xs = jnp.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    knots = ts[jnp.array([0, 4])]
    control = SignatureInterpolation(
        driver, knots, depth=depth, solution="stratonovich"
    )
    coefficient = ControlledVectorField.from_driver(
        lambda x, y, args: jnp.array([x[1], 0.0]), driver, depth=depth
    )
    actual = solve(RoughTerm(coefficient, control), knots, jnp.array(0.0))[-1]
    np.testing.assert_allclose(actual, -a * b, atol=2e-7)


def test_gradient_through_driver_values_and_signatures():
    ts = jnp.array([0.0, 0.5, 1.0])

    @jax.jit
    @jax.value_and_grad
    def terminal(xs):
        driver = diffrax.LinearInterpolation(ts=ts, ys=xs[:, None])
        coefficient = ControlledVectorField.from_driver(
            lambda x, y, args: x, driver, depth=2
        )
        control = SignatureInterpolation.from_logsignatures(
            ts, jnp.diff(xs)[:, None], input_dim=1, depth=2
        )
        return solve(RoughTerm(coefficient, control), ts, jnp.array(0.0))[-1]

    xs = jnp.array([0.3, 0.7, 0.5])
    value, gradient = terminal(xs)
    np.testing.assert_allclose(value, 0.5 * (xs[-1] ** 2 - xs[0] ** 2), atol=2e-7)
    np.testing.assert_allclose(gradient, jnp.array([-xs[0], 0.0, xs[-1]]), atol=2e-7)


def test_spatial_and_controlled_derivatives_both_contribute():
    ts = jnp.array([0.0, 1.0])
    xs = jnp.array([[0.3], [0.4]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    control = SignatureInterpolation.from_logsignatures(
        ts, jnp.diff(xs, axis=0), input_dim=1, depth=3
    )
    coefficient = ControlledVectorField.from_driver(
        lambda x, y, args: x * y, driver, depth=3
    )
    actual = solve(
        RoughTerm(coefficient, control), ts, jnp.array(1.0), solver=diffrax.Dopri5()
    )[-1]
    np.testing.assert_allclose(
        actual, jnp.exp(0.5 * (xs[-1, 0] ** 2 - xs[0, 0] ** 2)), atol=2e-7
    )


def test_controlled_manifold_uses_product_geometry():
    ts = jnp.array([0.0, 1.0])
    xs = jnp.array([[0.0], [0.3]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    control = SignatureInterpolation.from_logsignatures(
        ts, jnp.diff(xs, axis=0), input_dim=1, depth=2
    )
    axis = jnp.array([1.0, 0.0, 0.0])
    coefficient = ControlledVectorField.from_driver(
        lambda x, y, args: x[:, None] * axis, driver, depth=2
    )
    geometry = SO(3)
    actual = solve(
        RoughTerm(coefficient, control, geometry),
        ts,
        jnp.eye(3),
        solver=RKMK(diffrax.Heun()),
    )[-1]
    geometry.select_chart(2)
    expected = geometry.apply_increment(jnp.eye(3), 0.5 * xs[-1, 0] ** 2 * axis)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    np.testing.assert_allclose(actual.T @ actual, jnp.eye(3), atol=2e-7)


def test_backward_solve_refreshes_coefficients_at_physical_time():
    ts = jnp.array([0.0, 0.5, 1.0])
    xs = jnp.array([[0.3], [0.7], [0.5]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    coefficient = ControlledVectorField.from_driver(
        lambda x, y, args: x, driver, depth=2
    )
    control = SignatureInterpolation.from_logsignatures(
        ts, jnp.diff(xs, axis=0), input_dim=1, depth=2
    )
    actual = solve(RoughTerm(coefficient, control), ts[::-1], jnp.array(0.0))
    np.testing.assert_allclose(
        actual, 0.5 * (xs[::-1, 0] ** 2 - xs[-1, 0] ** 2), atol=2e-7
    )


def test_controlled_hierarchy_validation():
    ts = jnp.array([0.0, 1.0])
    control = SignatureInterpolation.from_logsignatures(
        ts, jnp.ones((1, 1)), input_dim=1, depth=3
    )
    with pytest.raises(ValueError, match="Depth 3 requires 3"):
        RoughTerm(ControlledVectorField(lambda t, y, args: jnp.ones(1), None), control)
    with pytest.raises(ValueError, match="derivative 1 must return shape"):
        term = RoughTerm(
            ControlledVectorField(
                lambda t, y, args: jnp.ones(1), lambda t, y, args: jnp.ones(1), None
            ),
            control,
        )
        term.vf(0.0, jnp.array(0.0), None)
    branched = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=ts[:, None]), ts, depth=2, solution="ito"
    )
    with pytest.raises(ValueError, match="geometric"):
        RoughTerm(
            ControlledVectorField(lambda t, y, args: jnp.ones(1), None),
            branched,
            Euclidean(),
        )
