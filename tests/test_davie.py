from itertools import product

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pysiglib
import pytest
from georax import SO

from roughrax import ControlledVectorField, Davie, RoughTerm, SignatureInterpolation


def scalar_control(ts, increments, input_dim, depth):
    assert input_dim == 1
    from math import factorial

    coeffs = jnp.concatenate(
        [increments**k / factorial(k) for k in range(1, depth + 1)], axis=-1
    )
    return SignatureInterpolation.from_signatures(ts, coeffs, input_dim, depth)


def solve(term, ts, y0, *, args=None, dense=False):
    return diffrax.diffeqsolve(
        term,
        Davie(),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.StepTo(ts),
        saveat=diffrax.SaveAt(ts=ts, dense=dense),
        max_steps=len(ts) + 2,
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_nonlinear_scalar_taylor_polynomial(depth):
    ts = jnp.array([0.0, 1.0])
    h, initial = 0.2, 0.7
    control = scalar_control(ts, jnp.array([[h]]), input_dim=1, depth=depth)
    # dy = y^2 dx has solution y / (1 - y dx).
    term = RoughTerm(lambda y: jnp.array([y**2]), control)
    sol = solve(term, ts, jnp.array(initial), dense=True)
    expected = sum(initial ** (k + 1) * h**k for k in range(depth + 1))
    np.testing.assert_allclose(sol.ys[-1], expected, atol=2e-7)
    np.testing.assert_allclose(sol.evaluate(0.5), (initial + expected) / 2, atol=2e-7)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_noncommuting_fields_against_full_signature(depth):
    ts = jnp.linspace(0, 1, 4)
    xs = jnp.array([[0.0, 0.0], [0.4, 0.1], [-0.1, 0.3], [0.2, 0.2]])
    matrices = np.array([[[0.1, 0.7], [-0.2, 0.3]], [[0.4, -0.1], [0.6, -0.3]]])
    initial = np.array([0.3, -0.2])
    knots = ts[jnp.array([0, 3])]
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=xs), knots, depth, "stratonovich"
    )
    term = RoughTerm(lambda y: jnp.einsum("ijk,k->ij", matrices, y), control)
    signature = pysiglib.sig(np.array(xs), depth)
    expected = initial.copy()
    index = 0
    for degree in range(1, depth + 1):
        for word in product(range(2), repeat=degree):
            value = initial
            for letter in word:
                value = matrices[letter] @ value
            expected += signature[index] * value
            index += 1
    np.testing.assert_allclose(
        solve(term, knots, jnp.asarray(initial)).ys[-1], expected, atol=2e-7
    )


@pytest.mark.parametrize("backward", [False, True])
def test_controlled_integral_and_zero_derivative(backward):
    ts = jnp.array([0.0, 0.5, 1.0])
    xs = jnp.array([[0.3], [0.7], [0.5]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    control = scalar_control(ts, jnp.diff(xs, axis=0), input_dim=1, depth=2)
    grid, values = (ts[::-1], xs[::-1, 0]) if backward else (ts, xs[:, 0])
    field = ControlledVectorField.from_driver(lambda x, y, args: x, driver, depth=2)
    expected = (values**2 - values[0] ** 2) / 2
    actual = solve(RoughTerm(field, control), grid, jnp.array(0.0)).ys
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    frozen = ControlledVectorField(lambda t, y, args: driver.evaluate(t), None)
    actual = solve(RoughTerm(frozen, control), grid, jnp.array(0.0)).ys
    quadratic_variation = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(jnp.diff(values) ** 2)]
    )
    np.testing.assert_allclose(actual, expected - quadratic_variation / 2, atol=2e-7)


def test_controlled_spatial_derivatives():
    ts = jnp.array([0.0, 1.0])
    x, h = 0.3, 0.2
    driver = diffrax.LinearInterpolation(ts=ts, ys=jnp.array([[x], [x + h]]))
    control = scalar_control(ts, jnp.array([[h]]), input_dim=1, depth=3)
    field = ControlledVectorField.from_driver(lambda x, y, args: x * y, driver, depth=3)
    # Degree-three truncation of exp(x h + h^2 / 2).
    expected = 1 + x * h + (1 + x**2) * h**2 / 2 + (3 * x + x**3) * h**3 / 6
    actual = solve(RoughTerm(field, control), ts, jnp.array(1.0)).ys[-1]
    np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_controlled_nonsymmetric_derivative_hierarchy():
    a, b = 0.7, 0.4
    ts = jnp.linspace(0, 1, 5)
    xs = jnp.array([[0, 0], [a, 0], [a, b], [0, b], [0, 0]])
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    knots = ts[jnp.array([0, 4])]
    control = SignatureInterpolation(driver, knots, 3, "stratonovich")
    # At the initial point of this single step, Z = integral X^1 dX^2
    # and its first derivative vanish; its (1, 2) derivative equals one.
    field = ControlledVectorField(
        lambda t, y, args: jnp.zeros(2),
        lambda t, y, args: jnp.zeros((2, 2)),
        lambda t, y, args: jnp.zeros((2, 2, 2)).at[0, 1, 0].set(1.0),
    )
    actual = solve(RoughTerm(field, control), knots, jnp.array(0.0)).ys[-1]
    np.testing.assert_allclose(actual, -(a**2) * b, atol=2e-7)


def test_controlled_jit_vmap_grad_matrix_state():
    ts = jnp.array([0.0, 0.5, 1.0])

    @jax.jit
    def terminal(xs, scale):
        driver = diffrax.LinearInterpolation(ts=ts, ys=xs[:, None])
        field = ControlledVectorField.from_driver(
            lambda x, y, args: args * x[:, None, None] ** 3 * jnp.ones_like(y),
            driver,
            depth=4,
        )
        control = scalar_control(ts, jnp.diff(xs)[:, None], input_dim=1, depth=4)
        return solve(RoughTerm(field, control), ts, jnp.zeros((2, 2)), args=scale).ys[
            -1
        ]

    xs = jnp.array([0.3, 0.7, 0.5])
    exact = (xs[-1] ** 4 - xs[0] ** 4) / 4
    actual = jax.vmap(lambda scale: terminal(xs, scale))(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(
        actual,
        jnp.broadcast_to(jnp.array([exact, 2 * exact])[:, None, None], (2, 2, 2)),
        atol=2e-7,
    )
    gradient = jax.grad(lambda xs: terminal(xs, 1.0).sum())(xs)
    np.testing.assert_allclose(
        gradient, 4 * jnp.array([-(xs[0] ** 3), 0.0, xs[-1] ** 3]), atol=3e-7
    )


def test_ito_milstein_correction():
    ts = jnp.array([0.0, 0.5, 1.0])
    xs = jnp.array([[0.0], [0.2], [0.1]])
    knots = ts[jnp.array([0, 2])]
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=xs),
        knots,
        2,
        "ito",
        # PySigLib adds this directly to the degree-two log coefficient.
        # The Itô lift needs -dt / 2 per segment, with dt = 0.5.
        correction=jnp.array([-0.25]),
    )
    term = RoughTerm(lambda y: jnp.array([y**2]), control)
    initial, increment = 0.7, 0.1
    expected = initial + initial**2 * increment + initial**3 * (increment**2 - 1)
    np.testing.assert_allclose(
        solve(term, knots, jnp.array(initial)).ys[-1], expected, atol=2e-7
    )


@pytest.mark.parametrize("depth", [3, 4])
def test_branched_taylor_against_full_signature(depth):
    from collections import Counter
    from math import factorial, prod

    ts = jnp.array([0.0, 0.5, 1.0])
    xs = np.array([[0.0], [0.2], [0.1]], dtype=np.float32)
    correction = np.array([-0.05], dtype=np.float32)
    knots = ts[jnp.array([0, 2])]
    control = SignatureInterpolation(
        diffrax.LinearInterpolation(ts=ts, ys=jnp.asarray(xs)),
        knots,
        depth,
        "ito",
        correction=jnp.asarray(correction),
    )
    term = RoughTerm(lambda y: jnp.array([y**2]), control)
    initial = 0.7

    def differential(tree):
        children = tree[:-1]
        derivative = (initial**2, 2 * initial, 2, 0, 0)[len(children)]
        return derivative * prod(differential(child) for child in children)

    def symmetry(tree):
        return prod(
            factorial(count) * symmetry(child) ** count
            for child, count in Counter(tree[:-1]).items()
        )

    signature = pysiglib.branched_sig(xs, depth, correction=correction)
    trees = [tree for tree in pysiglib.trees(1, depth) if tree is not None]
    expected = initial + sum(
        coefficient * differential(tree) / symmetry(tree)
        for coefficient, tree in zip(signature, trees, strict=True)
    )
    np.testing.assert_allclose(
        solve(term, knots, jnp.array(initial)).ys[-1], expected, atol=2e-7
    )


def test_solver_control_types():
    from roughrax import LogODE, LogSignatureInterpolation

    ts = jnp.array([0.0, 1.0])
    logged = LogSignatureInterpolation.from_logsignatures(ts, jnp.array([[0.1]]), 1, 2)
    full = scalar_control(ts, jnp.array([[0.1]]), 1, 2)

    def vf(y):
        return jnp.array([y])

    with pytest.raises(TypeError, match="requires SignatureInterpolation"):
        Davie().init(RoughTerm(vf, logged), 0.0, 1.0, jnp.array(1.0), None)
    with pytest.raises(TypeError, match="requires LogSignatureInterpolation"):
        LogODE(diffrax.Heun()).init(RoughTerm(vf, full), 0.0, 1.0, jnp.array(1.0), None)
    with pytest.raises(ValueError, match="Lifted vector fields"):
        RoughTerm.from_lifted_vector_field(vf, full)


@pytest.mark.parametrize("solution", ["stratonovich", "ito"])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_manifold_nonlinear_local_order(solution, depth):
    # Rotation angle solves theta' = 1 + theta + theta^2, theta(0)=0.
    # This exercises the bush, not only chains or constant frame fields.
    with jax.enable_x64():
        geometry = SO(2)
        ts = jnp.array([0.0, 1.0])

        def field(y):
            theta = jnp.arctan2(y[0, 1], y[0, 0])
            return jnp.array([[1 + theta + theta**2]])

        errors = []
        for h in (0.12, 0.06):
            driver = diffrax.LinearInterpolation(ts=ts, ys=jnp.array([[0.0], [h]]))
            control = SignatureInterpolation(driver, ts, depth, solution)
            term = RoughTerm(field, control, geometry)
            actual = solve(term, ts, jnp.eye(2), dense=True)
            theta = (
                jnp.sqrt(3.0) * jnp.tan(jnp.sqrt(3.0) * h / 2 + jnp.pi / 6) - 1
            ) / 2
            expected = jnp.array(
                [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]]
            )
            errors.append(float(jnp.linalg.norm(actual.ys[-1] - expected)))
            middle = actual.evaluate(0.5)
            np.testing.assert_allclose(middle.T @ middle, jnp.eye(2), atol=2e-12)
            np.testing.assert_allclose(
                actual.ys[-1].T @ actual.ys[-1], jnp.eye(2), atol=2e-12
            )
        assert errors[0] / errors[1] > 2 ** (depth + 0.5)


@pytest.mark.parametrize("solution", ["stratonovich", "ito"])
def test_manifold_noncommuting_fields(solution):
    with jax.enable_x64():
        geometry = SO(3)
        ts = jnp.array([0.0, 0.5, 1.0])
        knots = ts[jnp.array([0, 2])]
        axes = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        def field(y):
            return axes

        errors = []
        for h in (0.16, 0.08):
            xs = jnp.array([[0.0, 0.0], [h, 0.0], [h, h]])
            control = SignatureInterpolation(
                diffrax.LinearInterpolation(ts=ts, ys=xs), knots, 3, solution
            )
            actual = solve(RoughTerm(field, control, geometry), knots, jnp.eye(3)).ys[
                -1
            ]
            from jax.scipy.linalg import expm

            expected = expm(geometry._coords_to_alg(h * axes[0])) @ expm(
                geometry._coords_to_alg(h * axes[1])
            )
            errors.append(float(jnp.linalg.norm(actual - expected)))
            np.testing.assert_allclose(actual.T @ actual, jnp.eye(3), atol=2e-12)
        assert errors[0] / errors[1] > 12


def test_controlled_manifold_jit_grad():
    with jax.enable_x64():
        ts = jnp.array([0.0, 1.0])
        geometry = SO(2)

        @jax.jit
        def terminal(h):
            driver = diffrax.LinearInterpolation(
                ts=ts, ys=jnp.array([[0.3], [0.3 + h]])
            )
            field = ControlledVectorField.from_driver(
                lambda x, y, args: x[:, None], driver, depth=3
            )
            control = SignatureInterpolation(driver, ts, 3, "stratonovich")
            return solve(
                RoughTerm(field, control, geometry), ts, jnp.eye(2), dense=True
            ).ys[-1]

        h = 0.05
        actual = terminal(h)
        np.testing.assert_allclose(actual.T @ actual, jnp.eye(2), atol=2e-12)
        derivative = jax.grad(lambda h: terminal(h)[0, 1])(h)
        eps = 1e-5
        finite_difference = (terminal(h + eps)[0, 1] - terminal(h - eps)[0, 1]) / (
            2 * eps
        )
        np.testing.assert_allclose(derivative, finite_difference, atol=1e-9)
        np.testing.assert_allclose(actual[0, 1], jnp.sin(0.3 * h + h**2 / 2), atol=1e-7)


def test_mkw_matches_tensor_for_state_dependent_noncommuting_fields():
    with jax.enable_x64():
        ts = jnp.array([0.0, 0.5, 1.0])
        knots = ts[jnp.array([0, 2])]
        xs = jnp.array([[0.0, 0.0], [0.1, -0.04], [0.06, 0.13]])
        driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
        geometry = SO(3)

        def field(y):
            return jnp.array(
                [[1 + y[0, 1], 0.2 * y[2, 1], 0.1], [0.2 * y[1, 2], 0.1, 1 - y[1, 2]]]
            )

        values = []
        for solution in ("stratonovich", "ito"):
            control = SignatureInterpolation(driver, knots, 3, solution)
            values.append(
                solve(RoughTerm(field, control, geometry), knots, jnp.eye(3)).ys[-1]
            )
        np.testing.assert_allclose(values[0], values[1], atol=2e-12)


def test_manifold_nonzero_branched_correction_order():
    from scipy.integrate import solve_ivp

    with jax.enable_x64():
        ts = jnp.array([0.0, 1.0])
        geometry = SO(2)

        def field(y):
            theta = jnp.arctan2(y[0, 1], y[0, 0])
            return jnp.array([[1 + theta + theta**2]])

        errors = []
        for h in (0.12, 0.06):
            correction = -0.2 * h**2
            driver = diffrax.LinearInterpolation(ts=ts, ys=jnp.array([[0.0], [h]]))
            control = SignatureInterpolation(
                driver, ts, 3, "ito", correction=jnp.array([correction])
            )
            actual = solve(RoughTerm(field, control, geometry), ts, jnp.eye(2)).ys[-1]
            reference = solve_ivp(
                lambda t, y: (h + correction * (1 + 2 * y)) * (1 + y + y**2),
                (0, 1),
                [0.0],
                rtol=1e-12,
                atol=1e-14,
            ).y[0, -1]
            errors.append(
                abs(float(jnp.arctan2(actual[0, 1], actual[0, 0])) - reference)
            )
        assert errors[0] / errors[1] > 12


@pytest.mark.parametrize("kind", ["spd", "sphere"])
def test_other_manifold_charts(kind):
    from georax import SPD, Sphere
    from jax.scipy.linalg import expm

    with jax.enable_x64():
        geometry = SPD(2) if kind == "spd" else Sphere(3)
        y0 = jnp.eye(2) if kind == "spd" else jnp.array([1.0, 0.0, 0.0])
        axis = jnp.array([0.3, -0.2, 0.1])
        ts = jnp.array([0.0, 1.0])

        def field(y):
            return axis[None, :]

        errors = []
        for h in (0.2, 0.1):
            control = scalar_control(ts, jnp.array([[h]]), 1, 2)
            result = solve(RoughTerm(field, control, geometry), ts, y0, dense=True)
            actual = result.ys[-1]
            if kind == "spd":
                g = expm(h * geometry._coords_to_sym(axis))
                expected = g @ y0 @ g.T
                assert jnp.linalg.eigvalsh(result.evaluate(0.5)).min() > 0
                np.testing.assert_allclose(actual, actual.T, atol=1e-12)
            else:
                expected = expm(h * geometry._coords_to_alg(axis)) @ y0
                np.testing.assert_allclose(
                    jnp.linalg.norm(result.evaluate(0.5)), 1, atol=1e-12
                )
                np.testing.assert_allclose(jnp.linalg.norm(actual), 1, atol=1e-12)
            errors.append(float(jnp.linalg.norm(actual - expected)))
        assert errors[0] / errors[1] > 6
