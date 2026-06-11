from __future__ import annotations

import jax
import jax.numpy as jnp
from georax import SO, Euclidean
from georax._geometry.base import covariant_derivative, post_lie_bracket

from roughrax._bases import make_lyndon_basis
from roughrax._pseudo_bialgebra_map import (
    _total_covariant_derivative,
    form_pseudo_bialgebra_map,
)


def _reference_total_covariant_derivative(geometry, field, args, x):
    if not args:
        return field(x)

    first, *rest_list = args
    rest = tuple(rest_list)

    def lower(z):
        return _reference_total_covariant_derivative(geometry, field, rest, z)

    out = covariant_derivative(geometry, first, lower, x)
    for index, arg in enumerate(rest):

        def corrected_arg(z, *, first=first, arg=arg):
            return covariant_derivative(geometry, first, arg, z)

        corrected_rest = rest[:index] + (corrected_arg,) + rest[index + 1 :]
        out = out - _reference_total_covariant_derivative(
            geometry, field, corrected_rest, x
        )
    return out


def so3_frame(y):
    return jnp.eye(3, dtype=y.dtype)


def variable_so3_frame(y):
    return jnp.stack(
        [
            jnp.array([1.0 + y[0, 0], y[0, 1], 0.0], dtype=y.dtype),
            jnp.array([y[1, 0], 1.0 + y[1, 1], 0.0], dtype=y.dtype),
            jnp.array([0.0, y[2, 1], 1.0 + y[2, 2]], dtype=y.dtype),
        ]
    )


def test_total_covariant_derivative_matches_corrected_reference_under_jit():
    geometry = Euclidean()

    def root(y):
        return jnp.sin(y) + 0.1 * y**3

    def arg0(y):
        return jnp.cos(y) + 0.2

    def arg1(y):
        return y**2 - 0.3 * y

    def arg2(y):
        return jnp.sin(2.0 * y) - 0.1

    args = (arg0, arg1, arg2)
    x = jnp.asarray(0.37)

    actual = jax.jit(lambda y: _total_covariant_derivative(geometry, root, args, y))(x)
    expected = _reference_total_covariant_derivative(geometry, root, args, x)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_total_covariant_derivative_result_has_reference_derivative():
    geometry = Euclidean()

    def root(y):
        return jnp.sin(y) + 0.1 * y**3

    def arg0(y):
        return jnp.cos(y) + 0.2

    def arg1(y):
        return y**2 - 0.3 * y

    def direction(y):
        return jnp.sin(2.0 * y) + 0.4

    args = (arg0, arg1)
    x = jnp.asarray(0.37)

    def actual_field(y):
        return _total_covariant_derivative(geometry, root, args, y)

    def expected_field(y):
        return _reference_total_covariant_derivative(geometry, root, args, y)

    actual = jax.jit(
        lambda y: covariant_derivative(geometry, direction, actual_field, y)
    )(x)
    expected = covariant_derivative(geometry, direction, expected_field, x)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_total_covariant_derivative_matches_corrected_reference_on_so3():
    geometry = SO(3)

    def field(index):
        def out(y, *, index=index):
            return variable_so3_frame(y)[index]

        return out

    root = field(0)
    args = (field(1), field(2), field(0))
    x = jnp.asarray(
        [
            [1.0, 0.1, 0.2],
            [0.3, 1.0, 0.4],
            [0.5, 0.6, 1.0],
        ]
    )

    actual = jax.jit(lambda y: _total_covariant_derivative(geometry, root, args, y))(x)
    expected = _reference_total_covariant_derivative(geometry, root, args, x)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_so3_constant_frame_lyndon_brackets_match_frame_bracket():
    geometry = SO(3)
    basis = make_lyndon_basis(depth=2, dim=3)
    fields = form_pseudo_bialgebra_map(so3_frame, basis, geometry)

    values = jnp.stack([field(jnp.eye(3)) for field in fields])

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    assert jnp.allclose(values, expected)


def test_so3_nonconstant_lyndon_brackets_still_use_post_lie_bracket():
    geometry = SO(3)
    basis = make_lyndon_basis(depth=2, dim=3)
    fields = form_pseudo_bialgebra_map(variable_so3_frame, basis, geometry)
    x = jnp.asarray(
        [
            [1.0, 0.1, 0.2],
            [0.3, 1.0, 0.4],
            [0.5, 0.6, 1.0],
        ]
    )

    expected = post_lie_bracket(geometry, fields[0], fields[1], x)
    assert jnp.allclose(fields[3](x), expected)
