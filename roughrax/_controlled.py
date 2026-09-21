from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractPath
from georax import Euclidean, LocalChart, Manifold


class _DriverCoefficients(eqx.Module):
    function: Callable
    driver: AbstractPath
    depth: int = eqx.field(static=True)

    def __len__(self):
        return self.depth

    def __getitem__(self, order):
        if not 0 <= order < self.depth:
            raise IndexError(order)

        def coefficient(t, y, args):
            def function(x):
                return self.function(x, y, args)

            for _ in range(order):
                function = jax.jacfwd(function)
            value = function(self.driver.evaluate(t))
            # jacfwd appends input axes; put chronological word axes first.
            axes = tuple(range(value.ndim - order, value.ndim))
            return jnp.moveaxis(value, axes, tuple(range(order)))

        return coefficient


class ControlledVectorField(eqx.Module):
    r"""A geometric controlled coefficient and its derivative hierarchy.

    ``ControlledVectorField(vf, vf_prime, vf_prime_prime, ...)`` takes callbacks
    ``(t, y, args)``. At depth N, supply N entries, including ``vf``. An explicit
    ``None`` derivative asserts that this level is zero.

    Entry k returns shape ``(driver_dim,) * (k + 1) + frame_shape``. The first k
    axes index a chronological signature word; the next axis selects the driving
    field. Thus the local coefficient is ``sum_k <V^(k), S^k>``. The callbacks
    must form a compatible controlled hierarchy with the required remainder and
    spatial regularity; this cannot be checked numerically.

    Local signature coordinates start at zero on every outer step. They evolve
    together with y through the same autonomous lift and inner ODE solver.
    """

    coefficients: tuple[Callable | None, ...] | _DriverCoefficients

    def __init__(
        self,
        vector_field: Callable | _DriverCoefficients,
        *derivatives: Callable | None,
    ):
        if isinstance(vector_field, _DriverCoefficients):
            self.coefficients = vector_field
            return
        if not callable(vector_field) or any(
            derivative is not None and not callable(derivative)
            for derivative in derivatives
        ):
            raise TypeError(
                "Controlled coefficients must be callable (or None derivatives)."
            )
        self.coefficients = (vector_field, *derivatives)

    @classmethod
    def from_driver(
        cls, function: Callable, driver: AbstractPath, *, depth: int
    ) -> ControlledVectorField:
        """Build the hierarchy for ``function(X_t, y, args)`` using autodiff.

        ``driver.evaluate(t)`` must return a vector of shape ``(driver_dim,)``.
        Its values must agree with the first level of the RoughTerm's control.
        The driver may have a nonzero initial value.
        """
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer.")
        # Keep a single copy of the function and driver in the parameter PyTree.
        return cls(_DriverCoefficients(function, driver, depth))

    def _augment(self, t, y, args, *, dim, depth, geometry):
        """Realise the local coefficient jet as autonomous augmented fields."""
        shape = jnp.shape(y)
        frame_shape = (
            shape if isinstance(geometry, Euclidean) else geometry.coordinate_shape
        )
        sizes = tuple(dim**k for k in range(1, depth))
        auxiliary_size = sum(sizes)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)

        def project(state):
            return state[auxiliary_size:].reshape(shape)

        def vector_field(state):
            current_y = project(state)
            value = jnp.asarray(self.coefficients[0](t, current_y, args))
            expected = (dim, *frame_shape)
            if value.shape != expected:
                raise ValueError(
                    f"Controlled vector field must return shape {expected}, got {value.shape}."
                )
            signature_fields = []
            previous_level = jnp.ones((1,), dtype=state.dtype)
            eye = jnp.eye(dim, dtype=state.dtype)
            for k, size in enumerate(sizes, start=1):
                level = state[offsets[k - 1] : offsets[k]]
                # dS^{w j} = S^w dX^j, including dS^j = dX^j.
                signature_fields.append(
                    (previous_level[None, :, None] * eye[:, None, :]).reshape(dim, size)
                )
                coefficient = self.coefficients[k]
                if coefficient is not None:
                    derivative = jnp.asarray(coefficient(t, current_y, args))
                    expected = (dim,) * (k + 1) + frame_shape
                    if derivative.shape != expected:
                        raise ValueError(
                            f"Controlled derivative {k} must return shape {expected}, got {derivative.shape}."
                        )
                    value = value + jnp.tensordot(
                        level, derivative.reshape((size, dim, *frame_shape)), axes=1
                    )
                previous_level = level
            return jnp.concatenate([*signature_fields, value.reshape(dim, -1)], axis=1)

        state = jnp.concatenate(
            [jnp.zeros(auxiliary_size, dtype=y.dtype), y.reshape(-1)]
        )
        augmented_geometry = (
            Euclidean()
            if isinstance(geometry, Euclidean)
            else _AugmentedGeometry(geometry, auxiliary_size, shape)
        )
        return vector_field, state, augmented_geometry, project


class _AugmentedChart(LocalChart):
    order: int

    def apply(self, x, a, geometry):
        n = geometry.auxiliary_size
        y = x[n:].reshape(geometry.base_shape)
        value = geometry.base.apply_increment(y, a[n:])
        return jnp.concatenate([x[:n] + a[:n], value.reshape(-1)])

    def inverse_differential(self, x, a, b, geometry):
        n = geometry.auxiliary_size
        y = x[n:].reshape(geometry.base_shape)
        value = geometry.base.chart.inverse_differential(y, a[n:], b[n:], geometry.base)
        return jnp.concatenate([b[:n], value.reshape(-1)])


class _AugmentedGeometry(Manifold):
    """Flat storage for the product of signature coordinates and a manifold."""

    base: Manifold[Any]
    auxiliary_size: int = eqx.field(static=True)
    base_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, base, auxiliary_size, base_shape):
        self.base = base
        self.auxiliary_size = auxiliary_size
        self.base_shape = base_shape

    @property
    def state_shape(self):
        return (self.auxiliary_size + prod(self.base_shape),)

    @property
    def coordinate_shape(self):
        return (self.auxiliary_size + prod(self.base.coordinate_shape),)

    def trivialise(self, x, v):
        n = self.auxiliary_size
        value = self.base.trivialise(
            x[n:].reshape(self.base_shape), v[n:].reshape(self.base_shape)
        )
        return jnp.concatenate([v[:n], value.reshape(-1)])

    def detrivialise(self, x, a):
        n = self.auxiliary_size
        value = self.base.detrivialise(x[n:].reshape(self.base_shape), a[n:])
        return jnp.concatenate([a[:n], value.reshape(-1)])

    def frame_bracket(self, x, a, b):
        n = self.auxiliary_size
        value = self.base.frame_bracket(x[n:].reshape(self.base_shape), a[n:], b[n:])
        return jnp.concatenate([jnp.zeros_like(a[:n]), value.reshape(-1)])

    def select_chart(self, required_order):
        chart = self.base.select_chart(required_order)
        object.__setattr__(self, "chart", _AugmentedChart(chart.order))
        return self.chart
