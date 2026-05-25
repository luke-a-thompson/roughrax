from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from georax import Manifold, covariant_derivative, post_lie_bracket
from jaxtyping import Array

from roughrax._bases import PrimitiveBasis


VectorField = Callable[[Array], Array]
LiftedField = Callable[[Array], Array]
StackedLiftedField = Callable[[Array], Array]
_ContextField = Callable[[Array, Array], Array]


def _total_covariant_derivative(
    geometry: Manifold[Any],
    field: LiftedField,
    args: tuple[LiftedField, ...],
    x: Array,
) -> Array:
    """Evaluate (nabla^k field)(args[0], ..., args[k - 1])(x)."""

    if not args:
        return field(x)

    first, *rest_list = args
    rest = tuple(rest_list)

    def lower(z: Array) -> Array:
        return _total_covariant_derivative(geometry, field, rest, z)

    out = covariant_derivative(geometry, first, lower, x)
    for index, arg in enumerate(rest):

        def corrected_arg(
            z: Array,
            *,
            first: LiftedField = first,
            arg: LiftedField = arg,
        ) -> Array:
            return covariant_derivative(geometry, first, arg, z)

        corrected_rest = rest[:index] + (corrected_arg,) + rest[index + 1 :]
        out = out - _total_covariant_derivative(geometry, field, corrected_rest, x)
    return out


def _contextualise(
    vector_field: VectorField,
    field: _ContextField,
    x: Array,
    vf: Array,
) -> LiftedField:
    def wrapped(z: Array) -> Array:
        return field(z, vf if z is x else vector_field(z))

    return wrapped


def _covariant_derivative_context(
    geometry: Manifold[Any],
    vector_field: VectorField,
    a_field: _ContextField,
    b_field: _ContextField,
    x: Array,
    vf: Array,
) -> Array:
    a = a_field(x, vf)
    b_fn = _contextualise(vector_field, b_field, x, vf)
    return jax.jvp(b_fn, (x,), (geometry.detrivialise(x, a),))[1]


def _post_lie_bracket_context(
    geometry: Manifold[Any],
    vector_field: VectorField,
    left: _ContextField,
    right: _ContextField,
    x: Array,
    vf: Array,
) -> Array:
    left_fn = _contextualise(vector_field, left, x, vf)
    right_fn = _contextualise(vector_field, right, x, vf)
    return post_lie_bracket(geometry, left_fn, right_fn, x)


def _total_covariant_derivative_context(
    geometry: Manifold[Any],
    vector_field: VectorField,
    field: _ContextField,
    args: tuple[_ContextField, ...],
    x: Array,
    vf: Array,
) -> Array:
    """Evaluate total covariant derivatives while reusing ``vector_field(x)``."""

    if not args:
        return field(x, vf)

    first, *rest_list = args
    rest = tuple(rest_list)

    def lower(z: Array) -> Array:
        z_vf = vf if z is x else vector_field(z)
        return _total_covariant_derivative_context(
            geometry, vector_field, field, rest, z, z_vf
        )

    a = first(x, vf)
    out = jax.jvp(lower, (x,), (geometry.detrivialise(x, a),))[1]
    for index, arg in enumerate(rest):

        def corrected_arg(
            z: Array,
            z_vf: Array,
            *,
            first: _ContextField = first,
            arg: _ContextField = arg,
        ) -> Array:
            return _covariant_derivative_context(
                geometry, vector_field, first, arg, z, z_vf
            )

        corrected_rest = rest[:index] + (corrected_arg,) + rest[index + 1 :]
        out = out - _total_covariant_derivative_context(
            geometry, vector_field, field, corrected_rest, x, vf
        )
    return out


def form_pseudo_bialgebra_map(
    vector_field: VectorField,
    basis: PrimitiveBasis,
    geometry: Manifold[Any],
) -> tuple[LiftedField, ...]:
    """Form basis vector fields for the pseudo-bialgebra map.

    ``vector_field(x)`` must return frame-coordinate vector fields stacked on
    leading axis, so ``vector_field(x)[i]`` is ``V_i(x)``. The returned tuple is
    aligned with ``basis.keys``.
    """

    lifted: list[LiftedField | None] = [None] * len(basis.keys)

    def build(index: int) -> LiftedField:
        cached = lifted[index]
        if cached is not None:
            return cached

        child_ids = basis.children[index]
        root_colour = basis.root_colour[index]
        if not child_ids:

            def field(x: Array, *, root_colour: int = root_colour) -> Array:
                return vector_field(x)[root_colour]

        elif basis.kind == "lyndon":
            if len(child_ids) != 2:
                raise ValueError("Lyndon basis entries must have two children.")
            left, right = build(child_ids[0]), build(child_ids[1])

            def field(
                x: Array,
                *,
                left: LiftedField = left,
                right: LiftedField = right,
            ) -> Array:
                return post_lie_bracket(geometry, left, right, x)

        else:
            child_fields = tuple(build(child_id) for child_id in child_ids)

            def root(x: Array, *, root_colour: int = root_colour) -> Array:
                return vector_field(x)[root_colour]

            def field(
                x: Array,
                *,
                root: LiftedField = root,
                child_fields: tuple[LiftedField, ...] = child_fields,
            ) -> Array:
                return _total_covariant_derivative(geometry, root, child_fields, x)

        lifted[index] = field
        return field

    return tuple(build(index) for index in range(len(basis.keys)))


def form_pseudo_bialgebra_evaluator(
    vector_field: VectorField,
    basis: PrimitiveBasis,
    geometry: Manifold[Any],
) -> StackedLiftedField:
    """Form a stacked evaluator for the pseudo-bialgebra map.

    This is equivalent to stacking ``form_pseudo_bialgebra_map(...)``, but gives
    one evaluation context a shared ``vector_field(x)`` value for direct root
    lookups. JVP callbacks still evaluate ``vector_field`` at their traced input.
    """

    fields: list[_ContextField | None] = [None] * len(basis.keys)

    def build(index: int) -> _ContextField:
        cached = fields[index]
        if cached is not None:
            return cached

        child_ids = basis.children[index]
        root_colour = basis.root_colour[index]
        if not child_ids:

            def field(
                x: Array,
                vf: Array,
                *,
                root_colour: int = root_colour,
            ) -> Array:
                del x
                return vf[root_colour]

        elif basis.kind == "lyndon":
            if len(child_ids) != 2:
                raise ValueError("Lyndon basis entries must have two children.")
            left, right = build(child_ids[0]), build(child_ids[1])

            def field(
                x: Array,
                vf: Array,
                *,
                left: _ContextField = left,
                right: _ContextField = right,
            ) -> Array:
                return _post_lie_bracket_context(
                    geometry, vector_field, left, right, x, vf
                )

        else:
            child_fields = tuple(build(child_id) for child_id in child_ids)

            def root(
                x: Array,
                vf: Array,
                *,
                root_colour: int = root_colour,
            ) -> Array:
                del x
                return vf[root_colour]

            def field(
                x: Array,
                vf: Array,
                *,
                root: _ContextField = root,
                child_fields: tuple[_ContextField, ...] = child_fields,
            ) -> Array:
                return _total_covariant_derivative_context(
                    geometry, vector_field, root, child_fields, x, vf
                )

        fields[index] = field
        return field

    context_fields = tuple(build(index) for index in range(len(basis.keys)))

    def evaluate(x: Array) -> Array:
        vf = vector_field(x)
        return jnp.stack([field(x, vf) for field in context_fields])

    return evaluate


__all__ = [
    "LiftedField",
    "StackedLiftedField",
    "VectorField",
    "form_pseudo_bialgebra_evaluator",
    "form_pseudo_bialgebra_map",
]
