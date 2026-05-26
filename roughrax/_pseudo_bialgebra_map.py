from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from georax import Manifold, post_lie_bracket
from jaxtyping import Array

from roughrax._bases import PrimitiveBasis


VectorField = Callable[[Array], Array]
LiftedField = Callable[[Array], Array]


def _covariant_derivative_along(
    geometry: Manifold[Any],
    direction: Array,
    field: LiftedField,
    x: Array,
) -> Array:
    return jax.jvp(field, (x,), (geometry.detrivialise(x, direction),))[1]


def _total_covariant_derivative(
    geometry: Manifold[Any],
    field: LiftedField,
    args: tuple[LiftedField, ...],
    x: Array,
) -> Array:
    """Evaluate (nabla^k field)(args[0], ..., args[k - 1])(x)."""

    out = field
    for arg in reversed(args):
        direction = arg(x)
        lower = out

        def out(
            z: Array,
            *,
            direction: Array = direction,
            lower: LiftedField = lower,
        ) -> Array:
            return _covariant_derivative_along(geometry, direction, lower, z)

    return out(x)


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


__all__ = [
    "LiftedField",
    "VectorField",
    "form_pseudo_bialgebra_map",
]
