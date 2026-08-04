from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from math import factorial, prod
from typing import Any

import jax
from georax import Manifold, post_lie_bracket
from jaxtyping import Array

from roughrax._bases import PrimitiveBasis


VectorField = Callable[[Array], Array]
LiftedField = Callable[[Array], Array]
_RawFields = dict[int, LiftedField]


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
    """Evaluate (nabla^k field)(args[0], ..., args[k - 1])(x).

    ``_covariant_derivative_along`` differentiates the lower-order field via
    ``jax.jvp``, so each step must keep it as a callable; the closure chain is
    load-bearing here (cf. ``_left_nested_frame_bracket``, which can fold values).
    """

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


def _left_nested_frame_bracket(
    geometry: Manifold[Any],
    fields: tuple[LiftedField, ...],
    x: Array,
) -> Array:
    """Evaluate [...[[fields[0], fields[1]], fields[2]], ...] in frame coords."""

    if len(fields) < 2:
        raise ValueError("At least two fields are required.")

    # frame_bracket only consumes tangent values at x, so fold values directly
    # rather than chaining closures (cf. _total_covariant_derivative).
    acc = fields[0](x)
    for right in fields[1:]:
        acc = geometry.frame_bracket(x, acc, right(x))
    return acc


def _tree_symmetry(tree: Any) -> int:
    if len(tree) == 1:
        return 1
    children = tree[:-1]
    return prod(_tree_symmetry(child) for child in children) * prod(
        factorial(count) for count in Counter(children).values()
    )


def _scale_field(field: LiftedField, scale: float) -> LiftedField:
    def scaled(x: Array, *, field: LiftedField = field, scale: float = scale) -> Array:
        return scale * field(x)

    return scaled


def _build_raw_fields(
    vector_field: VectorField,
    basis: PrimitiveBasis,
    geometry: Manifold[Any],
    indices: Iterable[int],
) -> _RawFields:
    raw_lifted: dict[int, LiftedField] = {}

    def build_raw(index: int) -> LiftedField:
        cached = raw_lifted.get(index)
        if cached is not None:
            return cached

        child_ids = basis.children[index]
        root_colour = basis.root_colour[index]
        if not child_ids:
            if root_colour is None:
                raise ValueError("Basis leaves must have a root colour.")

            def field(x: Array, *, root_colour: int = root_colour) -> Array:
                return vector_field(x)[root_colour]

        elif basis.kind == "lyndon":
            if len(child_ids) != 2:
                raise ValueError("Lyndon basis entries must have two children.")
            left, right = build_raw(child_ids[0]), build_raw(child_ids[1])

            def field(
                x: Array,
                *,
                left: LiftedField = left,
                right: LiftedField = right,
            ) -> Array:
                return post_lie_bracket(geometry, left, right, x)

        else:
            if root_colour is None:
                raise ValueError(
                    "Only rooted tree entries have elementary differentials."
                )
            child_fields = tuple(build_raw(child_id) for child_id in child_ids)

            def root(x: Array, *, root_colour: int = root_colour) -> Array:
                return vector_field(x)[root_colour]

            def field(
                x: Array,
                *,
                root: LiftedField = root,
                child_fields: tuple[LiftedField, ...] = child_fields,
            ) -> Array:
                return _total_covariant_derivative(geometry, root, child_fields, x)

        raw_lifted[index] = field
        return field

    for index in indices:
        build_raw(index)
    return raw_lifted


def _realise_bck(
    basis: PrimitiveBasis,
    raw_fields: _RawFields,
) -> tuple[LiftedField, ...]:
    return tuple(
        _scale_field(raw_fields[index], 1.0 / _tree_symmetry(tree))
        for index, tree in enumerate(basis.keys)
    )


def _realise_mkw(
    basis: PrimitiveBasis,
    raw_fields: _RawFields,
    geometry: Manifold[Any],
) -> tuple[LiftedField, ...]:
    realised: list[LiftedField] = []
    for index, (root_colour, child_ids) in enumerate(
        zip(basis.root_colour, basis.children, strict=True)
    ):
        if root_colour is not None:
            realised.append(raw_fields[index])
            continue

        child_fields = tuple(raw_fields[child_id] for child_id in child_ids)
        weight = basis.degree[child_ids[0]] / basis.degree[index]

        def field(
            x: Array,
            *,
            child_fields: tuple[LiftedField, ...] = child_fields,
            weight: float = weight,
        ) -> Array:
            return weight * _left_nested_frame_bracket(geometry, child_fields, x)

        realised.append(field)
    return tuple(realised)


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

    basis_indices = range(len(basis.keys))
    raw_indices: Iterable[int] = basis_indices
    if basis.kind == "planar_tree":
        raw_indices = (
            index
            for index, root_colour in enumerate(basis.root_colour)
            if root_colour is not None
        )
    raw_fields = _build_raw_fields(vector_field, basis, geometry, raw_indices)

    match basis.kind:
        case "lyndon":
            return tuple(raw_fields[index] for index in basis_indices)
        case "tree":
            return _realise_bck(basis, raw_fields)
        case "planar_tree":
            return _realise_mkw(basis, raw_fields, geometry)
    raise ValueError(f"Unknown basis kind {basis.kind!r}.")


__all__ = [
    "LiftedField",
    "VectorField",
    "form_pseudo_bialgebra_map",
]
