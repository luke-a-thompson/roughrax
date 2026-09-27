from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from georax import Euclidean, LocalChart, Manifold
from jaxtyping import Array

from roughrax._bases import CoefficientBasis
from roughrax._pseudo_bialgebra_map import (
    LiftedField,
    VectorField,
    _build_raw_fields,
    form_pseudo_bialgebra_map,
)


def _differentiate(field: LiftedField, direction: LiftedField) -> LiftedField:
    def derivative(x: Array) -> Array:
        return jax.jvp(field, (x,), (direction(x),))[1]

    return derivative


def rough_taylor_columns(
    vector_field: VectorField,
    basis: CoefficientBasis,
    geometry: Manifold[Any],
    chart: LocalChart[Any],
    y: Array,
) -> Array:
    """Realise full signature operators on coordinates anchored at y.

    MKW trees give frame-valued elementary differentials; a forest is the ordered product of their frozen frame actions. Pulling those actions through the chart retains the forest terms needed for a manifold Taylor update. BCK alone includes symmetry divisors.
    """
    zero = geometry.zero_coordinates(y)
    match basis.kind:
        case "word":
            fields: dict[tuple[int, ...], LiftedField] = {}
            # Words are ordered by degree, so both factors are already built.
            for word in basis.keys:
                if len(word) == 1:

                    def field(a: Array, colour: int = word[0]) -> Array:
                        point = chart.apply(y, a, geometry)
                        return chart.inverse_differential(
                            y, a, vector_field(point)[colour], geometry
                        )
                else:
                    field = _differentiate(fields[word[1:]], fields[word[:1]])
                fields[word] = field
            return jnp.stack([fields[word](zero) for word in basis.keys])

        case "tree":
            if not isinstance(geometry, Euclidean):
                raise ValueError("Manifold Itô controls require planar MKW signatures.")
            return jnp.stack(
                [
                    field(y)
                    for field in form_pseudo_bialgebra_map(
                        vector_field, basis, geometry
                    )
                ]
            )

        case "planar_tree":
            tree_indices = (
                i for i, colour in enumerate(basis.root_colour) if colour is not None
            )
            raw = _build_raw_fields(vector_field, basis, geometry, tree_indices)
            values = {i: field(y) for i, field in raw.items()}

            def frozen(value: Array) -> LiftedField:
                def field(a: Array) -> Array:
                    return chart.inverse_differential(y, a, value, geometry)

                return field

            columns: list[Array] = []
            for index, root in enumerate(basis.root_colour):
                if root is not None:
                    columns.append(values[index])
                else:
                    children = basis.children[index]
                    field = frozen(values[children[-1]])
                    for child in reversed(children[:-1]):
                        field = _differentiate(field, frozen(values[child]))
                    columns.append(field(zero))
            return jnp.stack(columns)

        case _:
            raise ValueError(
                f"Unsupported Taylor basis {basis.kind!r}; expected word, tree, or planar_tree."
            )
