from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest
from georax import SO, Euclidean

from roughrax._bases import make_lyndon_basis, make_planar_tree_basis, make_tree_basis
from roughrax._pseudo_bialgebra_map import (
    form_pseudo_bialgebra_evaluator,
    form_pseudo_bialgebra_map,
)


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def so3_vector_field(y):
    return jnp.eye(3, dtype=y.dtype)


@eqx.filter_jit
def evaluate_lifted_fields(lifted_fields, y):
    return jnp.stack([field(y) for field in lifted_fields])


@eqx.filter_jit
def evaluate_stacked_field(evaluator, y):
    return evaluator(y)


@pytest.mark.parametrize(
    "basis, geometry, vector_field, y",
    [
        (make_lyndon_basis(3, 2), Euclidean(), rough_vector_field, jnp.asarray(0.25)),
        (make_tree_basis(3, 2), Euclidean(), rough_vector_field, jnp.asarray(0.25)),
        (make_planar_tree_basis(2, 3), SO(3), so3_vector_field, jnp.eye(3)),
    ],
)
def test_stacked_pseudo_bialgebra_evaluator_matches_lifted_fields(
    basis, geometry, vector_field, y
):
    lifted_fields = form_pseudo_bialgebra_map(vector_field, basis, geometry)
    evaluator = form_pseudo_bialgebra_evaluator(vector_field, basis, geometry)

    expected = evaluate_lifted_fields(lifted_fields, y)
    actual = evaluate_stacked_field(evaluator, y)

    assert actual.shape == expected.shape
    assert jnp.allclose(actual, expected)
