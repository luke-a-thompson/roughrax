from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest
from georax import Euclidean

from roughrax import LinearFer, RoughTerm
from roughrax._bases import make_lyndon_basis
from roughrax._solver._fer_coefficients import (
    FER_FACTORS,
    FER_MAX_DEPTH,
    LieWord,
)
from roughrax._solver.linear import _fer_factors
from roughrax._term import SignatureInterpolation


def _word_weight(word: LieWord) -> int:
    if isinstance(word, int):
        return word + 1
    return _word_weight(word[0]) + _word_weight(word[1])


def test_generated_fer_recipes_are_homogeneous_and_normalised():
    assert len(FER_FACTORS) == FER_MAX_DEPTH
    for depth, recipe in enumerate(FER_FACTORS, start=1):
        assert recipe[0] == (1, 1, depth - 1)
        assert all(_word_weight(word) == depth for _, _, word in recipe)


def _fer_product(components):
    product = jnp.eye(components.shape[-1], dtype=components.dtype)
    for factor in _fer_factors(list(components)):
        product = product @ jsl.expm(factor)
    return product


def test_depth6_fer_product_agrees_with_magnus_through_degree6():
    with jax.enable_x64():
        base_components = jnp.asarray(
            [
                [[0.0, 1.0], [-0.7, 0.2]],
                [[0.3, -0.4], [0.8, -0.1]],
                [[-0.2, 0.5], [0.1, 0.4]],
                [[0.6, 0.2], [-0.3, 0.1]],
                [[-0.1, 0.3], [0.4, -0.5]],
                [[0.2, -0.2], [0.7, 0.3]],
            ],
            dtype=jnp.float64,
        )

        def error(epsilon):
            powers = epsilon ** jnp.arange(1, FER_MAX_DEPTH + 1)
            components = powers[:, None, None] * base_components
            magnus = jsl.expm(jnp.sum(components, axis=0))
            return jnp.linalg.norm(_fer_product(components) - magnus)

        error_at_point_two = jax.block_until_ready(error(jnp.asarray(0.2)))
        error_at_point_one = jax.block_until_ready(error(jnp.asarray(0.1)))

        # The first omitted weighted degree is seven, so halving epsilon should
        # reduce the discrepancy by approximately 2**7.
        assert error_at_point_one < error_at_point_two / 100


class _RightLinearVectorField:
    matrices = jnp.asarray(
        [
            [[0.0, 1.0], [-1.0, 0.0]],
            [[0.2, -0.3], [0.4, 0.1]],
        ]
    )

    def __call__(self, y):
        return jnp.stack([y @ matrix for matrix in self.matrices])

    def matrix_basis(self):
        return self.matrices


def _solve_precomputed(depth):
    basis = make_lyndon_basis(depth, dim=2)
    ts = jnp.asarray([0.0, 1.0])
    coeffs = jnp.linspace(-0.03, 0.04, len(basis.keys))[None, :]
    control = SignatureInterpolation.from_logsignatures(
        ts,
        coeffs,
        input_dim=2,
        depth=depth,
    )
    term = RoughTerm(_RightLinearVectorField(), control, Euclidean())
    return diffrax.diffeqsolve(
        term,
        LinearFer(),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=jnp.eye(2),
        stepsize_controller=diffrax.StepTo(ts),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=2,
    ).ys[-1]


def test_linear_fer_supports_generated_maximum_depth():
    result = _solve_precomputed(FER_MAX_DEPTH)
    assert result.shape == (2, 2)
    assert jnp.isfinite(result).all()


def test_linear_fer_rejects_depth_beyond_generated_table():
    with pytest.raises(ValueError, match=rf"depth <= {FER_MAX_DEPTH}"):
        _solve_precomputed(FER_MAX_DEPTH + 1)
