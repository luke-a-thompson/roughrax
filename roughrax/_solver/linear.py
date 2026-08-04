from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from diffrax import AbstractLocalInterpolation, AbstractSolver, RESULTS
from jaxtyping import Array

from roughrax._bases import PrimitiveBasis
from roughrax._solver._fer_coefficients import FER_FACTORS, FER_MAX_DEPTH, LieWord
from roughrax._term import RoughTerm, unwrap_rough_term

Side = Literal["right", "left"]


def _matrix_commutator(a: Array, b: Array) -> Array:
    return a @ b - b @ a


def _check_linear_rough_term(rough_term: RoughTerm) -> None:
    if rough_term.control.solution != "stratonovich":
        raise ValueError("Linear solvers require solution='stratonovich'.")
    if rough_term.basis.kind != "lyndon":
        raise ValueError("Linear solvers require a Lyndon basis.")


def _build_lyndon_matrix_basis(
    level_one: Array,
    basis: PrimitiveBasis,
    side: Side,
) -> Array:
    matrices: list[Array | None] = [None] * len(basis.keys)

    def build(index: int) -> Array:
        matrix = matrices[index]
        if matrix is not None:
            return matrix

        child_ids = basis.children[index]
        root_colour = basis.root_colour[index]
        if not child_ids:
            matrix = level_one[root_colour]
        else:
            if len(child_ids) != 2:
                raise ValueError("Lyndon basis entries must have two children.")
            left = build(child_ids[0])
            right = build(child_ids[1])
            matrix = (
                _matrix_commutator(left, right)
                if side == "right"
                else _matrix_commutator(right, left)
            )
        matrices[index] = matrix
        return matrix

    return jnp.stack([build(index) for index in range(len(basis.keys))])


def _infer_level_one_matrices(rough_term: RoughTerm, y0: Array) -> Array:
    if y0.ndim != 2 or y0.shape[0] != y0.shape[1]:
        raise ValueError(
            "Linear solvers can infer matrices only from square matrix states. "
            "For other state shapes, attach a `matrix_basis` array to the "
            "vector field."
        )
    identity = jnp.eye(y0.shape[0], dtype=y0.dtype)
    values = jnp.asarray(rough_term.vector_field(identity), dtype=y0.dtype)
    if values.shape == (rough_term.basis.dim, *identity.shape):
        return values
    if values.shape == (*identity.shape, rough_term.basis.dim):
        return jnp.moveaxis(values, -1, 0)
    raise ValueError(
        "Could not infer level-one matrices from vector_field(I). Expected shape "
        f"{(rough_term.basis.dim, *identity.shape)} or "
        f"{(*identity.shape, rough_term.basis.dim)}, got {values.shape}."
    )


def _matrix_basis(rough_term: RoughTerm, y0: Array, side: Side) -> Array:
    matrix_basis = getattr(rough_term.vector_field, "matrix_basis", None)
    if matrix_basis is None:
        matrices = _infer_level_one_matrices(rough_term, y0)
    else:
        matrices = matrix_basis() if callable(matrix_basis) else matrix_basis
        matrices = jnp.asarray(matrices, dtype=y0.dtype)

    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "matrix_basis must have shape (basis_size, matrix_dim, matrix_dim), "
            f"got {matrices.shape}."
        )
    if matrices.shape[0] == rough_term.basis.dim:
        return _build_lyndon_matrix_basis(matrices, rough_term.basis, side)
    if matrices.shape[0] == len(rough_term.basis.keys):
        return matrices
    raise ValueError(
        "matrix_basis leading dimension must be either the driver dimension "
        f"{rough_term.basis.dim} or the log-signature dimension "
        f"{len(rough_term.basis.keys)}, got {matrices.shape[0]}."
    )


def _contract(coeffs: Array, matrices: Array) -> Array:
    return jnp.tensordot(coeffs, matrices, axes=1)


def _apply_generator(y: Array, generator: Array, side: Side) -> Array:
    exp_generator = jsl.expm(generator)
    return y @ exp_generator if side == "right" else exp_generator @ y


def _apply_generator_from_bool(y: Array, generator: Array, right: Array) -> Array:
    exp_generator = jsl.expm(generator)
    return jax.lax.cond(
        right,
        lambda: y @ exp_generator,
        lambda: exp_generator @ y,
    )


class _LinearMagnusInterpolation(AbstractLocalInterpolation):
    t0: Array
    t1: Array
    y0: Array
    omega: Array
    right: Array

    def evaluate(self, t0, t1=None, left: bool = True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        u = (t0 - self.t0) / (self.t1 - self.t0)
        return _apply_generator_from_bool(self.y0, u * self.omega, self.right)


def _apply_fer_factors(y0: Array, factors: Array, u: Array, right: Array) -> Array:
    num_factors = factors.shape[0]
    progress = u * num_factors
    fractions = jnp.clip(progress - jnp.arange(num_factors), 0.0, 1.0)
    eye = jnp.eye(factors.shape[-1], dtype=factors.dtype)
    product = eye
    for fraction, factor in zip(fractions, factors, strict=True):
        product = product @ jsl.expm(fraction * factor)
    return jax.lax.cond(
        right,
        lambda: y0 @ product,
        lambda: product @ y0,
    )


class _LinearFerInterpolation(AbstractLocalInterpolation):
    t0: Array
    t1: Array
    y0: Array
    factors: Array
    right: Array

    def evaluate(self, t0, t1=None, left: bool = True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        u = (t0 - self.t0) / (self.t1 - self.t0)
        return _apply_fer_factors(self.y0, self.factors, u, self.right)


def _degree_components(
    coeffs: Array,
    matrices: Array,
    basis: PrimitiveBasis,
) -> list[Array]:
    return [
        _contract(jnp.where(jnp.asarray(basis.degree) == degree, coeffs, 0.0), matrices)
        for degree in range(1, basis.depth + 1)
    ]


def _fer_factors(components: list[Array]) -> Array:
    values: dict[LieWord, Array] = {
        index: component for index, component in enumerate(components)
    }

    def evaluate(word: LieWord) -> Array:
        value = values.get(word)
        if value is None:
            if isinstance(word, int):
                raise ValueError(f"Fer component index {word} is unavailable.")
            value = _matrix_commutator(evaluate(word[0]), evaluate(word[1]))
            values[word] = value
        return value

    factors = []
    for recipe in FER_FACTORS[: len(components)]:
        factor = jnp.zeros_like(components[0])
        for numerator, denominator, word in recipe:
            value = evaluate(word)
            coefficient = jnp.asarray(numerator, dtype=value.dtype) / denominator
            factor = factor + coefficient * value
        factors.append(factor)
    return jnp.stack(factors)


class LinearMagnus(AbstractSolver[None]):
    """Linear RDE solver using one matrix Magnus exponential."""

    term_structure = RoughTerm
    interpolation_cls = _LinearMagnusInterpolation

    side: Side = eqx.field(static=True)

    def __init__(self, *, side: Side = "right"):
        if side not in {"right", "left"}:
            raise ValueError("side must be one of {'right', 'left'}.")
        object.__setattr__(self, "side", side)

    def init(self, terms, t0, t1, y0, args) -> None:
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del args, solver_state, made_jump
        rough_term = unwrap_rough_term(terms)
        _check_linear_rough_term(rough_term)

        matrices = _matrix_basis(rough_term, y0, self.side)
        omega = _contract(terms.contr(t0, t1), matrices)
        y1 = _apply_generator(y0, omega, self.side)
        dense_info = dict(
            y0=y0,
            omega=omega,
            right=jnp.asarray(self.side == "right"),
        )
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class LinearFer(AbstractSolver[None]):
    """Linear RDE solver using a truncated Fer product through depth 6."""

    term_structure = RoughTerm
    interpolation_cls = _LinearFerInterpolation

    side: Side = eqx.field(static=True)

    def __init__(self, *, side: Side = "right"):
        if side not in {"right", "left"}:
            raise ValueError("side must be one of {'right', 'left'}.")
        object.__setattr__(self, "side", side)

    def init(self, terms, t0, t1, y0, args) -> None:
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del args, solver_state, made_jump
        rough_term = unwrap_rough_term(terms)
        _check_linear_rough_term(rough_term)
        if rough_term.basis.depth > FER_MAX_DEPTH:
            raise ValueError(f"LinearFer currently supports depth <= {FER_MAX_DEPTH}.")

        matrices = _matrix_basis(rough_term, y0, self.side)
        components = _degree_components(terms.contr(t0, t1), matrices, rough_term.basis)
        factors = _fer_factors(components)

        y = y0
        if self.side == "right":
            for factor in factors:
                y = y @ jsl.expm(factor)
        else:
            y = _apply_fer_factors(
                y0,
                factors,
                jnp.asarray(1.0, dtype=factors.dtype),
                jnp.asarray(False),
            )
        dense_info = dict(
            y0=y0,
            factors=factors,
            right=jnp.asarray(self.side == "right"),
        )
        return y, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


__all__ = ["LinearFer", "LinearMagnus"]
