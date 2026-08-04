from __future__ import annotations

from typing import Literal

import equinox as eqx
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


def _matrix_basis(rough_term: RoughTerm, y0: Array, side: Side) -> Array:
    matrix_basis = getattr(rough_term.vector_field, "matrix_basis", None)
    if matrix_basis is None or callable(matrix_basis):
        raise ValueError(
            "Linear vector fields must expose a `matrix_basis` array with shape "
            "(driver_dim, matrix_dim, matrix_dim)."
        )
    matrices = jnp.asarray(matrix_basis, dtype=y0.dtype)

    if (
        matrices.ndim != 3
        or matrices.shape[0] != rough_term.basis.dim
        or matrices.shape[-1] != matrices.shape[-2]
    ):
        raise ValueError(
            "matrix_basis must have shape "
            f"({rough_term.basis.dim}, matrix_dim, matrix_dim), "
            f"got {matrices.shape}."
        )
    return _build_lyndon_matrix_basis(matrices, rough_term.basis, side)


def _contract(coeffs: Array, matrices: Array) -> Array:
    return jnp.tensordot(coeffs, matrices, axes=1)


def _apply_matrix(y: Array, matrix: Array, side: Side) -> Array:
    return y @ matrix if side == "right" else matrix @ y


def _apply_generator(y: Array, generator: Array, side: Side) -> Array:
    return _apply_matrix(y, jsl.expm(generator), side)


class _LinearMagnusInterpolation(AbstractLocalInterpolation):
    t0: Array
    t1: Array
    y0: Array
    omega: Array
    side: Side = eqx.field(static=True)

    def evaluate(self, t0, t1=None, left: bool = True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        u = (t0 - self.t0) / (self.t1 - self.t0)
        return _apply_generator(self.y0, u * self.omega, self.side)


def _apply_factor_product(y0: Array, factors: Array, side: Side) -> Array:
    eye = jnp.eye(factors.shape[-1], dtype=factors.dtype)
    product = eye
    for factor in factors:
        product = product @ jsl.expm(factor)
    return _apply_matrix(y0, product, side)


class _LinearFerInterpolation(AbstractLocalInterpolation):
    t0: Array
    t1: Array
    y0: Array
    components: Array
    side: Side = eqx.field(static=True)

    def evaluate(self, t0, t1=None, left: bool = True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        u = (t0 - self.t0) / (self.t1 - self.t0)
        factors = _fer_factors([u * component for component in self.components])
        return _apply_factor_product(self.y0, factors, self.side)


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
            side=self.side,
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
        y = _apply_factor_product(y0, factors, self.side)
        dense_info = dict(
            y0=y0,
            components=jnp.stack(components),
            side=self.side,
        )
        return y, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


__all__ = ["LinearFer", "LinearMagnus"]
