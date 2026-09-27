from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from diffrax import RESULTS, AbstractLocalInterpolation, AbstractSolver
from diffrax._custom_types import BoolScalarLike, RealScalarLike
from diffrax._term import WrapTerm
from jaxtyping import Array

from roughrax._bases import CoefficientBasis
from roughrax._solver._fer_coefficients import FER_FACTORS, FER_MAX_DEPTH, LieWord
from roughrax._term import LogSignatureInterpolation, RoughTerm, unwrap_rough_term

Side = Literal["right", "left"]


def _matrix_commutator(a: Array, b: Array) -> Array:
    return a @ b - b @ a


def _check_linear_rough_term(rough_term: RoughTerm) -> None:
    if not isinstance(rough_term.control, LogSignatureInterpolation):
        raise TypeError("Linear solvers require LogSignatureInterpolation.")
    if rough_term.control.solution != "stratonovich":
        raise ValueError("Linear solvers require solution='stratonovich'.")
    if rough_term.basis.kind != "lyndon":
        raise ValueError("Linear solvers require a Lyndon basis.")


def _build_lyndon_matrix_basis(
    level_one: Array,
    basis: CoefficientBasis,
    side: Side,
) -> Array:
    matrices: dict[int, Array] = {}
    # Children have lower degree; preserve the backend's ordering in the result.
    for index in sorted(range(len(basis.keys)), key=basis.degree.__getitem__):
        child_ids = basis.children[index]
        root_colour = basis.root_colour[index]
        if not child_ids:
            assert root_colour is not None
            matrix = level_one[root_colour]
        else:
            if len(child_ids) != 2:
                raise ValueError("Lyndon basis entries must have two children.")
            left, right = (matrices[child] for child in child_ids)
            matrix = (
                _matrix_commutator(left, right)
                if side == "right"
                else _matrix_commutator(right, left)
            )
        matrices[index] = matrix
    return jnp.stack([matrices[index] for index in range(len(basis.keys))])


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


def _apply_matrix(y: Array, matrix: Array, side: Side) -> Array:
    return y @ matrix if side == "right" else matrix @ y


def _apply_generator(y: Array, generator: Array, side: Side) -> Array:
    return _apply_matrix(y, jsl.expm(generator), side)


class _LinearMagnusInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: Array
    omega: Array
    side: Side = eqx.field(static=True)

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        u = (t0 - self.t0) / (self.t1 - self.t0)
        return _apply_generator(self.y0, u * self.omega, self.side)


def _apply_factor_product(y0: Array, factors: Array, side: Side) -> Array:
    product = jnp.eye(factors.shape[-1], dtype=factors.dtype)
    for factor in factors:
        product = product @ jsl.expm(factor)
    return _apply_matrix(y0, product, side)


class _LinearFerInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: Array
    components: Array
    side: Side = eqx.field(static=True)

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        u = (t0 - self.t0) / (self.t1 - self.t0)
        factors = _fer_factors(u * self.components)
        return _apply_factor_product(self.y0, factors, self.side)


def _degree_components(
    coeffs: Array,
    matrices: Array,
    basis: CoefficientBasis,
) -> Array:
    degrees = jnp.arange(1, basis.depth + 1)[:, None]
    weights = jnp.where(degrees == jnp.asarray(basis.degree), coeffs, 0.0)
    return jnp.tensordot(weights, matrices, axes=1)


def _fer_factors(components: Array) -> Array:
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

    factors: list[Array] = []
    for recipe in FER_FACTORS[: len(components)]:
        factor = jnp.zeros_like(components[0])
        for numerator, denominator, word in recipe:
            value = evaluate(word)
            coefficient = jnp.asarray(numerator, dtype=value.dtype) / denominator
            factor = factor + coefficient * value
        factors.append(factor)
    return jnp.stack(factors)


class _AbstractLinearSolver(AbstractSolver[None]):
    term_structure = RoughTerm
    side: Side = eqx.field(static=True)

    def __init__(self, *, side: Side = "right") -> None:
        if side not in {"right", "left"}:
            raise ValueError("side must be one of {'right', 'left'}.")
        object.__setattr__(self, "side", side)

    def init(
        self,
        terms: RoughTerm | WrapTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Any,
    ) -> None:
        del terms, t0, t1, y0, args

    def func(
        self, terms: RoughTerm | WrapTerm, t0: RealScalarLike, y0: Array, args: Any
    ) -> Array:
        return terms.vf(t0, y0, args)


class LinearMagnus(_AbstractLinearSolver):
    """Linear RDE solver using one matrix Magnus exponential."""

    interpolation_cls = _LinearMagnusInterpolation

    def step(
        self,
        terms: RoughTerm | WrapTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Any,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Array, None, dict[str, Array | Side], None, RESULTS]:
        del args, solver_state, made_jump
        rough_term = unwrap_rough_term(terms)
        _check_linear_rough_term(rough_term)

        matrices = _matrix_basis(rough_term, y0, self.side)
        omega = jnp.tensordot(terms.contr(t0, t1), matrices, axes=1)
        y1 = _apply_generator(y0, omega, self.side)
        dense_info = {"y0": y0, "omega": omega, "side": self.side}
        return y1, None, dense_info, None, RESULTS.successful


class LinearFer(_AbstractLinearSolver):
    """Linear RDE solver using a truncated Fer product through depth 6."""

    interpolation_cls = _LinearFerInterpolation

    def step(
        self,
        terms: RoughTerm | WrapTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Any,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Array, None, dict[str, Array | Side], None, RESULTS]:
        del args, solver_state, made_jump
        rough_term = unwrap_rough_term(terms)
        _check_linear_rough_term(rough_term)
        if rough_term.basis.depth > FER_MAX_DEPTH:
            raise ValueError(f"LinearFer currently supports depth <= {FER_MAX_DEPTH}.")

        matrices = _matrix_basis(rough_term, y0, self.side)
        components = _degree_components(terms.contr(t0, t1), matrices, rough_term.basis)
        factors = _fer_factors(components)
        y1 = _apply_factor_product(y0, factors, self.side)
        dense_info = {"y0": y0, "components": components, "side": self.side}
        return y1, None, dense_info, None, RESULTS.successful


__all__ = ["LinearFer", "LinearMagnus"]
