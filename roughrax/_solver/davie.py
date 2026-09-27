from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from diffrax import RESULTS, AbstractSolver
from diffrax._custom_types import BoolScalarLike, RealScalarLike
from diffrax._term import WrapTerm
from georax._solver._interpolation import GeometricInterpolation, geometric_dense_info
from jaxtyping import Array

from roughrax._rough_taylor import rough_taylor_columns
from roughrax._term import RoughTerm, SignatureInterpolation, unwrap_rough_term


class Davie(AbstractSolver[None]):
    r"""Explicit signature Taylor solver on Euclidean spaces and manifolds.

    Requires SignatureInterpolation and steps between adjacent signature knots. The depth sets the word/tree truncation degree. Geometric controls use iterated directional derivatives; branched controls use BCK elementary differentials or MKW ordered-forest operators. Manifold updates are evaluated in a Georax local chart and reconstructed on the manifold. No inner ODE solver or logarithm is used.

    ControlledVectorField uses the same autonomous augmentation as LogODE and requires geometric controls. No adaptive error estimate is supplied. Dense output follows a line in the local chart (ordinary linear interpolation in Euclidean space); it is not a degree-N continuous extension.

    References: Davie, https://arxiv.org/abs/0710.0772; Lie–Butcher series, https://arxiv.org/abs/1701.03654.
    """

    term_structure = RoughTerm
    interpolation_cls = GeometricInterpolation

    def init(
        self,
        terms: RoughTerm | WrapTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Any,
    ) -> None:
        del t0, t1, y0, args
        self._check_control(terms)

    @staticmethod
    def _check_control(terms: RoughTerm | WrapTerm) -> None:
        if not isinstance(unwrap_rough_term(terms).control, SignatureInterpolation):
            raise TypeError("Davie requires SignatureInterpolation.")

    def step(
        self,
        terms: RoughTerm | WrapTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Any,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Array, None, dict[str, Any], None, RESULTS]:
        del solver_state, made_jump
        self._check_control(terms)
        time, end_time = t0, t1
        wrapped = terms
        while isinstance(wrapped, WrapTerm):
            time = time * wrapped.direction
            end_time = end_time * wrapped.direction
            wrapped = wrapped.term
        # WrapTerm negates additive increments. Full signatures instead need
        # the group inverse, evaluated at the original physical times.
        rough_term = wrapped
        coeffs = rough_term.contr(time, end_time)
        term, state, project = rough_term.prepare_step(time, y0, args)
        chart = term.geometry.select_pullback_chart(term.basis.depth)
        columns = rough_taylor_columns(
            term.vector_field, term.basis, term.geometry, chart, state
        )
        increment = jnp.tensordot(coeffs, columns, axes=1)
        y1 = project(chart.apply(state, increment, term.geometry))
        # Dense output only needs the physical part of the product chart.
        geometry = rough_term.geometry
        physical_increment = increment.reshape(-1)[state.size - y0.size :].reshape(
            geometry.zero_coordinates(y0).shape
        )
        dense_info = geometric_dense_info(
            y0,
            y1,
            (physical_increment,),
            geometry,
            geometry.select_pullback_chart(term.basis.depth),
        )
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self, terms: RoughTerm | WrapTerm, t0: RealScalarLike, y0: Array, args: Any
    ) -> Array:
        return terms.vf(t0, y0, args)
