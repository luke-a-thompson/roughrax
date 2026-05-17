from __future__ import annotations

import equinox as eqx
from diffrax import (
    AbstractSolver,
    LocalLinearInterpolation,
    Tsit5,
)
from georax import GeometricTerm, RKMK

from roughrax._term import RoughTerm, unwrap_rough_term


class LogODE(AbstractSolver[None]):
    """Log-ODE solver for a ``RoughTerm`` with primitive log coefficients."""

    term_structure = RoughTerm
    interpolation_cls = LocalLinearInterpolation

    solver: AbstractSolver = eqx.field(static=True)

    def __init__(self, solver: AbstractSolver | None = None):
        if solver is None:
            solver = RKMK(Tsit5())
        if getattr(solver, "term_structure", None) is RoughTerm:
            raise TypeError(
                "LogODE wraps a deterministic GeometricTerm solver. Use "
                "LogODE(georax.RKMK(diffrax.Tsit5())), not a solver that already "
                "wraps LogODE/RoughTerm."
            )
        object.__setattr__(self, "solver", solver)

    def init(self, terms, t0, t1, y0, args) -> None:
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state

        rough_term = unwrap_rough_term(terms)
        coeffs = terms.contr(t0, t1)

        def frozen_vector_field(s, y, args):
            del s
            return terms.vf_prod(t0, y, args, coeffs)

        inner_term = GeometricTerm(frozen_vector_field, rough_term.geometry)
        inner_state = self.solver.init(inner_term, 0.0, 1.0, y0, args)
        y1, y_error, _, _, result = self.solver.step(
            inner_term,
            0.0,
            1.0,
            y0,
            args,
            inner_state,
            made_jump,
        )
        return y1, y_error, dict(y0=y0, y1=y1), None, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


__all__ = ["LogODE"]
