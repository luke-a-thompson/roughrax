from __future__ import annotations

import equinox as eqx
from diffrax import AbstractSolver, LocalLinearInterpolation, ODETerm
from georax import Euclidean, GeometricTerm

from roughrax._term import RoughTerm, VirtualPathInterpolation, unwrap_rough_term


class CommutatorFreeLogODE2(AbstractSolver[None]):
    """Depth-2 commutator-free Log-ODE solver.

    This solver is only for geometric/Stratonovich depth-2 rough paths. It
    replaces the depth-2 bracket vector field by a virtual piecewise-linear path
    whose depth-2 log-signature matches the PySigLib log-signature on each
    coarse interval.
    """

    term_structure = RoughTerm
    interpolation_cls = LocalLinearInterpolation

    solver: AbstractSolver = eqx.field(static=True)

    def __init__(self, solver: AbstractSolver):
        object.__setattr__(self, "solver", solver)

    def init(self, terms, t0, t1, y0, args) -> None:
        del terms, t0, t1, y0, args
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state

        rough_term = unwrap_rough_term(terms)
        if rough_term.control.solution != "stratonovich":
            raise ValueError(
                "CommutatorFreeLogODE2 only supports solution='stratonovich'."
            )
        if rough_term.control.depth != 2:
            raise ValueError("CommutatorFreeLogODE2 only supports depth=2.")
        if rough_term.basis.kind != "lyndon":
            raise ValueError("CommutatorFreeLogODE2 requires a Lyndon basis.")
        if not isinstance(rough_term.control, VirtualPathInterpolation):
            raise ValueError(
                "CommutatorFreeLogODE2 requires a VirtualPathInterpolation "
                "control. Wrap SignatureInterpolation with "
                "VirtualPathInterpolation to precompute virtual increments."
            )

        virtual_increments = rough_term.control.virtual_increments(t0, t1)

        y = y0
        y_error = None
        result = None

        for segment_index in range(virtual_increments.shape[0]):
            increment = virtual_increments[segment_index]

            def frozen_vector_field(s, y, args, *, increment=increment):
                del s, args
                return rough_term.base_vf_prod(y, increment)

            inner_term = (
                ODETerm(frozen_vector_field)
                if isinstance(rough_term.geometry, Euclidean)
                else GeometricTerm(frozen_vector_field, rough_term.geometry)
            )
            inner_state = self.solver.init(inner_term, 0.0, 1.0, y, args)
            y, y_error, _, _, result = self.solver.step(
                inner_term,
                0.0,
                1.0,
                y,
                args,
                inner_state,
                made_jump if segment_index == 0 else False,
            )

        return y, y_error, dict(y0=y0, y1=y), None, result

    def func(self, terms, t0, y0, args):
        del t0, args
        rough_term = unwrap_rough_term(terms)
        return rough_term.base_vf(y0)
