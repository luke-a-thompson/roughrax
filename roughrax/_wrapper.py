from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax.tree_util as jtu
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from diffrax._solution import RESULTS
from diffrax._solver.runge_kutta import ButcherTableau


def _tree_add(x: Y, y: Y) -> Y:
    return jtu.tree_map(lambda a, b: a + b, x, y)


def _tree_mul(weight: Any, tree: Y) -> Y:
    return jtu.tree_map(lambda x: weight * x, tree)


def _tree_weighted_sum(weights: tuple[float, ...], trees: tuple[Y, ...], like: Y) -> Y:
    total = jtu.tree_map(lambda x: 0 * x, like)
    for weight, tree in zip(weights, trees, strict=True):
        total = _tree_add(total, _tree_mul(weight, tree))
    return total


class RoughRK(diffrax.AbstractSolver[None]):
    solver: diffrax.AbstractERK = eqx.field(static=True)

    interpolation_cls = diffrax.LocalLinearInterpolation
    term_structure = property(lambda self: self.solver.term_structure)

    def __check_init__(self):
        if not isinstance(self.solver, diffrax.AbstractERK):
            raise TypeError("WrappedRoughRK expects an explicit diffrax RK solver.")
        if not isinstance(self.solver.tableau, ButcherTableau):
            raise TypeError("WrappedRoughRK expects a single Butcher tableau.")
        if self.solver.tableau.implicit:
            raise TypeError("WrappedRoughRK only supports explicit tableaus.")

    @property
    def tableau(self) -> ButcherTableau:
        return self.solver.tableau

    def init(
        self,
        terms: diffrax.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        del terms, t0, t1, y0, args
        return None

    def func(
        self,
        terms: diffrax.AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    def step(
        self,
        terms: diffrax.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, None, DenseInfo, None, RESULTS]:
        del solver_state, made_jump

        dt = t1 - t0
        tableau = self.tableau
        control = terms.contr(t0, t1)

        stage_times = (t0 + tableau.c1 * dt,) + tuple(
            t0 + c_i * dt for c_i in tableau.c
        )
        stage_increments = []

        for i, stage_time in enumerate(stage_times):
            if i == 0:
                stage_value = y0
            else:
                increment = _tree_weighted_sum(
                    tuple(float(a_ij) for a_ij in tableau.a_lower[i - 1]),
                    tuple(stage_increments),
                    y0,
                )
                stage_value = _tree_add(y0, increment)

            stage_vf = terms.vf(stage_time, stage_value, args)
            stage_increments.append(terms.prod(stage_vf, control))

        y1 = _tree_add(
            y0,
            _tree_weighted_sum(
                tuple(float(b_i) for b_i in tableau.b_sol),
                tuple(stage_increments),
                y0,
            ),
        )

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful
