from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pysiglib.jax_api as pysiglib
from diffrax import AbstractPath, AbstractTerm
from diffrax._term import WrapTerm
from georax import Euclidean, Manifold
from jaxtyping import Array

from roughrax._bases import (
    PrimitiveBasis,
    make_lyndon_basis,
    make_planar_tree_basis,
    make_tree_basis,
)
from roughrax._pseudo_bialgebra_map import (
    LiftedField,
    VectorField,
    form_pseudo_bialgebra_map,
)


class _SignaturePath(AbstractPath):
    ts: Array
    coeffs: Array

    @property
    def t0(self):
        return self.ts[0]

    @property
    def t1(self):
        return self.ts[-1]

    def evaluate(self, t0, t1=None, left=True):
        del t1, left
        index = jnp.searchsorted(self.ts, t0, side="right") - 1
        return self.coeffs[index]


class RoughTerm(AbstractTerm[Array, Array]):
    """Diffrax term over internally lifted rough-path coefficients."""

    vector_field: VectorField = eqx.field(static=True)
    control: AbstractPath
    basis: PrimitiveBasis = eqx.field(static=True)
    lifted_fields: tuple[LiftedField, ...] = eqx.field(static=True)
    geometry: Manifold[Any] = Euclidean()

    def __init__(
        self,
        vector_field: VectorField,
        control: AbstractPath,
        geometry: Manifold[Any] = Euclidean(),
        *,
        depth: int,
        interval_ts: Array | None = None,
        solution: Literal["ito", "stratonovich"],
    ):
        ts = getattr(control, "ts", None)
        ys = getattr(control, "ys", None)
        if ts is None or ys is None:
            raise TypeError(
                "Internal signature computation requires a diffrax "
                "LinearInterpolation-like path with `.ts` and `.ys`."
            )

        ts_np = np.asarray(ts)
        ys_np = np.ascontiguousarray(ys)
        dim = int(ys_np.shape[-1])

        interval_ts_np = ts_np if interval_ts is None else np.asarray(interval_ts)
        indices = np.searchsorted(ts_np, interval_ts_np)

        match solution:
            case "ito":
                planar = not isinstance(geometry, Euclidean)
                primitive_basis = (
                    make_planar_tree_basis(depth, dim)
                    if planar
                    else make_tree_basis(depth, dim)
                )
                pysiglib.prepare_branched_sig(dim, depth, planar=planar)
                coeffs = [
                    pysiglib.branched_log_sig(
                        ys_np[indices[j] : indices[j + 1] + 1],
                        depth,
                        tree_order="canonical",
                        planar=planar,
                    )
                    for j in range(len(indices) - 1)
                ]
            case "stratonovich":
                primitive_basis = make_lyndon_basis(depth, dim)
                pysiglib.prepare_log_sig(dim, depth, 1)
                coeffs = [
                    pysiglib.log_sig(ys_np[indices[j] : indices[j + 1] + 1], depth)
                    for j in range(len(indices) - 1)
                ]
            case _:
                raise ValueError(f"Unknown solution type {solution!r}.")

        coeff_control = _SignaturePath(jnp.asarray(interval_ts_np), jnp.asarray(coeffs))

        self.vector_field = vector_field
        self.control = coeff_control
        self.basis = primitive_basis
        self.geometry = geometry
        self.lifted_fields = form_pseudo_bialgebra_map(
            vector_field, primitive_basis, geometry
        )

    def vf(self, t, y, args):
        del t, args
        return jnp.stack([field(y) for field in self.lifted_fields])

    def contr(self, t0, t1, **kwargs):
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf, control):
        return jnp.tensordot(control, vf, axes=1)

    def is_vf_expensive(self, t0, t1, y, args) -> bool:
        del t0, t1, y, args
        return True


def unwrap_rough_term(term) -> RoughTerm:
    while isinstance(term, WrapTerm):
        term = term.term
    return term


__all__ = ["RoughTerm", "unwrap_rough_term"]
