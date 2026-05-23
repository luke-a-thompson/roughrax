from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
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


class SignatureInterpolation(AbstractPath):
    """Log-signature interpolation over a sampled control."""

    control: AbstractPath
    ts: Array
    coeffs: Array | None
    basis: PrimitiveBasis | None = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    solution: Literal["ito", "stratonovich"] = eqx.field(static=True)

    @property
    def t0(self):
        return self.ts[0]

    @property
    def t1(self):
        return self.ts[-1]

    def __init__(
        self,
        control: AbstractPath,
        signature_knots: Array,
        depth: int,
        solution: Literal["ito", "stratonovich"],
    ):
        if getattr(control, "ts", None) is None or getattr(control, "ys", None) is None:
            raise TypeError(
                "SignatureInterpolation requires a diffrax "
                "LinearInterpolation-like path with `.ts` and `.ys`."
            )

        self.control = control
        self.ts = jnp.asarray(signature_knots)
        self.coeffs = None
        self.basis = None
        self.depth = depth
        self.solution = solution

    def materialise(self, geometry: Manifold[Any]) -> SignatureInterpolation:
        if self.coeffs is not None:
            return self

        control_ts = getattr(self.control, "ts")
        ys = jnp.asarray(getattr(self.control, "ys"))
        dim = int(ys.shape[-1])
        num_intervals = self.ts.shape[0] - 1
        num_control_intervals = control_ts.shape[0] - 1
        if num_intervals < 1:
            raise ValueError("signature_knots must contain at least two points.")
        if num_control_intervals % num_intervals != 0:
            raise ValueError(
                "signature_knots must evenly subdivide the control sample grid."
            )

        stride = num_control_intervals // num_intervals
        windows = jnp.stack(
            [ys[j * stride : (j + 1) * stride + 1] for j in range(num_intervals)]
        )

        match self.solution:
            case "ito":
                planar = not isinstance(geometry, Euclidean)
                basis = (
                    make_planar_tree_basis(self.depth, dim)
                    if planar
                    else make_tree_basis(self.depth, dim)
                )
                pysiglib.prepare_branched_sig(dim, self.depth, planar=planar)
                coeffs = pysiglib.branched_log_sig(
                    windows,
                    self.depth,
                    tree_order="canonical",
                    planar=planar,
                )
            case "stratonovich":
                basis = make_lyndon_basis(self.depth, dim)
                pysiglib.prepare_log_sig(dim, self.depth, 1)
                coeffs = pysiglib.log_sig(windows, self.depth)
            case _:
                raise ValueError(f"Unknown solution type {self.solution!r}.")

        out = SignatureInterpolation(self.control, self.ts, self.depth, self.solution)
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "basis", basis)
        return out

    def evaluate(self, t0, t1=None, left=True):
        del left
        if self.coeffs is None:
            raise ValueError("SignatureInterpolation must be materialised first.")
        if t1 is None:
            return self._evaluate(t0)
        return self._evaluate(t1) - self._evaluate(t0)

    def _evaluate(self, t):
        assert self.coeffs is not None
        index = jnp.searchsorted(self.ts, t, side="right") - 1
        index = jnp.clip(index, 0, self.coeffs.shape[0] - 1)
        cumulative = jnp.concatenate(
            [
                jnp.zeros_like(self.coeffs[:1]),
                jnp.cumsum(self.coeffs, axis=0),
            ],
            axis=0,
        )
        fraction = (t - self.ts[index]) / (self.ts[index + 1] - self.ts[index])
        return cumulative[index] + fraction * self.coeffs[index]


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
        control: SignatureInterpolation,
        geometry: Manifold[Any] = Euclidean(),
    ):
        if not isinstance(control, SignatureInterpolation):
            raise TypeError("RoughTerm control must be a SignatureInterpolation.")
        control = control.materialise(geometry)
        assert control.basis is not None

        self.vector_field = vector_field
        self.control = control
        self.basis = control.basis
        self.geometry = geometry
        self.lifted_fields = form_pseudo_bialgebra_map(
            vector_field, control.basis, geometry
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


__all__ = ["RoughTerm", "SignatureInterpolation", "unwrap_rough_term"]
