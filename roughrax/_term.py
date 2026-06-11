from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
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
from roughrax._virtual_path import virtual_increments_depth2


class SignatureInterpolation(AbstractPath):
    """Log-signature interpolation over a sampled control."""

    control: AbstractPath
    ts: Array
    coeffs: Array | None
    cumulative_coeffs: Array | None
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
        self.cumulative_coeffs = None
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

        cumulative = jnp.concatenate(
            [jnp.zeros_like(coeffs[:1]), jnp.cumsum(coeffs, axis=0)],
            axis=0,
        )

        out = SignatureInterpolation(self.control, self.ts, self.depth, self.solution)
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "cumulative_coeffs", cumulative)
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
        assert self.cumulative_coeffs is not None
        index = jnp.searchsorted(self.ts, t, side="right") - 1
        index = jnp.clip(index, 0, self.coeffs.shape[0] - 1)
        fraction = (t - self.ts[index]) / (self.ts[index + 1] - self.ts[index])
        return self.cumulative_coeffs[index] + fraction * self.coeffs[index]


class VirtualPathInterpolation(AbstractPath):
    """Precomputed virtual increments for depth-2 commutator-free solves."""

    signature: SignatureInterpolation
    ts: Array
    coeffs: Array | None
    virtual_increments_array: Array | None
    basis: PrimitiveBasis | None = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    solution: Literal["ito", "stratonovich"] = eqx.field(static=True)
    factorisation: Literal["spectral", "pairwise"] = eqx.field(static=True)

    @property
    def t0(self):
        return self.ts[0]

    @property
    def t1(self):
        return self.ts[-1]

    def __init__(
        self,
        signature: SignatureInterpolation,
        *,
        factorisation: Literal["spectral", "pairwise"] = "spectral",
    ):
        if not isinstance(signature, SignatureInterpolation):
            raise TypeError(
                "VirtualPathInterpolation requires a SignatureInterpolation."
            )

        self.signature = signature
        self.ts = signature.ts
        self.coeffs = signature.coeffs
        self.virtual_increments_array = None
        self.basis = signature.basis
        self.depth = signature.depth
        self.solution = signature.solution
        self.factorisation = factorisation

    def materialise(self, geometry: Manifold[Any]) -> VirtualPathInterpolation:
        if self.virtual_increments_array is not None:
            return self

        if self.solution != "stratonovich":
            raise ValueError(
                "VirtualPathInterpolation only supports solution='stratonovich'."
            )
        if self.depth != 2:
            raise ValueError("VirtualPathInterpolation only supports depth=2.")

        signature = self.signature.materialise(geometry)
        if signature.basis is None:
            raise ValueError("SignatureInterpolation must have a materialised basis.")
        if signature.basis.kind != "lyndon":
            raise ValueError("VirtualPathInterpolation requires a Lyndon basis.")
        if signature.coeffs is None:
            raise ValueError(
                "SignatureInterpolation must have materialised coefficients."
            )

        virtual_increments = jax.vmap(
            lambda coeffs: virtual_increments_depth2(
                coeffs,
                signature.basis,
                factorisation=self.factorisation,
            )
        )(signature.coeffs)

        out = VirtualPathInterpolation(signature, factorisation=self.factorisation)
        object.__setattr__(out, "coeffs", signature.coeffs)
        object.__setattr__(out, "virtual_increments_array", virtual_increments)
        object.__setattr__(out, "basis", signature.basis)
        return out

    def evaluate(self, t0, t1=None, left=True):
        return self.signature.evaluate(t0, t1, left=left)

    def virtual_increments(self, t0, t1):
        if self.virtual_increments_array is None:
            raise ValueError("VirtualPathInterpolation must be materialised first.")

        index = jnp.searchsorted(self.ts, t0, side="right") - 1
        index = jnp.clip(index, 0, self.virtual_increments_array.shape[0] - 1)
        increments = self.virtual_increments_array[index]
        invalid_interval = (t0 != self.ts[index]) | (t1 != self.ts[index + 1])
        return eqx.error_if(
            increments,
            invalid_interval,
            "VirtualPathInterpolation only supports exact knot-to-knot intervals.",
        )


class RoughTerm(AbstractTerm[Array, Array]):
    """Diffrax term over rough-path coefficients."""

    vector_field: VectorField = eqx.field(static=True)
    control: SignatureInterpolation | VirtualPathInterpolation
    basis: PrimitiveBasis = eqx.field(static=True)
    lifted_fields: tuple[LiftedField, ...] = eqx.field(static=True)
    geometry: Manifold[Any] = Euclidean()

    def __init__(
        self,
        vector_field: VectorField,
        control: SignatureInterpolation | VirtualPathInterpolation,
        geometry: Manifold[Any] = Euclidean(),
    ):
        if not isinstance(control, (SignatureInterpolation, VirtualPathInterpolation)):
            raise TypeError(
                "RoughTerm control must be a SignatureInterpolation or "
                "VirtualPathInterpolation."
            )
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
        fields = jnp.asarray(self.vector_field(y))
        logsig_size = len(self.basis.keys)
        columns_shape = (*jnp.shape(y), logsig_size)
        base_shape = (self.basis.dim, *jnp.shape(y))
        if fields.shape == columns_shape and fields.shape != base_shape:
            return jnp.moveaxis(fields, -1, 0)
        if (
            fields.ndim == 1
            and fields.size == jnp.size(y) * logsig_size
            and fields.shape != base_shape
        ):
            columns = jnp.reshape(fields, columns_shape)
            return jnp.moveaxis(columns, -1, 0)
        return jnp.stack([field(y) for field in self.lifted_fields])

    def base_vf(self, y):
        """Evaluate only the base vector fields V_i."""

        fields = jnp.asarray(self.vector_field(y))
        coordinate_shape = (
            jnp.shape(y)
            if isinstance(self.geometry, Euclidean)
            else self.geometry.coordinate_shape
        )
        expected_shape = (self.basis.dim, *coordinate_shape)
        if fields.shape != expected_shape:
            raise ValueError(
                "Commutator-free solvers require vector_field(y) to return "
                "base vector fields with shape "
                f"{expected_shape}, got {fields.shape}."
            )
        return fields

    def base_vf_prod(self, y, control):
        """Contract a first-level control increment against base vector fields."""

        if hasattr(self.vector_field, "vf_prod"):
            return self.vector_field.vf_prod(y, control)
        return jnp.tensordot(control, self.base_vf(y), axes=1)

    def contr(self, t0, t1, **kwargs):
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf, control):
        return jnp.tensordot(control, vf, axes=1)

    def is_vf_expensive(self, t0, t1, y, args) -> bool:
        del t0, t1, y, args
        return True


def unwrap_rough_term(term: AbstractTerm) -> RoughTerm:
    while isinstance(term, WrapTerm):
        term = term.term
    return term


__all__ = [
    "RoughTerm",
    "SignatureInterpolation",
    "VirtualPathInterpolation",
    "unwrap_rough_term",
]
