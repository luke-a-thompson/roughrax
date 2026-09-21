from __future__ import annotations

from copy import copy
from numbers import Integral
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
from roughrax._controlled import ControlledVectorField
from roughrax._pseudo_bialgebra_map import (
    LiftedField,
    VectorField,
    form_pseudo_bialgebra_map,
)


class SignatureInterpolation(AbstractPath):
    """Log-signature interpolation over a knot sequence."""

    control: AbstractPath | None
    ts: Array
    coeffs: Array | None
    correction: Array | None
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
        *,
        correction: Array | None = None,
    ):
        if getattr(control, "ts", None) is None or getattr(control, "ys", None) is None:
            raise TypeError(
                "SignatureInterpolation requires a diffrax "
                "LinearInterpolation-like path with `.ts` and `.ys`."
            )
        if solution == "stratonovich" and correction is not None:
            raise ValueError("correction requires solution='ito'.")

        self.control = control
        self.ts = jnp.asarray(signature_knots)
        self.coeffs = None
        self.correction = None if correction is None else jnp.asarray(correction)
        self.basis = None
        self.depth = depth
        self.solution = solution

    @classmethod
    def from_logsignatures(
        cls,
        ts: Array,
        coeffs: Array,
        input_dim: int,
        depth: int,
    ) -> SignatureInterpolation:
        """Construct from local PySigLib method-1 Lyndon log-signatures.

        ``coeffs[i]`` must be the log-signature over ``[ts[i], ts[i + 1]]``.
        Its shape must be ``(num_intervals, logsig_dim)``. The final coefficient
        axis follows ``pysiglib.lyndon_words`` ordering and does not include a
        scalar term. Use ``jax.vmap`` over independent controls and solves.
        """
        if (
            not isinstance(input_dim, Integral)
            or isinstance(input_dim, bool)
            or input_dim < 1
        ):
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(depth, Integral) or isinstance(depth, bool) or depth < 1:
            raise ValueError("depth must be a positive integer.")

        input_dim = int(input_dim)
        depth = int(depth)
        ts = jnp.asarray(ts)
        coeffs = jnp.asarray(coeffs)

        if ts.ndim != 1:
            raise ValueError(
                f"ts must have shape (num_intervals + 1,), got {ts.shape}."
            )
        if ts.shape[0] < 2:
            raise ValueError("ts must contain at least two points.")
        if coeffs.ndim != 2:
            raise ValueError(
                "coeffs must have shape (num_intervals, logsig_dim), "
                f"got {coeffs.shape}."
            )

        basis = make_lyndon_basis(depth, input_dim)
        num_intervals = ts.shape[0] - 1
        logsig_dim = len(basis.keys)
        if coeffs.shape[0] != num_intervals:
            raise ValueError(
                "coeffs first axis must have length "
                f"num_intervals={num_intervals}, got {coeffs.shape[0]}."
            )
        if coeffs.shape[-1] != logsig_dim:
            raise ValueError(
                "coeffs last axis must have length "
                f"logsig_dim={logsig_dim} for input_dim={input_dim} and "
                f"depth={depth}, got {coeffs.shape[-1]}."
            )

        ts = eqx.error_if(
            ts,
            (~jnp.isfinite(ts)).any() | (ts[1:] <= ts[:-1]).any(),
            "ts must be finite and strictly increasing.",
        )

        out = object.__new__(cls)
        object.__setattr__(out, "control", None)
        object.__setattr__(out, "ts", ts)
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "correction", None)
        object.__setattr__(out, "basis", basis)
        object.__setattr__(out, "depth", depth)
        object.__setattr__(out, "solution", "stratonovich")
        return out

    def materialise(self, geometry: Manifold[Any]) -> SignatureInterpolation:
        if self.coeffs is not None:
            return self

        if self.control is None:
            raise RuntimeError(
                "SignatureInterpolation has neither a sampled control nor "
                "precomputed coefficients."
            )

        control_ts = jnp.asarray(getattr(self.control, "ts"))
        ys = jnp.asarray(getattr(self.control, "ys"))
        dim = int(ys.shape[-1])
        if self.ts.ndim != 1:
            raise ValueError("signature_knots must be one-dimensional.")
        num_intervals = self.ts.shape[0] - 1
        num_control_intervals = control_ts.shape[0] - 1
        if num_intervals < 1:
            raise ValueError("signature_knots must contain at least two points.")
        if num_control_intervals % num_intervals != 0:
            raise ValueError(
                "signature_knots must evenly subdivide the control sample grid."
            )

        stride = num_control_intervals // num_intervals
        expected_knots = control_ts[::stride]
        signature_knots = eqx.error_if(
            self.ts,
            (~jnp.isfinite(self.ts)).any()
            | (self.ts[1:] <= self.ts[:-1]).any()
            | (self.ts != expected_knots).any(),
            "signature_knots must be finite, strictly increasing, and equal "
            "control.ts[::stride].",
        )
        indices = stride * jnp.arange(num_intervals)[:, None] + jnp.arange(stride + 1)
        windows = ys[indices]

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
                    planar=planar,
                    correction=self.correction,
                )
            case "stratonovich":
                basis = make_lyndon_basis(self.depth, dim)
                pysiglib.prepare_log_sig(dim, self.depth, 1)
                coeffs = pysiglib.log_sig(windows, self.depth)
            case _:
                raise ValueError(f"Unknown solution type {self.solution!r}.")

        out = copy(self)
        object.__setattr__(out, "ts", signature_knots)
        object.__setattr__(out, "coeffs", coeffs)
        object.__setattr__(out, "basis", basis)
        return out

    def evaluate(self, t0, t1=None, left=True):
        del left
        if self.coeffs is None:
            raise ValueError("SignatureInterpolation must be materialised first.")
        if t1 is None:
            return self._evaluate(t0)
        lower = jnp.minimum(t0, t1)
        upper = jnp.maximum(t0, t1)
        index = jnp.searchsorted(self.ts, lower, side="right") - 1
        index = jnp.clip(index, 0, self.coeffs.shape[0] - 1)
        denominator = self.ts[index + 1] - self.ts[index]
        increment = ((t1 - t0) / denominator) * self.coeffs[index]
        return eqx.error_if(
            increment,
            upper > self.ts[index + 1],
            "SignatureInterpolation intervals may not cross signature knots; "
            "clip solver steps at the signature knots.",
        )

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
    """Diffrax term over rough-path coefficients."""

    vector_field: VectorField | ControlledVectorField
    control: SignatureInterpolation
    basis: PrimitiveBasis = eqx.field(static=True)
    lifted_fields: tuple[LiftedField, ...] = eqx.field(static=True)
    has_lifted_vector_field: bool = eqx.field(static=True)
    geometry: Manifold[Any] = Euclidean()

    def __init__(
        self,
        vector_field: VectorField | ControlledVectorField,
        control: SignatureInterpolation,
        geometry: Manifold[Any] = Euclidean(),
        *,
        _has_lifted_vector_field: bool = False,
    ):
        if not isinstance(control, SignatureInterpolation):
            raise TypeError("RoughTerm control must be a SignatureInterpolation.")
        control = control.materialise(geometry)
        assert control.basis is not None

        if isinstance(vector_field, ControlledVectorField):
            if _has_lifted_vector_field:
                raise ValueError(
                    "ControlledVectorField supplies level-one coefficients, not lifted columns."
                )
            if control.basis.kind != "lyndon":
                raise ValueError(
                    "ControlledVectorField currently requires a geometric (stratonovich) control."
                )
            if len(vector_field.coefficients) < control.depth:
                raise ValueError(
                    f"Depth {control.depth} requires {control.depth} controlled coefficient "
                    "entries (including the vector field); use None for known zero derivatives."
                )

        self.vector_field = vector_field
        self.control = control
        self.basis = control.basis
        self.geometry = geometry
        self.has_lifted_vector_field = _has_lifted_vector_field
        self.lifted_fields = (
            ()
            if _has_lifted_vector_field
            or isinstance(vector_field, ControlledVectorField)
            else form_pseudo_bialgebra_map(vector_field, control.basis, geometry)
        )

    @classmethod
    def from_lifted_vector_field(
        cls,
        vector_field: VectorField,
        control: SignatureInterpolation,
        geometry: Manifold[Any] = Euclidean(),
    ) -> RoughTerm:
        """Construct from a field returning all log-signature columns."""
        return cls(
            vector_field,
            control,
            geometry,
            _has_lifted_vector_field=True,
        )

    def vf(self, t, y, args):
        if isinstance(self.vector_field, ControlledVectorField):
            term, state, _ = self.prepare_step(t, y, args)
            auxiliary_size = state.size - y.size
            frame_shape = (
                y.shape
                if isinstance(self.geometry, Euclidean)
                else self.geometry.coordinate_shape
            )
            return term.vf(t, state, args)[:, auxiliary_size:].reshape(
                (len(self.basis.keys), *frame_shape)
            )
        if self.has_lifted_vector_field:
            fields = jnp.asarray(self.vector_field(y))
            logsig_size = len(self.basis.keys)
            columns_shape = (*jnp.shape(y), logsig_size)
            if fields.shape != columns_shape and not (
                fields.ndim == 1 and fields.size == jnp.size(y) * logsig_size
            ):
                raise ValueError(
                    "A lifted vector field must return shape "
                    f"{columns_shape} or a flat array of the same size, got "
                    f"{fields.shape}."
                )
            columns = jnp.reshape(fields, columns_shape)
            return jnp.moveaxis(columns, -1, 0)
        return jnp.stack([field(y) for field in self.lifted_fields])

    def prepare_step(self, t, y, args):
        """Return an autonomous term, inner state, and projection for one step."""
        if not isinstance(self.vector_field, ControlledVectorField):
            return self, y, lambda state: state
        field, state, geometry, project = self.vector_field._augment(
            t,
            y,
            args,
            dim=self.basis.dim,
            depth=self.basis.depth,
            geometry=self.geometry,
        )
        return RoughTerm(field, self.control, geometry), state, project

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
