from __future__ import annotations

from copy import copy
from numbers import Integral
from typing import Any, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import pysiglib.jax_api as pysiglib
from diffrax import AbstractPath, AbstractTerm
from diffrax._custom_types import RealScalarLike
from diffrax._term import WrapTerm
from georax import Euclidean, Manifold
from jaxtyping import Array

from roughrax._bases import (
    CoefficientBasis,
    make_lyndon_basis,
    make_planar_tree_basis,
    make_tree_basis,
    make_word_basis,
)
from roughrax._controlled import ControlledVectorField
from roughrax._pseudo_bialgebra_map import (
    LiftedField,
    VectorField,
    form_pseudo_bialgebra_map,
)
from roughrax._rough_taylor import rough_taylor_columns


class _SignatureInterpolation(AbstractPath):
    control: AbstractPath | None
    ts: Array
    coeffs: Array | None
    correction: Array | None
    basis: CoefficientBasis | None = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    solution: Literal["ito", "stratonovich"] = eqx.field(static=True)

    @property
    def t0(self) -> Array:
        return self.ts[0]

    @property
    def t1(self) -> Array:
        return self.ts[-1]

    def __init__(
        self,
        control: AbstractPath,
        signature_knots: Array,
        depth: int,
        solution: Literal["ito", "stratonovich"],
        *,
        correction: Array | None = None,
    ) -> None:
        if getattr(control, "ts", None) is None or getattr(control, "ys", None) is None:
            raise TypeError(
                "Signature controls require a sampled path with `.ts` and `.ys`."
            )
        _check_dimensions(1, depth)
        if solution not in ("ito", "stratonovich"):
            raise ValueError(f"Unknown solution type {solution!r}.")
        if solution == "stratonovich" and correction is not None:
            raise ValueError("correction requires solution='ito'.")
        self.control = control
        self.ts = jnp.asarray(signature_knots)
        self.coeffs = None
        self.correction = None if correction is None else jnp.asarray(correction)
        self.basis = None
        self.depth = int(depth)
        self.solution = solution

    @classmethod
    def _from_coefficients(
        cls,
        ts: Array,
        coeffs: Array,
        basis: CoefficientBasis,
        solution: Literal["ito", "stratonovich"],
    ) -> Self:
        ts, coeffs = jnp.asarray(ts), jnp.asarray(coeffs)
        if ts.ndim != 1:
            raise ValueError(
                f"ts must have shape (num_intervals + 1,), got {ts.shape}."
            )
        if ts.shape[0] < 2:
            raise ValueError("ts must contain at least two points.")
        if coeffs.ndim != 2:
            raise ValueError(
                f"coeffs must have shape (num_intervals, coefficient_dim), got {coeffs.shape}."
            )
        if coeffs.shape[0] != ts.shape[0] - 1:
            raise ValueError("coeffs first axis must have length num_intervals.")
        if coeffs.shape[-1] != len(basis.keys):
            raise ValueError(f"coeffs last axis must have length {len(basis.keys)}.")
        ts = eqx.error_if(
            ts,
            (~jnp.isfinite(ts)).any() | (ts[1:] <= ts[:-1]).any(),
            "ts must be finite and strictly increasing.",
        )
        out = object.__new__(cls)
        for name, value in dict(
            control=None,
            ts=ts,
            coeffs=coeffs,
            correction=None,
            basis=basis,
            depth=basis.depth,
            solution=solution,
        ).items():
            object.__setattr__(out, name, value)
        return out

    def materialise(self, geometry: Manifold[Any]) -> Self:
        if self.coeffs is not None:
            if self.solution == "ito" and self.basis is not None:
                planar = not isinstance(geometry, Euclidean)
                if planar != (self.basis.kind == "planar_tree"):
                    raise ValueError(
                        "Precomputed branched coefficients must match the target geometry's BCK or MKW basis."
                    )
            return self
        if self.control is None:
            raise RuntimeError("No sampled control or precomputed coefficients.")
        control_ts = jnp.asarray(getattr(self.control, "ts"))
        ys = jnp.asarray(getattr(self.control, "ys"))
        if self.ts.ndim != 1:
            raise ValueError("signature_knots must be one-dimensional.")
        intervals = self.ts.shape[0] - 1
        samples = control_ts.shape[0] - 1
        if intervals < 1:
            raise ValueError("signature_knots must contain at least two points.")
        if samples < intervals or samples % intervals:
            raise ValueError(
                "signature_knots must evenly subdivide the control sample grid."
            )
        stride = samples // intervals
        ts = eqx.error_if(
            self.ts,
            (~jnp.isfinite(self.ts)).any()
            | (self.ts[1:] <= self.ts[:-1]).any()
            | (self.ts != control_ts[::stride]).any(),
            "signature_knots must be finite, strictly increasing, and equal control.ts[::stride].",
        )
        windows = ys[stride * jnp.arange(intervals)[:, None] + jnp.arange(stride + 1)]
        basis, coeffs = self._compute(windows, geometry)
        out = copy(self)
        object.__setattr__(out, "ts", ts)
        object.__setattr__(out, "basis", basis)
        object.__setattr__(out, "coeffs", coeffs)
        return out

    def _compute(
        self, windows: Array, geometry: Manifold[Any]
    ) -> tuple[CoefficientBasis, Array]:
        raise NotImplementedError


def _check_dimensions(input_dim: int, depth: int) -> None:
    for name, value in (("input_dim", input_dim), ("depth", depth)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


class LogSignatureInterpolation(_SignatureInterpolation):
    """Piecewise-linear log coefficients for LogODE, LinearMagnus and LinearFer."""

    @classmethod
    def from_logsignatures(
        cls,
        ts: Array,
        coeffs: Array,
        input_dim: int,
        depth: int,
    ) -> LogSignatureInterpolation:
        """Load local method-1 Lyndon log-signatures, without a scalar term."""
        _check_dimensions(input_dim, depth)
        return cls._from_coefficients(
            ts, coeffs, make_lyndon_basis(int(depth), int(input_dim)), "stratonovich"
        )

    def _compute(
        self, windows: Array, geometry: Manifold[Any]
    ) -> tuple[CoefficientBasis, Array]:
        dim = windows.shape[-1]
        if self.solution == "stratonovich":
            pysiglib.prepare_log_sig(dim, self.depth, 1)
            return make_lyndon_basis(self.depth, dim), pysiglib.log_sig(
                windows, self.depth
            )
        planar = not isinstance(geometry, Euclidean)
        basis = (
            make_planar_tree_basis(self.depth, dim)
            if planar
            else make_tree_basis(self.depth, dim)
        )
        pysiglib.prepare_branched_log_sig(dim, self.depth, 0, planar=planar)
        return basis, pysiglib.branched_log_sig(
            windows, self.depth, method=0, planar=planar, correction=self.correction
        )

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Array:
        del left
        if self.coeffs is None:
            raise ValueError("LogSignatureInterpolation must be materialised first.")
        t0 = eqx.error_if(
            jnp.asarray(t0),
            ~((t0 >= self.ts[0]) & (t0 <= self.ts[-1])),
            "LogSignatureInterpolation times must lie within the signature knot range.",
        )
        if t1 is None:
            return self._evaluate(t0)
        t1 = eqx.error_if(
            jnp.asarray(t1),
            ~((t1 >= self.ts[0]) & (t1 <= self.ts[-1])),
            "LogSignatureInterpolation times must lie within the signature knot range.",
        )
        lower, upper = jnp.minimum(t0, t1), jnp.maximum(t0, t1)
        index = jnp.clip(
            jnp.searchsorted(self.ts, lower, side="right") - 1,
            0,
            self.coeffs.shape[0] - 1,
        )
        increment = ((t1 - t0) / (self.ts[index + 1] - self.ts[index])) * self.coeffs[
            index
        ]
        return eqx.error_if(
            increment,
            upper > self.ts[index + 1],
            "LogSignatureInterpolation intervals may not cross signature knots; clip solver steps at the signature knots.",
        )

    def _evaluate(self, t: RealScalarLike) -> Array:
        assert self.coeffs is not None
        index = jnp.clip(
            jnp.searchsorted(self.ts, t, side="right") - 1, 0, self.coeffs.shape[0] - 1
        )
        cumulative = jnp.concatenate(
            [jnp.zeros_like(self.coeffs[:1]), jnp.cumsum(self.coeffs, axis=0)], axis=0
        )
        fraction = (t - self.ts[index]) / (self.ts[index + 1] - self.ts[index])
        return cumulative[index] + fraction * self.coeffs[index]


class SignatureInterpolation(_SignatureInterpolation):
    """Full local signatures for Davie, evaluated over adjacent signature knots.

    Geometric controls use tensor words. Itô controls use BCK trees in Euclidean space and MKW ordered forests on manifolds. Coefficients omit the scalar term. Use StepTo(ts); fractional and multi-knot increments are not defined by this container. Reverse increments use the signature group inverse.
    """

    @classmethod
    def from_signatures(
        cls,
        ts: Array,
        coeffs: Array,
        input_dim: int,
        depth: int,
        *,
        solution: Literal["ito", "stratonovich"] = "stratonovich",
        geometry: Manifold[Any] = Euclidean(),
    ) -> SignatureInterpolation:
        """Load local full signatures in PySigLib order, without a scalar term."""
        _check_dimensions(input_dim, depth)
        basis = cls._basis(int(depth), int(input_dim), solution, geometry)
        if basis.kind != "word":
            pysiglib.prepare_branched_sig(
                int(input_dim), int(depth), planar=basis.kind == "planar_tree"
            )
        return cls._from_coefficients(ts, coeffs, basis, solution)

    @staticmethod
    def _basis(
        depth: int, dim: int, solution: str, geometry: Manifold[Any]
    ) -> CoefficientBasis:
        if solution == "stratonovich":
            return make_word_basis(depth, dim)
        if solution != "ito":
            raise ValueError(f"Unknown solution type {solution!r}.")
        return (
            make_tree_basis(depth, dim)
            if isinstance(geometry, Euclidean)
            else make_planar_tree_basis(depth, dim)
        )

    def _compute(
        self, windows: Array, geometry: Manifold[Any]
    ) -> tuple[CoefficientBasis, Array]:
        dim = windows.shape[-1]
        basis = self._basis(self.depth, dim, self.solution, geometry)
        if basis.kind == "word":
            coeffs = pysiglib.sig(windows, self.depth)
        else:
            planar = basis.kind == "planar_tree"
            pysiglib.prepare_branched_sig(dim, self.depth, planar=planar)
            coeffs = pysiglib.branched_sig(
                windows, self.depth, planar=planar, correction=self.correction
            )
        return basis, coeffs

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Array:
        del left
        if self.coeffs is None or self.basis is None:
            raise ValueError("SignatureInterpolation must be materialised first.")
        if t1 is None:
            raise ValueError(
                "SignatureInterpolation requires two adjacent signature knots."
            )
        lower, upper = jnp.minimum(t0, t1), jnp.maximum(t0, t1)
        index = jnp.clip(
            jnp.searchsorted(self.ts, lower, side="right") - 1,
            0,
            self.coeffs.shape[0] - 1,
        )
        coeffs = eqx.error_if(
            self.coeffs[index],
            (t0 != t1) & ((lower != self.ts[index]) | (upper != self.ts[index + 1])),
            "SignatureInterpolation requires adjacent signature knots; use StepTo(control.ts).",
        )

        def inverse(value: Array) -> Array:
            assert self.basis is not None
            if self.basis.kind == "word":
                indices = {word: i for i, word in enumerate(self.basis.keys)}
                return jnp.stack(
                    [
                        (-1) ** len(word) * value[indices[word[::-1]]]
                        for word in self.basis.keys
                    ]
                )
            # The positive-degree convolution is triangular by degree. Each
            # iteration fixes another degree of S * S^{-1} = 1.
            result = -value
            for _ in range(self.depth - 1):
                result = result - pysiglib.branched_sig_combine(
                    value,
                    result,
                    self.basis.dim,
                    self.depth,
                    planar=self.basis.kind == "planar_tree",
                )
            return result

        value = jax.lax.cond(t1 < t0, inverse, lambda value: value, coeffs)
        return jnp.where(t0 == t1, jnp.zeros_like(value), value)


class RoughTerm(AbstractTerm[Array, Array]):
    """Diffrax term over rough-path coefficients."""

    vector_field: VectorField | ControlledVectorField
    control: LogSignatureInterpolation | SignatureInterpolation
    basis: CoefficientBasis = eqx.field(static=True)
    lifted_fields: tuple[LiftedField, ...] = eqx.field(static=True)
    has_lifted_vector_field: bool = eqx.field(static=True)
    geometry: Manifold[Any] = Euclidean()

    def __init__(
        self,
        vector_field: VectorField | ControlledVectorField,
        control: LogSignatureInterpolation | SignatureInterpolation,
        geometry: Manifold[Any] = Euclidean(),
        *,
        _has_lifted_vector_field: bool = False,
    ):
        if not isinstance(control, (LogSignatureInterpolation, SignatureInterpolation)):
            raise TypeError(
                "RoughTerm control must be a LogSignatureInterpolation or SignatureInterpolation."
            )
        if _has_lifted_vector_field and isinstance(control, SignatureInterpolation):
            raise ValueError("Lifted vector fields require LogSignatureInterpolation.")
        control = control.materialise(geometry)
        assert control.basis is not None

        if isinstance(vector_field, ControlledVectorField):
            if _has_lifted_vector_field:
                raise ValueError(
                    "ControlledVectorField supplies level-one coefficients, not lifted columns."
                )
            if control.solution != "stratonovich":
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
            if isinstance(control, SignatureInterpolation)
            or _has_lifted_vector_field
            or isinstance(vector_field, ControlledVectorField)
            else form_pseudo_bialgebra_map(vector_field, control.basis, geometry)
        )

    @classmethod
    def from_lifted_vector_field(
        cls,
        vector_field: VectorField,
        control: LogSignatureInterpolation,
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
        if isinstance(self.control, SignatureInterpolation):
            chart = self.geometry.select_pullback_chart(self.basis.depth)
            return rough_taylor_columns(
                self.vector_field, self.basis, self.geometry, chart, y
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


__all__ = [
    "LogSignatureInterpolation",
    "RoughTerm",
    "SignatureInterpolation",
    "unwrap_rough_term",
]
