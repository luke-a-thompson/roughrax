from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import cache
from itertools import product
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import (
    AbstractStratonovichSolver,
    LocalLinearInterpolation,
    RESULTS,
)
from georax import Euclidean

from roughrax._bases import make_lyndon_basis
from roughrax._solver._hm_completion import (
    completion_stage_count,
    completion_tableau,
)
from roughrax._term import RoughTerm, unwrap_rough_term


def _validate_term(term: RoughTerm) -> None:
    if not isinstance(term.geometry, Euclidean):
        raise NotImplementedError(
            "HM-SigRK currently supports Euclidean state spaces only. "
            "Use LogODE with a geometric inner solver on manifolds."
        )
    if term.control.solution != "stratonovich":
        raise ValueError(
            "HM-SigRK requires a geometric/Stratonovich signature. "
            "It does not currently implement branched Itô order conditions."
        )
    if getattr(term, "has_lifted_vector_field", False):
        raise ValueError(
            "HM-SigRK requires the level-one vector fields, not a pre-lifted "
            "log-signature vector field."
        )
    if term.basis.kind != "lyndon":
        raise ValueError("HM-SigRK requires Lyndon log-signature coordinates.")
    if term.basis.depth < 3:
        raise ValueError("HM-SigRK requires SignatureInterpolation(depth >= 3).")

    degree_three_basis = make_lyndon_basis(3, term.basis.dim)
    size = len(degree_three_basis.keys)
    if term.basis.keys[:size] != degree_three_basis.keys:
        raise ValueError(
            "HM-SigRK expects PySigLib method-1 Lyndon-word ordering through "
            "degree three."
        )


def _normalise_vector_fields(term: RoughTerm, y):
    """Return ``(driver_dimension, *y.shape)`` stacked vector fields."""
    fields = jnp.asarray(term.vector_field(y))
    dimension = term.basis.dim
    leading_shape = (dimension, *jnp.shape(y))
    trailing_shape = (*jnp.shape(y), dimension)

    if fields.shape == trailing_shape and fields.shape != leading_shape:
        fields = jnp.moveaxis(fields, -1, 0)
    if fields.shape != leading_shape:
        raise ValueError(
            "vector_field(y) must have shape "
            f"{leading_shape} (fields on the leading axis) or "
            f"{trailing_shape} (fields on the trailing axis); got {fields.shape}."
        )
    return fields


@cache
def _quadratic_cloud_data(dimension: int):
    eye = np.eye(dimension)
    pairs = tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )
    pair_nodes = (
        np.stack([eye[left] + eye[right] for left, right in pairs])
        if pairs
        else np.empty((0, dimension))
    )
    cloud = np.concatenate(
        [
            np.zeros((1, dimension)),
            eye,
            2 * eye,
            pair_nodes,
        ],
        axis=0,
    )
    left_indices = np.asarray([left for left, _ in pairs], dtype=np.int32)
    right_indices = np.asarray([right for _, right in pairs], dtype=np.int32)
    return cloud, left_indices, right_indices


def _quadratic_cloud(dimension: int, dtype):
    cloud, left_indices, right_indices = _quadratic_cloud_data(dimension)
    return (
        jnp.asarray(cloud, dtype=dtype),
        jnp.asarray(left_indices),
        jnp.asarray(right_indices),
    )


def _bracket_expansion(left, right):
    expansion: defaultdict[tuple[int, ...], int] = defaultdict(int)
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            coefficient = left_coefficient * right_coefficient
            expansion[left_word + right_word] += coefficient
            expansion[right_word + left_word] -= coefficient
    return {word: value for word, value in expansion.items() if value != 0}


def _flat_word_index(word: tuple[int, ...], dimension: int) -> int:
    offset = sum(dimension**degree for degree in range(1, len(word)))
    lexical_index = 0
    for letter in word:
        lexical_index = dimension * lexical_index + letter
    return offset + lexical_index


@cache
def _method_one_log_to_tensor_map(dimension: int):
    """Sparse map from method-1 log coordinates to the expanded tensor log."""
    basis = make_lyndon_basis(3, dimension)
    expansions: list[dict[tuple[int, ...], int]] = []

    for key, child_ids in zip(basis.keys, basis.children, strict=True):
        if not isinstance(key, tuple):
            raise TypeError("Lyndon basis keys must be words.")
        if not child_ids:
            expansions.append({key: 1})
        else:
            left_id, right_id = child_ids
            expansions.append(
                _bracket_expansion(expansions[left_id], expansions[right_id])
            )

    bracket_coordinates: list[dict[int, int]] = []
    for index, key in enumerate(basis.keys):
        if expansions[index].get(key) != 1:
            raise RuntimeError("Lyndon bracket expansion must have unit diagonal.")

        coordinate: defaultdict[int, int] = defaultdict(int)
        coordinate[index] = 1
        for previous in range(index):
            coefficient = expansions[previous].get(key, 0)
            if coefficient == 0:
                continue
            for input_index, value in bracket_coordinates[previous].items():
                coordinate[input_index] -= coefficient * value
        bracket_coordinates.append(
            {
                input_index: value
                for input_index, value in coordinate.items()
                if value != 0
            }
        )

    expanded_coordinates: defaultdict[
        tuple[int, ...], defaultdict[int, int]
    ] = defaultdict(lambda: defaultdict(int))
    for expansion, coordinate in zip(
        expansions, bracket_coordinates, strict=True
    ):
        for word, expansion_coefficient in expansion.items():
            for input_index, coordinate_coefficient in coordinate.items():
                expanded_coordinates[word][input_index] += (
                    expansion_coefficient * coordinate_coefficient
                )

    output_indices: list[int] = []
    input_indices: list[int] = []
    values: list[int] = []
    for degree in range(1, 4):
        for word in product(range(dimension), repeat=degree):
            output_index = _flat_word_index(word, dimension)
            for input_index, value in sorted(
                expanded_coordinates[word].items()
            ):
                if value != 0:
                    output_indices.append(output_index)
                    input_indices.append(input_index)
                    values.append(value)

    return (
        np.asarray(output_indices, dtype=np.int32),
        np.asarray(input_indices, dtype=np.int32),
        np.asarray(values, dtype=np.int8),
        len(basis.keys),
    )


def _expanded_tensor_log_level_three(term: RoughTerm, log_signature):
    dimension = term.basis.dim
    output_indices, input_indices, values, log_size = (
        _method_one_log_to_tensor_map(dimension)
    )
    tensor_size = dimension + dimension**2 + dimension**3

    output_indices = jnp.asarray(output_indices)
    input_indices = jnp.asarray(input_indices)
    values = jnp.asarray(values, dtype=log_signature.dtype)
    expanded_log = jnp.zeros((tensor_size,), dtype=log_signature.dtype)
    expanded_log = expanded_log.at[output_indices].add(
        values * log_signature[:log_size][input_indices]
    )

    level_one_end = dimension
    level_two_end = level_one_end + dimension**2
    log_level_one = expanded_log[:level_one_end]
    log_level_two = expanded_log[level_one_end:level_two_end].reshape(
        dimension, dimension
    )
    log_level_three = expanded_log[level_two_end:].reshape(
        dimension, dimension, dimension
    )
    return log_level_one, log_level_two, log_level_three


def _tensor_signature_levels_four(term: RoughTerm, log_signature):
    """Canonical signature completion ``exp(log_signature[:3])`` through level 4."""
    log_level_one, log_level_two, log_level_three = (
        _expanded_tensor_log_level_three(term, log_signature)
    )

    level_one = log_level_one
    level_two = log_level_two + 0.5 * jnp.einsum(
        "i,j->ij", log_level_one, log_level_one
    )
    level_three = log_level_three
    level_three = level_three + 0.5 * jnp.einsum(
        "i,jk->ijk", log_level_one, log_level_two
    )
    level_three = level_three + 0.5 * jnp.einsum(
        "ij,k->ijk", log_level_two, log_level_one
    )
    level_three = level_three + (1.0 / 6.0) * jnp.einsum(
        "i,j,k->ijk", log_level_one, log_level_one, log_level_one
    )

    level_four = 0.5 * (
        jnp.einsum("i,jkl->ijkl", log_level_one, log_level_three)
        + jnp.einsum("ijk,l->ijkl", log_level_three, log_level_one)
        + jnp.einsum("ij,kl->ijkl", log_level_two, log_level_two)
    )
    level_four = level_four + (1.0 / 6.0) * (
        jnp.einsum("i,j,kl->ijkl", log_level_one, log_level_one, log_level_two)
        + jnp.einsum("i,jk,l->ijkl", log_level_one, log_level_two, log_level_one)
        + jnp.einsum("ij,k,l->ijkl", log_level_two, log_level_one, log_level_one)
    )
    level_four = level_four + (1.0 / 24.0) * jnp.einsum(
        "i,j,k,l->ijkl",
        log_level_one,
        log_level_one,
        log_level_one,
        log_level_one,
    )
    return level_one, level_two, level_three, level_four


def _tensor_signature_level_three(term: RoughTerm, log_signature):
    return _tensor_signature_levels_four(term, log_signature)[:3]


def _moment_weights(
    level_one,
    level_two,
    level_three,
    rho_scale: float,
    beta_scale: float,
):
    """Build the legacy homogeneous quadratic-cloud weights."""
    dimension = level_one.shape[0]
    cloud, pair_left, pair_right = _quadratic_cloud(
        dimension, level_one.dtype
    )

    base_rho = jnp.maximum(
        jnp.max(jnp.abs(level_one)),
        jnp.maximum(
            jnp.sqrt(jnp.max(jnp.abs(level_two))),
            jnp.cbrt(jnp.max(jnp.abs(level_three))),
        ),
    )
    rho = rho_scale * base_rho
    safe_rho = jnp.where(rho > 0, rho, jnp.ones_like(rho))

    beta = beta_scale * jnp.cbrt(
        jnp.max(jnp.abs(level_three), axis=(0, 1))
    )
    safe_beta = jnp.where(beta > 0, beta, jnp.ones_like(beta))

    zeta = jnp.einsum("qlk->kl", level_three) / (
        safe_beta[:, None] * safe_rho**2
    )

    mu_zero = (level_one - beta) / safe_rho
    mu_one = jnp.swapaxes(level_two, 0, 1) / safe_rho**2
    mu_one = mu_one - (beta / safe_rho)[:, None] * zeta

    symmetric_level_three = jnp.transpose(level_three, (2, 0, 1))
    symmetric_level_three = symmetric_level_three + jnp.transpose(
        level_three, (2, 1, 0)
    )
    mu_two = symmetric_level_three / safe_rho**3
    mu_two = mu_two - (beta / safe_rho)[:, None, None] * jnp.einsum(
        "kl,km->klm", zeta, zeta
    )

    diagonal = jnp.diagonal(mu_two, axis1=1, axis2=2)
    twice_axis_weights = 0.5 * (diagonal - mu_one)
    axis_weights = (
        2 * mu_one
        - diagonal
        - (jnp.sum(mu_two, axis=2) - diagonal)
    )

    if pair_left.size:
        pair_weights = mu_two[:, pair_left, pair_right]
        pair_sum = jnp.sum(pair_weights, axis=1)
    else:
        pair_weights = jnp.empty((dimension, 0), dtype=level_one.dtype)
        pair_sum = jnp.zeros((dimension,), dtype=level_one.dtype)

    origin_weight = (
        mu_zero
        - 1.5 * jnp.sum(mu_one, axis=1)
        + 0.5 * jnp.sum(diagonal, axis=1)
        + pair_sum
    )

    omega = jnp.concatenate(
        [
            origin_weight[None, :],
            jnp.swapaxes(axis_weights, 0, 1),
            jnp.swapaxes(twice_axis_weights, 0, 1),
            jnp.swapaxes(pair_weights, 0, 1),
        ],
        axis=0,
    )
    return cloud, omega, rho, safe_rho, beta, safe_beta


def _hm_sigrk3_step(
    term: RoughTerm,
    log_signature,
    y0,
    rho_scale: float,
    beta_scale: float,
):
    level_one, level_two, level_three = _tensor_signature_level_three(
        term, log_signature
    )
    dimension = term.basis.dim
    cloud, omega, rho, safe_rho, beta, safe_beta = _moment_weights(
        level_one,
        level_two,
        level_three,
        rho_scale,
        beta_scale,
    )

    fields_at_origin = _normalise_vector_fields(term, y0)
    core_displacements = jnp.tensordot(cloud, fields_at_origin, axes=1)
    core_stages = y0 + rho * core_displacements

    def fields_at(y):
        return _normalise_vector_fields(term, y)

    fields_at_nonzero_core_stages = jax.vmap(fields_at)(core_stages[1:])
    fields_at_core_stages = jnp.concatenate(
        [fields_at_origin[None, ...], fields_at_nonzero_core_stages], axis=0
    )
    core_increment = rho * jnp.einsum(
        "ik,ik...->...",
        omega,
        fields_at_core_stages,
    )

    fields_at_axis_stages = fields_at_core_stages[1 : 1 + dimension]
    carrier_coefficients = jnp.transpose(level_three, (2, 0, 1)) / (
        safe_beta[:, None, None] * safe_rho
    )
    carrier_displacements = jnp.einsum(
        "kql,ql...->k...",
        carrier_coefficients,
        fields_at_axis_stages,
    )
    carrier_stages = y0 + carrier_displacements
    fields_at_carrier_stages = jax.vmap(fields_at)(carrier_stages)
    carrier_diagonal = fields_at_carrier_stages[
        jnp.arange(dimension), jnp.arange(dimension)
    ]
    beta_shape = (dimension,) + (1,) * len(jnp.shape(y0))
    carrier_increment = jnp.sum(
        jnp.reshape(beta, beta_shape) * carrier_diagonal,
        axis=0,
    )

    return y0 + core_increment + carrier_increment


def _hm_sigrk3_c4_single_step(
    term: RoughTerm,
    log_signature,
    y0,
    rho_scale: float,
):
    level_one, level_two, level_three, level_four = (
        _tensor_signature_levels_four(term, log_signature)
    )
    base_rho = jnp.maximum(
        jnp.max(jnp.abs(level_one)),
        jnp.maximum(
            jnp.sqrt(jnp.max(jnp.abs(level_two))),
            jnp.cbrt(jnp.max(jnp.abs(level_three))),
        ),
    )
    rho = rho_scale * base_rho
    safe_rho = jnp.where(rho > 0, rho, jnp.ones_like(rho))

    dimension = term.basis.dim
    data = completion_tableau(dimension)
    tensor_signature = jnp.concatenate(
        [
            level_one.reshape(-1),
            level_two.reshape(-1),
            level_three.reshape(-1),
            level_four.reshape(-1),
        ]
    )
    target_map = jnp.asarray(data.target_map, dtype=tensor_signature.dtype)
    moments = target_map @ tensor_signature
    moments = moments.reshape(data.feature_count, dimension)
    orders = jnp.asarray(data.orders, dtype=tensor_signature.dtype)
    moments = moments / safe_rho ** orders[:, None]

    solve = jnp.asarray(data.solve, dtype=tensor_signature.dtype)
    update_weights = solve @ moments
    a = jnp.asarray(data.a, dtype=tensor_signature.dtype)

    def fields_at(y):
        return _normalise_vector_fields(term, y)

    fields = _normalise_vector_fields(term, y0)[None, ...]
    bounds = data.layer_bounds
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        layer_displacements = jnp.einsum(
            "ijd,jd...->i...",
            a[start:stop, :start, :],
            fields[:start],
        )
        layer_stages = y0 + rho * layer_displacements
        layer_fields = jax.vmap(fields_at)(layer_stages)
        fields = jnp.concatenate([fields, layer_fields], axis=0)

    increment = rho * jnp.einsum(
        "ik,ik...->...",
        update_weights,
        fields,
    )
    return y0 + increment


class HMSigRK3(AbstractStratonovichSolver[None]):
    r"""Legacy third-order homogeneous moment-completion SigRK method.

    This is the original quadratic-cloud construction. It matches every rough
    B-series condition through degree three, but its first unconstrained
    degree-four layer is not the canonical truncated-log-signature completion.
    Use :class:`HMSigRK3C4` for the completion-corrected method.
    """

    term_structure: ClassVar = RoughTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    rho_scale: float = eqx.field(static=True)
    beta_scale: float = eqx.field(static=True)

    def __init__(self, rho_scale: float = 1.0, beta_scale: float = 1.0):
        if not np.isfinite(rho_scale) or rho_scale <= 0:
            raise ValueError("rho_scale must be finite and positive.")
        if not np.isfinite(beta_scale) or beta_scale <= 0:
            raise ValueError("beta_scale must be finite and positive.")
        object.__setattr__(self, "rho_scale", float(rho_scale))
        object.__setattr__(self, "beta_scale", float(beta_scale))

    @staticmethod
    def num_stages(driver_dimension: int) -> int:
        if driver_dimension < 1:
            raise ValueError("driver_dimension must be positive.")
        return (
            (driver_dimension + 1) * (driver_dimension + 2) // 2
            + driver_dimension
        )

    def init(self, terms, t0, t1, y0, args) -> None:
        del t0, t1, y0, args
        _validate_term(unwrap_rough_term(terms))
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del args, solver_state, made_jump
        rough_term = unwrap_rough_term(terms)
        log_signature = terms.contr(t0, t1)
        y1 = _hm_sigrk3_step(
            rough_term,
            log_signature,
            y0,
            self.rho_scale,
            self.beta_scale,
        )
        return y1, None, dict(y0=y0, y1=y1), None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class HMSigRK3C4(AbstractStratonovichSolver[None]):
    r"""Degree-four log-completed, derivative-free HM-SigRK method.

    ``HMSigRK3C4`` retains depth-three signature input. On each internal
    substep it constructs the canonical level-four completion

    .. math::

        \exp_\otimes(L_1 + L_2 + L_3)^{(4)},

    and solves all labelled-tree order conditions through degree four by a
    fixed, three-layer, unisolvent B-series tableau. The update weights depend
    linearly on the realised completed signature; the stage coefficients are a
    fixed bounded tableau multiplied by the homogeneous rough gauge.

    The default uses two equal log-signature substeps. Since
    ``exp(L / 2) exp(L / 2) = exp(L)``, this preserves the same canonical
    completion whilst reducing the first unmatched degree-five tableau error.
    It requires no Lie-bracket vector fields, automatic differentiation,
    opposite probes, or derivative quotients.

    For a two-dimensional driver, one internal substep has 48 vector-field
    evaluation points arranged in three parallel layers. With the default two
    substeps this is six sequential batched vector-field evaluations.
    """

    term_structure: ClassVar = RoughTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    substeps: int = eqx.field(static=True)
    rho_scale: float = eqx.field(static=True)

    def __init__(self, substeps: int = 2, rho_scale: float = 1.0):
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer.")
        if not np.isfinite(rho_scale) or rho_scale <= 0:
            raise ValueError("rho_scale must be finite and positive.")
        object.__setattr__(self, "substeps", substeps)
        object.__setattr__(self, "rho_scale", float(rho_scale))

    @staticmethod
    def stages_per_substep(driver_dimension: int) -> int:
        if driver_dimension < 1:
            raise ValueError("driver_dimension must be positive.")
        return completion_stage_count(driver_dimension)

    @staticmethod
    def num_stages(driver_dimension: int, substeps: int = 2) -> int:
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer.")
        return substeps * HMSigRK3C4.stages_per_substep(driver_dimension)

    @staticmethod
    def num_sequential_batches(substeps: int = 2) -> int:
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer.")
        return 3 * substeps

    def init(self, terms, t0, t1, y0, args) -> None:
        del t0, t1, y0, args
        rough_term = unwrap_rough_term(terms)
        _validate_term(rough_term)
        completion_tableau(rough_term.basis.dim)
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del args, solver_state, made_jump
        rough_term = unwrap_rough_term(terms)
        log_signature = terms.contr(t0, t1) / self.substeps

        def body(_, y):
            return _hm_sigrk3_c4_single_step(
                rough_term,
                log_signature,
                y,
                self.rho_scale,
            )

        y1 = jax.lax.fori_loop(0, self.substeps, body, y0)
        return y1, None, dict(y0=y0, y1=y1), None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


__all__ = ["HMSigRK3", "HMSigRK3C4"]
