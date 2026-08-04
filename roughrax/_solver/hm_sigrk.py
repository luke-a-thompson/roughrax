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
from roughrax._term import RoughTerm, unwrap_rough_term


def _validate_term(term: RoughTerm) -> None:
    if not isinstance(term.geometry, Euclidean):
        raise NotImplementedError(
            "HMSigRK3 currently supports Euclidean state spaces only. "
            "Use LogODE with a geometric inner solver on manifolds."
        )
    if term.control.solution != "stratonovich":
        raise ValueError(
            "HMSigRK3 requires a geometric/Stratonovich signature. "
            "It does not currently implement branched Itô order conditions."
        )
    if term.basis.kind != "lyndon":
        raise ValueError("HMSigRK3 requires Lyndon log-signature coordinates.")
    if term.basis.depth < 3:
        raise ValueError("HMSigRK3 requires SignatureInterpolation(depth >= 3).")

    degree_three_basis = make_lyndon_basis(3, term.basis.dim)
    size = len(degree_three_basis.keys)
    if term.basis.keys[:size] != degree_three_basis.keys:
        raise ValueError(
            "HMSigRK3 expects PySigLib method-1 Lyndon-word ordering through "
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
    """Sparse map from method-1 log coordinates to the expanded tensor log.

    PySigLib method 1 stores the coefficients of the expanded tensor logarithm
    at Lyndon words. These values are not, in dimensions above two, the
    coefficients of the standard Lyndon bracket basis. The triangular conversion
    below first recovers the bracket coefficients and then expands the brackets
    into words. All nonzero entries through degree three are small integers.
    """
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

    # If u_i is the tensor-log coefficient at the i-th Lyndon word and c_i is
    # the standard Lyndon-bracket coefficient, then u = M c with unit lower
    # triangular M. Store every c_i as a sparse integer linear form in u.
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


def _tensor_signature_level_three(term: RoughTerm, log_signature):
    """Recover tensor-signature levels one to three in pure JAX."""
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
    return level_one, level_two, level_three


def _moment_weights(
    level_one,
    level_two,
    level_three,
    rho_scale: float = 1.0,
    beta_scale: float = 1.0,
):
    """Build homogeneous quadratic-cloud weights for every root field.

    ``rho_scale`` rescales the common core-cloud radius, whilst ``beta_scale``
    rescales each root carrier's update weight. The moment targets are recomputed
    after both rescalings, so all degree-three elementary weights remain exact.
    """
    dimension = level_one.shape[0]
    cloud, pair_left, pair_right = _quadratic_cloud(
        dimension, level_one.dtype
    )

    raw_rho = jnp.maximum(
        jnp.max(jnp.abs(level_one)),
        jnp.maximum(
            jnp.sqrt(jnp.max(jnp.abs(level_two))),
            jnp.cbrt(jnp.max(jnp.abs(level_three))),
        ),
    )
    rho = jnp.asarray(rho_scale, dtype=level_one.dtype) * raw_rho
    safe_rho = jnp.where(raw_rho > 0, rho, jnp.ones_like(rho))

    raw_beta = jnp.cbrt(jnp.max(jnp.abs(level_three), axis=(0, 1)))
    beta = jnp.asarray(beta_scale, dtype=level_one.dtype) * raw_beta
    safe_beta = jnp.where(raw_beta > 0, beta, jnp.ones_like(beta))

    # zeta[root, field] is the degree-zero row sum of the carrier stage,
    # divided by rho.
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

    # omega[stage, root] follows the same ordering as ``cloud``:
    # 0, e_l, 2 e_l, and e_l + e_m for l < m.
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


def _carrier_stage_data(
    level_three,
    rho,
    safe_rho,
    beta,
    safe_beta,
    step_size,
    *,
    brownian_chain_correction: bool,
):
    """Return direct carrier coefficients and optional Brownian chain weights.

    Without the correction, the carrier coefficients are

    ``X[q, l, k] / (beta[k] * rho)``.

    For standard independent Brownian channels, the expected degree-four
    signature coefficient at the word ``(r, r, k, k)`` is ``h**2 / 8``. The
    optional correction creates this coefficient through

    ``H_k -> M_r -> K_{e_r} -> Y_n``.

    Its lower-order contribution is removed by subtracting the same coefficient
    from every direct ``H_k -> K_{e_q}`` edge carrying field ``k``. Consequently
    the carrier row sums and every degree-three chain weight are unchanged.
    """
    dimension = level_three.shape[0]
    direct_coefficients = jnp.transpose(level_three, (2, 0, 1)) / (
        safe_beta[:, None, None] * safe_rho
    )
    chain_coefficients = jnp.zeros_like(beta)

    if brownian_chain_correction:
        h = jnp.abs(jnp.asarray(step_size, dtype=level_three.dtype))
        active = (rho > 0) & (beta > 0)
        chain_coefficients = jnp.where(
            active,
            h**2 / (8.0 * safe_beta * safe_rho**2),
            jnp.zeros_like(beta),
        )

        # direct_coefficients[root, axis_stage, field]. For every root k and
        # every axis stage q, subtract D_k from the field-k edge. The d such
        # subtractions cancel the d new H_k -> M_r field-k row-sum terms.
        root_field_selector = jnp.eye(dimension, dtype=level_three.dtype)[
            :, None, :
        ]
        direct_coefficients = direct_coefficients - (
            chain_coefficients[:, None, None] * root_field_selector
        )

    return direct_coefficients, chain_coefficients


def _hm_sigrk3_step(
    term: RoughTerm,
    log_signature,
    y0,
    step_size=0.0,
    *,
    rho_scale: float = 1.0,
    beta_scale: float = 1.0,
    brownian_chain_correction: bool = False,
):
    level_one, level_two, level_three = _tensor_signature_level_three(
        term, log_signature
    )
    dimension = term.basis.dim
    cloud, omega, rho, safe_rho, beta, safe_beta = _moment_weights(
        level_one,
        level_two,
        level_three,
        rho_scale=rho_scale,
        beta_scale=beta_scale,
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
    core_increment = rho * jnp.tensordot(
        omega,
        fields_at_core_stages,
        axes=([0, 1], [0, 1]),
    )

    # The e_q stages occupy positions 1, ..., dimension in the quadratic cloud.
    fields_at_axis_stages = fields_at_core_stages[1 : 1 + dimension]
    carrier_coefficients, chain_coefficients = _carrier_stage_data(
        level_three,
        rho,
        safe_rho,
        beta,
        safe_beta,
        step_size,
        brownian_chain_correction=brownian_chain_correction,
    )
    direct_carrier_displacements = jnp.tensordot(
        carrier_coefficients,
        fields_at_axis_stages,
        axes=([1, 2], [0, 1]),
    )

    if brownian_chain_correction:
        # M_r = y0 + rho f_r(K_{e_r}). These are the only additional vector-field
        # evaluation points: one middle stage per driver coordinate.
        indices = jnp.arange(dimension)
        fields_r_at_axis_r = fields_at_axis_stages[indices, indices]
        middle_stages = y0 + rho * fields_r_at_axis_r
        fields_at_middle_stages = jax.vmap(fields_at)(middle_stages)

        # H_k receives D_k f_k(M_r) from every r. Summing over the middle-stage
        # axis leaves one field value for each carrier root k.
        middle_sum_by_field = jnp.sum(fields_at_middle_stages, axis=0)
        coefficient_shape = (dimension,) + (1,) * len(jnp.shape(y0))
        chain_carrier_displacements = jnp.reshape(
            chain_coefficients, coefficient_shape
        ) * middle_sum_by_field
        carrier_displacements = (
            direct_carrier_displacements + chain_carrier_displacements
        )
    else:
        carrier_displacements = direct_carrier_displacements

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


class HMSigRK3(AbstractStratonovichSolver[None]):
    r"""Experimental homogeneous moment-completion SigRK method.

    ``HMSigRK3`` is an explicit, derivative-free solver for Euclidean geometric
    rough differential equations. It reconstructs tensor-signature levels one
    to three from the method-1 Lyndon-word log-signature supplied by
    :class:`roughrax.SignatureInterpolation`. It then matches the one-node,
    chain, and cherry B-series moments with a homogeneous quadratic cloud and
    one carrier stage per driver coordinate.

    Parameters:
        rho_scale: Positive multiplier for the common quadratic-cloud radius.
            Changing it preserves every degree-three order condition but changes
            the leading degree-four error constant and conditioning.
        beta_scale: Positive multiplier for each root carrier scale. Changing it
            likewise preserves every degree-three order condition.
        brownian_chain_correction: Add one middle stage per driver coordinate and
            match the expected fully nested four-node chain coefficient
            ``E[X^(r,r,k,k)] = h**2 / 8`` for standard independent Brownian
            channels. This correction is Brownian-specific and is disabled by
            default, so the generic pathwise solver retains its original
            homogeneous behavior.

    The uncorrected method uses

    .. math::

        \frac{(d + 1)(d + 2)}{2} + d

    vector-field evaluation points. The optional Brownian chain correction adds
    exactly ``d`` middle stages. It preserves all pathwise degree-three
    elementary weights. It does *not* impose the other degree-four Brownian mean
    conditions, so it should be treated as an experimental diagnostic rather
    than a claimed strong-order-3/2 method.

    Notes:
        - Use ``SignatureInterpolation(depth=3, solution="stratonovich")``.
          Larger depths are accepted, but levels above three are ignored.
        - Use ``diffrax.StepTo(control.ts)`` so every solver step corresponds to
          one stored local log-signature.
        - The current implementation is Euclidean and fixed-step. It supplies
          no embedded error estimate.
        - The Brownian correction assumes unit covariance and independent driver
          channels. Do not enable it for a deterministic control, fractional
          Brownian motion, or a non-identity Brownian covariance without first
          changing the expected degree-four coefficient.
    """

    term_structure: ClassVar = RoughTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    rho_scale: float = eqx.field(static=True)
    beta_scale: float = eqx.field(static=True)
    brownian_chain_correction: bool = eqx.field(static=True)

    def __init__(
        self,
        rho_scale: float = 1.0,
        beta_scale: float = 1.0,
        *,
        brownian_chain_correction: bool = False,
    ):
        rho_scale = float(rho_scale)
        beta_scale = float(beta_scale)
        if not np.isfinite(rho_scale) or rho_scale <= 0:
            raise ValueError("rho_scale must be a positive finite number.")
        if not np.isfinite(beta_scale) or beta_scale <= 0:
            raise ValueError("beta_scale must be a positive finite number.")
        if not isinstance(brownian_chain_correction, bool):
            raise TypeError("brownian_chain_correction must be a bool.")

        object.__setattr__(self, "rho_scale", rho_scale)
        object.__setattr__(self, "beta_scale", beta_scale)
        object.__setattr__(
            self,
            "brownian_chain_correction",
            brownian_chain_correction,
        )

    @staticmethod
    def num_stages(
        driver_dimension: int,
        *,
        brownian_chain_correction: bool = False,
    ) -> int:
        """Number of vector-field evaluation points per rough step."""
        if driver_dimension < 1:
            raise ValueError("driver_dimension must be positive.")
        stages = (
            (driver_dimension + 1) * (driver_dimension + 2) // 2
            + driver_dimension
        )
        if brownian_chain_correction:
            stages += driver_dimension
        return stages

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
            t1 - t0,
            rho_scale=self.rho_scale,
            beta_scale=self.beta_scale,
            brownian_chain_correction=self.brownian_chain_correction,
        )
        return y1, None, dict(y0=y0, y1=y1), None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


__all__ = ["HMSigRK3"]
