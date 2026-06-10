from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from roughrax._bases import PrimitiveBasis


def _safe_sqrt(x: Array) -> Array:
    """Square root with zero (rather than NaN) gradient at zero."""

    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def _assemble_increments(first: Array, p: Array, q: Array) -> Array:
    """Prepend ``first`` to triangle loops with legs ``p_k``, ``q_k``, ``-p_k - q_k``.

    Each closed triangle adds Lévy area ``(p ∧ q) / 2`` and no first-level
    increment, so cross-terms between segments vanish.
    """

    loops = jnp.stack([p, q, -p - q], axis=1).reshape(-1, first.shape[0])
    return jnp.concatenate([first[None, :], loops], axis=0)


def depth2_logsig_to_first_area(
    coeffs: Array,
    basis: PrimitiveBasis,
) -> tuple[Array, Array]:
    """Convert depth-2 Lyndon log-signature coefficients to first level and area."""

    if basis.kind != "lyndon":
        raise ValueError("depth-2 virtual path factorisation requires a Lyndon basis.")
    if basis.depth != 2:
        raise ValueError(
            "depth-2 virtual path factorisation requires basis.depth == 2."
        )

    singles = np.array([(n, *k) for n, k in enumerate(basis.keys) if len(k) == 1])
    pairs = np.array([(n, *k) for n, k in enumerate(basis.keys) if len(k) == 2])

    first = jnp.zeros((basis.dim,), coeffs.dtype).at[singles[:, 1]].set(
        coeffs[singles[:, 0]]
    )
    area = jnp.zeros((basis.dim, basis.dim), coeffs.dtype)
    if pairs.size:
        pair_coeffs = coeffs[pairs[:, 0]]
        area = area.at[pairs[:, 1], pairs[:, 2]].set(pair_coeffs)
        area = area.at[pairs[:, 2], pairs[:, 1]].set(-pair_coeffs)

    return first, area


def spectral_virtual_increments_depth2(
    coeffs: Array,
    basis: PrimitiveBasis,
) -> Array:
    """Return virtual increments matching a depth-2 geometric log-signature."""

    first, area = depth2_logsig_to_first_area(coeffs, basis)
    rank = basis.dim // 2
    if rank == 0:
        return first[None, :]

    # Eigenvalues of the Hermitian matrix iA come in +/- pairs, so the top
    # ``rank`` capture all of A; the clamp only absorbs numerical noise.
    eigvals, eigvecs = jnp.linalg.eigh(1j * area)
    scales = 2.0 * _safe_sqrt(jnp.maximum(eigvals[-rank:], 0.0))
    p = scales[:, None] * jnp.imag(eigvecs[:, -rank:]).T
    q = scales[:, None] * jnp.real(eigvecs[:, -rank:]).T

    return _assemble_increments(first, p, q)


def pairwise_virtual_increments_depth2(
    coeffs: Array,
    basis: PrimitiveBasis,
) -> Array:
    """Simple exact pairwise triangle factorisation."""

    first, area = depth2_logsig_to_first_area(coeffs, basis)
    i_idx, j_idx = jnp.triu_indices(basis.dim, k=1)
    pair_areas = area[i_idx, j_idx]

    scale = _safe_sqrt(2.0 * jnp.abs(pair_areas))
    rows = jnp.arange(pair_areas.shape[0])
    zeros = jnp.zeros((pair_areas.shape[0], basis.dim), area.dtype)
    p = zeros.at[rows, i_idx].set(scale)
    q = zeros.at[rows, j_idx].set(jnp.sign(pair_areas) * scale)

    return _assemble_increments(first, p, q)


def virtual_increments_depth2(
    coeffs: Array,
    basis: PrimitiveBasis,
    *,
    factorisation: Literal["spectral", "pairwise"] = "spectral",
) -> Array:
    """Return virtual first-level increments for a depth-2 log-signature."""

    if factorisation == "spectral":
        return spectral_virtual_increments_depth2(coeffs, basis)
    if factorisation == "pairwise":
        return pairwise_virtual_increments_depth2(coeffs, basis)
    raise ValueError(f"Unknown depth-2 virtual path factorisation {factorisation!r}.")


def depth2_first_area_from_increments(increments: Array) -> tuple[Array, Array]:
    """Compute first level and degree-2 log area of a piecewise-linear path."""

    # Area = (1/2) sum_{r<s} v_r ^ v_s. Summing explicit per-pair wedges keeps
    # exact float cancellations that a prefix-sum/matmul formulation loses.
    r_idx, s_idx = jnp.triu_indices(increments.shape[0], k=1)
    wedge = jnp.einsum("ki,kj->kij", increments[r_idx], increments[s_idx])
    area = 0.5 * jnp.sum(wedge - wedge.transpose(0, 2, 1), axis=0)
    return jnp.sum(increments, axis=0), area


__all__ = [
    "depth2_first_area_from_increments",
    "depth2_logsig_to_first_area",
    "pairwise_virtual_increments_depth2",
    "spectral_virtual_increments_depth2",
    "virtual_increments_depth2",
]
