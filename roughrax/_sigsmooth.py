from __future__ import annotations

from functools import lru_cache

import numpy as np
import pysiglib
from numpy.typing import ArrayLike, NDArray

from roughrax._bases import PrimitiveBasis, make_lyndon_basis


FloatArray = NDArray[np.floating]


def _inverse_increments(increments: FloatArray) -> FloatArray:
    return -increments[::-1]


@lru_cache(maxsize=None)
def _lyndon_basis_increments(
    dimension: int,
    degree: int,
) -> tuple[PrimitiveBasis, tuple[FloatArray, ...]]:
    basis = make_lyndon_basis(degree, dimension)
    increments: list[FloatArray | None] = [None] * len(basis.keys)

    def build(index: int) -> FloatArray:
        cached = increments[index]
        if cached is not None:
            return cached

        child_ids = basis.children[index]
        if not child_ids:
            out = np.zeros((1, dimension), dtype=np.float64)
            out[0, basis.root_colour[index]] = 1.0
        else:
            left = build(child_ids[0])
            right = build(child_ids[1])
            out = np.concatenate(
                [left, right, _inverse_increments(left), _inverse_increments(right)],
                axis=0,
            )

        increments[index] = out
        return out

    return basis, tuple(build(index) for index in range(len(basis.keys)))


def _increments_to_path(
    increments: FloatArray,
    *,
    basepoint: ArrayLike | None,
    dimension: int,
) -> FloatArray:
    start = (
        np.zeros(dimension, dtype=increments.dtype)
        if basepoint is None
        else np.asarray(basepoint, dtype=increments.dtype)
    )
    if start.shape != (dimension,):
        raise ValueError(
            "`basepoint` must have shape "
            f"({dimension},), got {start.shape} instead."
        )

    path = np.empty((len(increments) + 1, dimension), dtype=increments.dtype)
    path[0] = start
    if len(increments):
        path[1:] = start + np.cumsum(increments, axis=0)
    return path


def _validate_signature(
    signature: ArrayLike,
    *,
    dimension: int,
    degree: int,
) -> FloatArray:
    if dimension < 1:
        raise ValueError("`dimension` must be positive.")
    if degree < 1:
        raise ValueError("`degree` must be positive.")

    sig = np.asarray(signature, dtype=np.float64)
    expected = pysiglib.sig_length(dimension, degree)
    valid_shapes = {(expected,), (expected + 1,)}
    if sig.shape not in valid_shapes:
        raise ValueError(
            "`signature` must be a single truncated signature with shape "
            f"({expected},) or ({expected + 1},), got {sig.shape} instead."
        )
    return sig


def signature_to_loopy_path(
    signature: ArrayLike,
    dimension: int,
    degree: int,
    *,
    basepoint: ArrayLike | None = None,
    atol: float = 1e-12,
    n_jobs: int = 1,
) -> FloatArray:
    """Construct a piecewise-linear path with the same truncated signature.

    The construction works in Lyndon log-signature coordinates. It walks through
    the Lyndon basis in increasing degree and appends a line segment or
    commutator loop whose first non-zero log-signature coordinate corrects the
    current residual. Corrections at one degree do not affect lower degrees, so
    later loops repair the higher-order terms introduced by BCH multiplication.

    The returned path starts at ``basepoint`` when provided, and otherwise starts
    at the origin. Translation does not change the signature.
    """

    sig = _validate_signature(signature, dimension=dimension, degree=degree)
    pysiglib.prepare_log_sig(dimension, degree, 2, device="cpu")

    target_logsig = np.asarray(
        pysiglib.sig_to_log_sig(sig, dimension, degree, method=2, n_jobs=n_jobs),
        dtype=np.float64,
    )
    current_logsig = np.zeros_like(target_logsig)
    basis, basis_increments = _lyndon_basis_increments(dimension, degree)
    pieces: list[FloatArray] = []

    order = sorted(
        range(len(basis_increments)),
        key=lambda i: (basis.degree[i], i),
    )
    for index in order:
        basis_path_increments = basis_increments[index]
        residual = float(target_logsig[index] - current_logsig[index])
        if abs(residual) <= atol:
            continue

        basis_degree = basis.degree[index]
        scale = abs(residual) ** (1.0 / basis_degree)
        correction = scale * basis_path_increments
        if residual < 0.0:
            correction = _inverse_increments(correction)
        pieces.append(correction)

        for displacement in correction:
            displacement = np.array(displacement, copy=True, order="C")
            current_logsig = np.asarray(
                pysiglib.log_sig_join(
                    current_logsig,
                    displacement,
                    dimension,
                    degree,
                    n_jobs=n_jobs,
                ),
                dtype=np.float64,
            )

    increments = (
        np.concatenate(pieces, axis=0)
        if pieces
        else np.zeros((0, dimension), dtype=np.float64)
    )
    return _increments_to_path(increments, basepoint=basepoint, dimension=dimension)


def sigsmooth(
    signature: ArrayLike,
    dimension: int,
    degree: int,
    *,
    basepoint: ArrayLike | None = None,
    atol: float = 1e-12,
    n_jobs: int = 1,
) -> FloatArray:
    """Alias for :func:`signature_to_loopy_path`."""

    return signature_to_loopy_path(
        signature,
        dimension,
        degree,
        basepoint=basepoint,
        atol=atol,
        n_jobs=n_jobs,
    )


__all__ = ["signature_to_loopy_path", "sigsmooth"]
