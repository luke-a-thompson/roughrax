from __future__ import annotations

from functools import lru_cache

import numpy as np
import pysiglib
from numpy.typing import ArrayLike, NDArray

from roughrax._bases import make_lyndon_basis


@lru_cache(maxsize=None)
def _lyndon_basis_increments(
    dimension: int, degree: int
) -> tuple[tuple[int, ...], tuple[NDArray[np.floating], ...]]:
    """Unit-residual path increments for each Lyndon basis element.

    Letters give a single unit step in their colour; bracketed words give the
    commutator loop ``[L, R]`` (forward L, forward R, reverse L, reverse R).
    pysiglib returns Lyndon words in length-then-lex order, so a child's index
    is always lower than its parent's and a single forward pass suffices.
    """
    basis = make_lyndon_basis(degree, dimension)
    increments: list[NDArray[np.floating]] = []
    for children, colour in zip(basis.children, basis.root_colour):
        if not children:
            inc = np.zeros((1, dimension), dtype=np.float64)
            inc[0, colour] = 1.0
        else:
            left, right = increments[children[0]], increments[children[1]]
            inc = np.concatenate([left, right, -left[::-1], -right[::-1]], axis=0)
        increments.append(inc)
    return basis.degree, tuple(increments)


def nonstandard_wong_zakai(
    signature: ArrayLike,
    dimension: int,
    degree: int,
    *,
    basepoint: ArrayLike | None = None,
    atol: float = 1e-12,
    n_jobs: int = 1,
) -> NDArray[np.floating]:
    """Construct a piecewise-linear path with the same truncated signature.

    The construction works in Lyndon log-signature coordinates. It walks through
    the Lyndon basis in increasing degree and appends a line segment or
    commutator loop whose first non-zero log-signature coordinate corrects the
    current residual. Corrections at one degree do not affect lower degrees, so
    later loops repair the higher-order terms introduced by BCH multiplication.

    The returned path starts at ``basepoint`` when provided, and otherwise starts
    at the origin. Translation does not change the signature.
    """
    if dimension < 1:
        raise ValueError("`dimension` must be positive.")
    if degree < 1:
        raise ValueError("`degree` must be positive.")

    sig = np.asarray(signature, dtype=np.float64)
    expected = pysiglib.sig_length(dimension, degree)
    if sig.shape not in {(expected,), (expected + 1,)}:
        raise ValueError(
            "`signature` must be a single truncated signature with shape "
            f"({expected},) or ({expected + 1},), got {sig.shape} instead."
        )

    pysiglib.prepare_log_sig(dimension, degree, 2, device="cpu")
    target_logsig = np.asarray(
        pysiglib.sig_to_log_sig(sig, dimension, degree, method=2, n_jobs=n_jobs),
        dtype=np.float64,
    )
    current_logsig = np.zeros_like(target_logsig)
    basis_degrees, basis_increments = _lyndon_basis_increments(dimension, degree)

    pieces: list[NDArray[np.floating]] = []
    for index, basis_degree in enumerate(basis_degrees):
        residual = float(target_logsig[index] - current_logsig[index])
        if abs(residual) <= atol:
            continue

        correction = abs(residual) ** (1.0 / basis_degree) * basis_increments[index]
        if residual < 0.0:
            correction = -correction[::-1]
        pieces.append(correction)

        for displacement in correction:
            current_logsig = np.asarray(
                pysiglib.log_sig_join(
                    current_logsig,
                    np.ascontiguousarray(displacement),
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

    start = (
        np.zeros(dimension, dtype=np.float64)
        if basepoint is None
        else np.asarray(basepoint, dtype=np.float64)
    )
    if start.shape != (dimension,):
        raise ValueError(
            f"`basepoint` must have shape ({dimension},), got {start.shape} instead."
        )

    path = np.empty((len(increments) + 1, dimension), dtype=np.float64)
    path[0] = start
    path[1:] = start + np.cumsum(increments, axis=0)
    return path


__all__ = ["nonstandard_wong_zakai"]
