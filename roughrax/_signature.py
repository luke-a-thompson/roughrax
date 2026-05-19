from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


def _is_lyndon_word(word: tuple[int, ...]) -> bool:
    return bool(word) and all(word < word[index:] for index in range(1, len(word)))


def _lyndon_standard_factorisation(
    word: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(word) <= 1:
        raise ValueError("A Lyndon factorisation needs a word of length at least 2.")

    for index in range(len(word) - 1, 0, -1):
        prefix = word[:index]
        suffix = word[index:]
        if _is_lyndon_word(prefix) and _is_lyndon_word(suffix):
            return prefix, suffix
    raise ValueError(f"Could not factor Lyndon word {word!r}.")


def _tensor_poly_product(
    left: dict[tuple[int, ...], float],
    right: dict[tuple[int, ...], float],
) -> dict[tuple[int, ...], float]:
    out: dict[tuple[int, ...], float] = {}
    for left_word, left_coeff in left.items():
        for right_word, right_coeff in right.items():
            word = left_word + right_word
            out[word] = out.get(word, 0.0) + left_coeff * right_coeff
    return {word: coeff for word, coeff in out.items() if coeff != 0.0}


@lru_cache(maxsize=None)
def _lyndon_tensor_expansion(
    word: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], float], ...]:
    if len(word) == 1:
        return ((word, 1.0),)

    left_word, right_word = _lyndon_standard_factorisation(word)
    left = dict(_lyndon_tensor_expansion(left_word))
    right = dict(_lyndon_tensor_expansion(right_word))
    out = _tensor_poly_product(left, right)
    for tensor_word, coeff in _tensor_poly_product(right, left).items():
        out[tensor_word] = out.get(tensor_word, 0.0) - coeff
    return tuple(
        sorted((tensor_word, coeff) for tensor_word, coeff in out.items() if coeff != 0.0)
    )


def _default_depth3_lyndon_keys(dim: int) -> tuple[tuple[int, ...], ...]:
    words: list[tuple[int, ...]] = []
    for length in range(1, 4):
        words.extend(
            word
            for word in np.ndindex(*(dim for _ in range(length)))
            if _is_lyndon_word(tuple(int(letter) for letter in word))
        )
    return tuple(words)


def _normalise_depth3_basis_keys(
    basis_keys,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(letter) for letter in key)
        for key in basis_keys
        if len(key) <= 3
    )


@lru_cache(maxsize=None)
def _log_tensor_maps(
    dim: int,
    basis_keys: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level1 = np.zeros((len(basis_keys), dim), dtype=np.float64)
    level2 = np.zeros((len(basis_keys), dim, dim), dtype=np.float64)
    level3 = np.zeros((len(basis_keys), dim, dim, dim), dtype=np.float64)

    for basis_index, key in enumerate(basis_keys):
        for word, coeff in _lyndon_tensor_expansion(key):
            if len(word) == 1:
                level1[(basis_index, *word)] += coeff
            elif len(word) == 2:
                level2[(basis_index, *word)] += coeff
            elif len(word) == 3:
                level3[(basis_index, *word)] += coeff

    return level1, level2, level3


def signature_tensors_from_log_signature(
    log_coeffs,
    basis_keys,
    dim: int,
):
    """Convert depth-3 Lyndon log coefficients to word-signature tensors."""

    basis_keys = _normalise_depth3_basis_keys(basis_keys)
    log_coeffs = jnp.asarray(log_coeffs)
    level1_map, level2_map, level3_map = _log_tensor_maps(dim, basis_keys)
    dtype = jnp.result_type(log_coeffs)
    coeffs = log_coeffs[..., : len(basis_keys)]
    level1_map = jnp.asarray(level1_map, dtype=dtype)
    level2_map = jnp.asarray(level2_map, dtype=dtype)
    level3_map = jnp.asarray(level3_map, dtype=dtype)

    lie1 = jnp.tensordot(coeffs, level1_map, axes=((-1,), (0,)))
    lie2 = jnp.tensordot(coeffs, level2_map, axes=((-1,), (0,)))
    lie3 = jnp.tensordot(coeffs, level3_map, axes=((-1,), (0,)))

    sig2 = lie2 + 0.5 * jnp.einsum("...i,...j->...ij", lie1, lie1)
    sig3 = (
        lie3
        + 0.5
        * (
            jnp.einsum("...i,...jk->...ijk", lie1, lie2)
            + jnp.einsum("...ij,...k->...ijk", lie2, lie1)
        )
        + (1.0 / 6.0) * jnp.einsum("...i,...j,...k->...ijk", lie1, lie1, lie1)
    )
    return lie1, sig2, sig3


def log_signature_to_depth3_words(log_coeffs, basis_keys=None, dim: int | None = None):
    """Convert depth-3 Lyndon log coefficients to flattened word signatures.

    The output is ordered as scalar term, all level-1 words, all level-2 words,
    then all level-3 words in lexicographic tensor order. If ``basis_keys`` and
    ``dim`` are omitted, the historical 2D pysiglib Lyndon basis is used.
    """

    if dim is None:
        dim = 2
    if basis_keys is None:
        basis_keys = _default_depth3_lyndon_keys(dim)

    dx, x2, x3 = signature_tensors_from_log_signature(log_coeffs, basis_keys, dim)
    return jnp.concatenate(
        [
            jnp.ones((*dx.shape[:-1], 1), dtype=dx.dtype),
            dx.reshape(*dx.shape[:-1], -1),
            x2.reshape(*x2.shape[:-2], -1),
            x3.reshape(*x3.shape[:-3], -1),
        ],
        axis=-1,
    )


__all__ = [
    "log_signature_to_depth3_words",
    "signature_tensors_from_log_signature",
]
