from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np

# Variable-length integer lists are stored CSR-style: a flat `ids` array plus
# a `ptr` array of offsets, so item i's entries live at ids[ptr[i]:ptr[i+1]].
# The `Coproduct` extends that layout to a list-of-lists-of-(coeff, left, right)
# terms.


@dataclass(frozen=True, slots=True)
class Coproduct:
    """Packed coproduct table.

    For tree i, terms live at indices ``tree_ptr[i] : tree_ptr[i+1]``.
    Term k has coefficient ``coeff[k]`` and tensor factors
    ``left_ids[left_ptr[k] : left_ptr[k+1]] ⊗ right_ids[right_ptr[k] : right_ptr[k+1]]``.
    """

    tree_ptr: np.ndarray
    coeff: np.ndarray
    left_ptr: np.ndarray
    left_ids: np.ndarray
    right_ptr: np.ndarray
    right_ids: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class PrimitiveBasis:
    """An indexed basis of trees / words with packed children and coproduct."""

    kind: Literal["lyndon", "kauri", "kauri_planar"]
    depth: int
    dim: int
    degree: np.ndarray            # number of nodes / word length, per basis element
    keys: tuple[Hashable, ...]    # canonical key for each basis element
    root_colour: np.ndarray       # colour of root for letters/leaves; -1 otherwise
    child_ptr: np.ndarray         # CSR layout for unjoin-children ids
    child_ids: np.ndarray
    coproduct: Coproduct


# --------------------------------------------------------------------------- #
# Lyndon word basis
# --------------------------------------------------------------------------- #


def enumerate_lyndon_basis(depth: int, dim: int) -> list[np.ndarray]:
    """Lyndon words over an alphabet of size ``dim``, grouped by length.

    Returns ``levels`` where ``levels[k]`` is an int32 array of shape
    ``(n_k, k+1)`` containing every Lyndon word of length ``k+1``.
    Implementation: Duval's algorithm.
    """
    if depth < 0 or dim < 0:
        raise ValueError("depth and dim must be non-negative")

    levels: list[list[list[int]]] = [[] for _ in range(depth)]
    if depth > 0 and dim > 0:
        word = [-1]
        while word:
            word[-1] += 1
            m = len(word)
            levels[m - 1].append(list(word))
            while len(word) < depth:
                word.append(word[-m])
            while word and word[-1] == dim - 1:
                word.pop()

    return [
        np.asarray(rows, np.int32) if rows else np.empty((0, k + 1), np.int32)
        for k, rows in enumerate(levels)
    ]


def make_lyndon_basis(depth: int, dim: int) -> PrimitiveBasis:
    words = tuple(
        tuple(int(x) for x in row)
        for level in enumerate_lyndon_basis(depth, dim)
        for row in level
    )
    word_id = {w: i for i, w in enumerate(words)}

    def standard_factorization(w: tuple[int, ...]) -> tuple[int, ...]:
        """Split ``w`` into its longest proper Lyndon suffix and prefix.

        Returns the (left_id, right_id) pair, or ``()`` for letters.
        """
        if len(w) == 1:
            return ()
        for split in range(1, len(w)):
            left, right = w[:split], w[split:]
            if left in word_id and right in word_id:
                return (word_id[left], word_id[right])
        raise ValueError(f"could not split Lyndon word {w}")

    children = [standard_factorization(w) for w in words]
    # Trivial primitive coproduct on each word: 1·() ⊗ (w) + 1·(w) ⊗ ().
    coproduct_terms = [[(1, (), (i,)), (1, (i,), ())] for i in range(len(words))]

    child_ptr, child_ids = _pack_csr(children)
    return PrimitiveBasis(
        kind="lyndon",
        depth=depth,
        dim=dim,
        degree=np.asarray([len(w) for w in words], np.int32),
        keys=words,
        root_colour=np.asarray(
            [w[0] if len(w) == 1 else -1 for w in words], np.int32
        ),
        child_ptr=child_ptr,
        child_ids=child_ids,
        coproduct=_pack_coproduct(coproduct_terms),
    )


# --------------------------------------------------------------------------- #
# Kauri (rooted-tree) bases
# --------------------------------------------------------------------------- #


def make_kauri_tree_basis(depth: int, dim: int) -> PrimitiveBasis:
    return _make_kauri_basis("kauri", depth, dim, planar=False)


def make_kauri_planar_tree_basis(depth: int, dim: int) -> PrimitiveBasis:
    return _make_kauri_basis("kauri_planar", depth, dim, planar=True)


def _make_kauri_basis(
    kind: Literal["kauri", "kauri_planar"],
    depth: int,
    dim: int,
    *,
    planar: bool,
) -> PrimitiveBasis:
    import kauri as kr

    # Note the argument-order quirk between the two kauri enumerators.
    raw_trees = (
        kr.colored_planar_trees_up_to_order(depth, dim)
        if planar
        else kr.colored_trees(dim, depth)
    )
    trees = tuple(t for t in raw_trees if t.list_repr is not None)
    tree_id = {t: i for i, t in enumerate(trees)}
    coproduct_of = kr.mkw.coproduct if planar else kr.gl.coproduct

    def forest_ids(forest) -> tuple[int, ...]:
        return tuple(tree_id[t] for t in forest if t.list_repr is not None)

    children = [forest_ids(t.unjoin()) for t in trees]
    coproduct_terms = [
        [(int(c), forest_ids(left), forest_ids(right)) for c, left, right in coproduct_of(t)]
        for t in trees
    ]
    keys = tuple(t.list_repr if planar else t.sorted_list_repr() for t in trees)

    child_ptr, child_ids = _pack_csr(children)
    return PrimitiveBasis(
        kind=kind,
        depth=depth,
        dim=dim,
        degree=np.asarray([t.nodes() for t in trees], np.int32),
        keys=keys,
        root_colour=np.asarray([t.list_repr[-1] for t in trees], np.int32),
        child_ptr=child_ptr,
        child_ids=child_ids,
        coproduct=_pack_coproduct(coproduct_terms),
    )


# --------------------------------------------------------------------------- #
# CSR packing helpers
# --------------------------------------------------------------------------- #


def _pack_csr(rows) -> tuple[np.ndarray, np.ndarray]:
    """Flatten variable-length integer rows into ``(ptr, ids)`` CSR arrays."""
    ptr = [0]
    ids: list[int] = []
    for row in rows:
        ids.extend(row)
        ptr.append(len(ids))
    return np.asarray(ptr, np.int32), np.asarray(ids, np.int32)


def _pack_coproduct(terms_per_tree) -> Coproduct:
    tree_ptr, coeff = [0], []
    left_ptr, left_ids = [0], []
    right_ptr, right_ids = [0], []
    for terms in terms_per_tree:
        for c, left, right in terms:
            coeff.append(c)
            left_ids.extend(left)
            right_ids.extend(right)
            left_ptr.append(len(left_ids))
            right_ptr.append(len(right_ids))
        tree_ptr.append(len(coeff))
    return Coproduct(
        tree_ptr=np.asarray(tree_ptr, np.int32),
        coeff=np.asarray(coeff, np.int64),
        left_ptr=np.asarray(left_ptr, np.int32),
        left_ids=np.asarray(left_ids, np.int32),
        right_ptr=np.asarray(right_ptr, np.int32),
        right_ids=np.asarray(right_ids, np.int32),
    )


__all__ = [
    "Coproduct",
    "PrimitiveBasis",
    "enumerate_lyndon_basis",
    "make_lyndon_basis",
    "make_kauri_tree_basis",
    "make_kauri_planar_tree_basis",
]
