from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import pysiglib


@dataclass(frozen=True, slots=True, eq=False)
class PrimitiveBasis:
    """An indexed basis of trees / words with recursive children."""

    kind: Literal["lyndon", "tree", "planar_tree"]
    depth: int
    dim: int
    degree: tuple[int, ...]  # number of nodes / word length, per basis element
    keys: tuple[Hashable, ...]  # canonical key for each basis element
    root_colour: tuple[int, ...]  # colour of root; -1 for non-letter Lyndon
    children: tuple[tuple[int, ...], ...]  # unjoin-child ids per basis element


# --------------------------------------------------------------------------- #
# Lyndon word basis
# --------------------------------------------------------------------------- #


def make_lyndon_basis(depth: int, dim: int) -> PrimitiveBasis:
    words = tuple(pysiglib.lyndon_words(dim, depth))
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

    children = tuple(standard_factorization(w) for w in words)

    return PrimitiveBasis(
        kind="lyndon",
        depth=depth,
        dim=dim,
        degree=tuple(len(w) for w in words),
        keys=words,
        root_colour=tuple(w[0] if len(w) == 1 else -1 for w in words),
        children=children,
    )


# --------------------------------------------------------------------------- #
# Rooted-tree bases
# --------------------------------------------------------------------------- #


def make_tree_basis(depth: int, dim: int) -> PrimitiveBasis:
    return _make_tree_basis("tree", depth, dim, planar=False)


def make_planar_tree_basis(depth: int, dim: int) -> PrimitiveBasis:
    return _make_tree_basis("planar_tree", depth, dim, planar=True)


def _make_tree_basis(
    kind: Literal["tree", "planar_tree"],
    depth: int,
    dim: int,
    *,
    planar: bool,
) -> PrimitiveBasis:
    trees = tuple(
        t
        for t in pysiglib.trees(dim, depth, tree_order="canonical", planar=planar)
        if t is not None
    )
    tree_id = {t: i for i, t in enumerate(trees)}

    def tree_degree(tree) -> int:
        return 1 + sum(tree_degree(child) for child in tree[:-1])

    children = tuple(tuple(tree_id[child] for child in t[:-1]) for t in trees)

    return PrimitiveBasis(
        kind=kind,
        depth=depth,
        dim=dim,
        degree=tuple(tree_degree(t) for t in trees),
        keys=trees,
        root_colour=tuple(t[-1] for t in trees),
        children=children,
    )


__all__ = [
    "PrimitiveBasis",
    "make_lyndon_basis",
    "make_tree_basis",
    "make_planar_tree_basis",
]
