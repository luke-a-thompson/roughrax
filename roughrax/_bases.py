from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import pysiglib


@dataclass(frozen=True, slots=True, eq=False)
class PrimitiveBasis:
    """A coefficient basis aligned with the signature backend output.

    The planar MKW backend returns expanded ordered-forest coordinates, even for
    log signatures, so this internal basis is not tree-only in that case.
    """

    kind: Literal["lyndon", "tree", "planar_tree"]
    depth: int
    dim: int
    degree: tuple[int, ...]  # number of nodes / word length, per basis element
    keys: tuple[Hashable, ...]  # canonical key for each basis/forest element
    root_colour: tuple[int | None, ...]  # colour of root, if there is one
    # Recursive child ids per basis element. For a planar multi-tree forest
    # (root_colour is None) these are the forest's constituent trees, not node
    # children; consumers left-nest-bracket them. Otherwise they are the node
    # children of the single tree/word.
    children: tuple[tuple[int, ...], ...]


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
        root_colour=tuple(w[0] if len(w) == 1 else None for w in words),
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
    def tree_degree(tree) -> int:
        return 1 + sum(tree_degree(child) for child in tree[:-1])

    if planar:
        # pySigLib indexes planar branched signatures by ordered forests of
        # planar trees. ``tree_to_idx`` accepts a single tree as shorthand for a
        # one-tree forest, but the full coefficient vector includes forests.
        forests = tuple(
            forest
            for forest in pysiglib.trees(dim, depth, planar=True)
            if forest is not None
        )
        expected = pysiglib.branched_sig_length(
            dim,
            depth,
            planar=True,
            scalar_term=False,
        )
        if len(forests) != expected:
            raise RuntimeError(
                "pysiglib planar tree enumeration does not match branched "
                "signature coefficient length"
            )
        forest_id = {forest: i for i, forest in enumerate(forests)}

        def single_tree_id(tree) -> int:
            return forest_id[(tree,)]

        def forest_degree(forest) -> int:
            return sum(tree_degree(tree) for tree in forest)

        def forest_children(forest) -> tuple[int, ...]:
            if len(forest) == 1:
                tree = forest[0]
                return tuple(single_tree_id(child) for child in tree[:-1])
            return tuple(single_tree_id(tree) for tree in forest)

        def forest_root_colour(forest) -> int | None:
            return forest[0][-1] if len(forest) == 1 else None

        return PrimitiveBasis(
            kind=kind,
            depth=depth,
            dim=dim,
            degree=tuple(forest_degree(forest) for forest in forests),
            keys=forests,
            root_colour=tuple(forest_root_colour(forest) for forest in forests),
            children=tuple(forest_children(forest) for forest in forests),
        )

    trees = tuple(t for t in pysiglib.trees(dim, depth, planar=False) if t is not None)
    tree_id = {t: i for i, t in enumerate(trees)}
    children = tuple(tuple(tree_id[child] for child in tree[:-1]) for tree in trees)

    return PrimitiveBasis(
        kind=kind,
        depth=depth,
        dim=dim,
        degree=tuple(tree_degree(tree) for tree in trees),
        keys=trees,
        root_colour=tuple(tree[-1] for tree in trees),
        children=children,
    )


__all__ = [
    "PrimitiveBasis",
    "make_lyndon_basis",
    "make_tree_basis",
    "make_planar_tree_basis",
]
