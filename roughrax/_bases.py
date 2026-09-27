from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import factorial, prod
from typing import Hashable, Literal

import pysiglib


@dataclass(frozen=True, slots=True, eq=False)
class CoefficientBasis:
    """A coefficient basis aligned with the signature backend output.

    Full planar MKW signatures and method-0 log signatures share the expanded
    ordered-forest basis. Their realisations as operators and vector fields differ.
    """

    kind: Literal["word", "lyndon", "tree", "planar_tree"]
    depth: int
    dim: int
    degree: tuple[int, ...]  # number of nodes / word length, per basis element
    keys: tuple[Hashable, ...]  # canonical key for each basis/forest element
    root_colour: tuple[int | None, ...]  # colour of root, if there is one
    # Recursive child ids per basis element. For a planar multi-tree forest
    # (root_colour is None) these are the forest's constituent trees, not node
    # children. Log realisation brackets these trees; signature realisation
    # composes their frozen frame actions. Otherwise these are node children.
    children: tuple[tuple[int, ...], ...]
    # BCK tree symmetry factors, aligned with keys; unused by other bases.
    symmetry: tuple[int, ...] | None = None


# --------------------------------------------------------------------------- #
# Lyndon word basis
# --------------------------------------------------------------------------- #


def make_word_basis(depth: int, dim: int) -> CoefficientBasis:
    words = tuple(word for word in pysiglib.words(dim, depth) if word)
    return CoefficientBasis(
        kind="word",
        depth=depth,
        dim=dim,
        degree=tuple(map(len, words)),
        keys=words,
        root_colour=tuple(word[0] if len(word) == 1 else None for word in words),
        children=tuple(() for _ in words),
    )


def make_lyndon_basis(depth: int, dim: int) -> CoefficientBasis:
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

    return CoefficientBasis(
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


def make_tree_basis(depth: int, dim: int) -> CoefficientBasis:
    return _make_tree_basis("tree", depth, dim, planar=False)


def make_planar_tree_basis(depth: int, dim: int) -> CoefficientBasis:
    return _make_tree_basis("planar_tree", depth, dim, planar=True)


def _make_tree_basis(
    kind: Literal["tree", "planar_tree"],
    depth: int,
    dim: int,
    *,
    planar: bool,
) -> CoefficientBasis:
    keys = tuple(
        key for key in pysiglib.trees(dim, depth, planar=planar) if key is not None
    )
    key_id = {key: i for i, key in enumerate(keys)}

    def tree_degree(tree) -> int:
        return 1 + sum(tree_degree(child) for child in tree[:-1])

    if planar:
        # pySigLib indexes planar branched signatures by ordered forests of
        # planar trees. ``tree_to_idx`` accepts a single tree as shorthand for a
        # one-tree forest, but the full coefficient vector includes forests.
        expected = pysiglib.branched_sig_length(
            dim,
            depth,
            planar=True,
            scalar_term=False,
        )
        if len(keys) != expected:
            raise RuntimeError(
                "pysiglib planar tree enumeration does not match branched "
                "signature coefficient length"
            )
        def single_tree_id(tree) -> int:
            return key_id[(tree,)]

        def forest_degree(forest) -> int:
            return sum(tree_degree(tree) for tree in forest)

        def forest_children(forest) -> tuple[int, ...]:
            if len(forest) == 1:
                tree = forest[0]
                return tuple(single_tree_id(child) for child in tree[:-1])
            return tuple(single_tree_id(tree) for tree in forest)

        def forest_root_colour(forest) -> int | None:
            return forest[0][-1] if len(forest) == 1 else None

        return CoefficientBasis(
            kind=kind,
            depth=depth,
            dim=dim,
            degree=tuple(forest_degree(forest) for forest in keys),
            keys=keys,
            root_colour=tuple(forest_root_colour(forest) for forest in keys),
            children=tuple(forest_children(forest) for forest in keys),
        )

    children = tuple(tuple(key_id[child] for child in tree[:-1]) for tree in keys)

    # pySigLib enumerates trees by degree, so children precede their parents.
    symmetry: list[int] = []
    for child_ids in children:
        symmetry.append(
            prod(
                symmetry[child] ** count * factorial(count)
                for child, count in Counter(child_ids).items()
            )
        )

    return CoefficientBasis(
        kind=kind,
        depth=depth,
        dim=dim,
        degree=tuple(tree_degree(tree) for tree in keys),
        keys=keys,
        root_colour=tuple(tree[-1] for tree in keys),
        children=children,
        symmetry=tuple(symmetry),
    )


__all__ = [
    "CoefficientBasis",
    "make_lyndon_basis",
    "make_planar_tree_basis",
    "make_tree_basis",
    "make_word_basis",
]
