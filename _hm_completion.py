from __future__ import annotations

from collections import defaultdict
from functools import cache
from math import ceil, comb, sqrt
from typing import NamedTuple

import numpy as np


Tree = tuple[int, tuple["Tree", ...]]
Forest = tuple[Tree, ...]


class CompletionTableau(NamedTuple):
    """Static data for the degree-four log-completion tableau."""

    a: np.ndarray
    solve: np.ndarray
    target_map: np.ndarray
    orders: np.ndarray
    layer_bounds: tuple[int, int, int, int]
    feature_count: int
    condition_number: float


def _tree_size(tree: Tree) -> int:
    return 1 + sum(_tree_size(child) for child in tree[1])


@cache
def _labelled_trees(size: int, dimension: int) -> tuple[Tree, ...]:
    if size < 1:
        raise ValueError("size must be positive.")
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if size == 1:
        return tuple((label, ()) for label in range(dimension))

    pool = tuple(
        sorted(
            (
                tree
                for child_size in range(1, size)
                for tree in _labelled_trees(child_size, dimension)
            ),
            key=repr,
        )
    )
    child_forests: set[Forest] = set()

    def build(start: int, remaining: int, children: list[Tree]) -> None:
        if remaining == 0:
            child_forests.add(tuple(children))
            return
        for index in range(start, len(pool)):
            child = pool[index]
            child_size = _tree_size(child)
            if child_size <= remaining:
                build(index, remaining - child_size, [*children, child])

    build(0, size - 1, [])
    return tuple(
        sorted(
            (
                (root, children)
                for root in range(dimension)
                for children in child_forests
            ),
            key=repr,
        )
    )


@cache
def _forests_upto(max_nodes: int, dimension: int) -> tuple[Forest, ...]:
    if max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative.")
    pool = tuple(
        sorted(
            (
                tree
                for size in range(1, max_nodes + 1)
                for tree in _labelled_trees(size, dimension)
            ),
            key=repr,
        )
    )
    forests: set[Forest] = {()}

    def build(start: int, remaining: int, children: list[Tree]) -> None:
        if children:
            forests.add(tuple(children))
        for index in range(start, len(pool)):
            child = pool[index]
            child_size = _tree_size(child)
            if child_size <= remaining:
                build(index, remaining - child_size, [*children, child])

    build(0, max_nodes, [])
    return tuple(
        sorted(
            forests,
            key=lambda forest: (
                sum(_tree_size(tree) for tree in forest),
                repr(forest),
            ),
        )
    )


def _stage_tree_values(a: np.ndarray, dimension: int) -> dict[Tree, np.ndarray]:
    values: dict[Tree, np.ndarray] = {}
    for size in range(1, 4):
        for tree in _labelled_trees(size, dimension):
            root, children = tree
            if not children:
                values[tree] = np.sum(a[:, :, root], axis=1)
            else:
                child_product = np.prod(
                    np.stack([values[child] for child in children]),
                    axis=0,
                )
                values[tree] = a[:, :, root] @ child_product
    return values


def _feature_matrix(
    a: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, tuple[Forest, ...]]:
    forests = _forests_upto(3, dimension)
    values = _stage_tree_values(a, dimension)
    features = np.empty((a.shape[0], len(forests)), dtype=np.float64)
    for index, forest in enumerate(forests):
        if not forest:
            features[:, index] = 1.0
        else:
            features[:, index] = np.prod(
                np.stack([values[tree] for tree in forest]),
                axis=0,
            )
    return features, forests


@cache
def _shuffle_words(left: tuple[int, ...], right: tuple[int, ...]):
    if not left:
        return ((right, 1),)
    if not right:
        return ((left, 1),)

    counts: defaultdict[tuple[int, ...], int] = defaultdict(int)
    for word, coefficient in _shuffle_words(left[1:], right):
        counts[(left[0], *word)] += coefficient
    for word, coefficient in _shuffle_words(left, right[1:]):
        counts[(right[0], *word)] += coefficient
    return tuple(sorted(counts.items()))


@cache
def _iota_words(tree: Tree):
    words: dict[tuple[int, ...], int] = {(): 1}
    for child in tree[1]:
        child_words = dict(_iota_words(child))
        product_words: defaultdict[tuple[int, ...], int] = defaultdict(int)
        for left, left_coefficient in words.items():
            for right, right_coefficient in child_words.items():
                for word, shuffle_coefficient in _shuffle_words(left, right):
                    product_words[word] += (
                        left_coefficient
                        * right_coefficient
                        * shuffle_coefficient
                    )
        words = dict(product_words)
    return tuple(
        sorted(((*word, tree[0]), coefficient) for word, coefficient in words.items())
    )


def _flat_word_index(word: tuple[int, ...], dimension: int) -> int:
    offset = sum(dimension**degree for degree in range(1, len(word)))
    lexical_index = 0
    for letter in word:
        lexical_index = dimension * lexical_index + letter
    return offset + lexical_index


def _target_map(
    dimension: int,
    forests: tuple[Forest, ...],
) -> tuple[np.ndarray, np.ndarray]:
    tensor_size = sum(dimension**degree for degree in range(1, 5))
    target = np.zeros(
        (len(forests) * dimension, tensor_size),
        dtype=np.float64,
    )
    orders = np.empty((len(forests),), dtype=np.int32)

    for forest_index, forest in enumerate(forests):
        orders[forest_index] = 1 + sum(_tree_size(tree) for tree in forest)
        for root in range(dimension):
            row = forest_index * dimension + root
            tree = (root, forest)
            for word, coefficient in _iota_words(tree):
                target[row, _flat_word_index(word, dimension)] += coefficient
    return target, orders


def _random_layered_tableau(
    dimension: int,
    counts: tuple[int, int, int],
    seed: int,
    scale: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    stage_count = 1 + sum(counts)
    a = np.zeros((stage_count, stage_count, dimension), dtype=np.float64)
    generator = np.random.default_rng(seed)
    bounds = (
        1,
        1 + counts[0],
        1 + counts[0] + counts[1],
        stage_count,
    )
    for layer, count in enumerate(counts):
        start = bounds[layer]
        stop = bounds[layer + 1]
        predecessors = start
        a[start:stop, :predecessors, :] = generator.normal(
            scale=scale / sqrt(max(predecessors, 1)),
            size=(count, predecessors, dimension),
        )
    return a, bounds


def _design_parameters(
    dimension: int,
    feature_count: int,
) -> tuple[tuple[int, int, int], int, float]:
    # The two-dimensional design was selected by an independent degree-five
    # B-series residual search and then checked on held-out global trajectories.
    # Higher dimensions use an overdetermined generic design.
    if dimension == 2:
        return (10, 16, 21), 109, 0.4

    total_stages = ceil(1.4 * feature_count)
    first = comb(dimension + 3, 3) - 1
    second = ceil(0.35 * feature_count)
    third = total_stages - 1 - first - second
    if third < 1:
        raise RuntimeError("Invalid completion-tableau layer allocation.")
    return (first, second, third), 26, 0.4


@cache
def completion_tableau(dimension: int) -> CompletionTableau:
    """Construct a deterministic three-layer degree-four completion tableau."""
    if dimension < 1:
        raise ValueError("dimension must be positive.")

    forests = _forests_upto(3, dimension)
    feature_count = len(forests)
    counts, preferred_seed, scale = _design_parameters(dimension, feature_count)

    candidates = (preferred_seed,) if dimension == 2 else tuple(
        range(preferred_seed, preferred_seed + 8)
    )
    best: tuple[float, np.ndarray, np.ndarray, tuple[int, int, int, int]] | None = None
    for seed in candidates:
        a, bounds = _random_layered_tableau(
            dimension,
            counts,
            seed,
            scale,
        )
        features, candidate_forests = _feature_matrix(a, dimension)
        if candidate_forests != forests:
            raise RuntimeError("Completion feature ordering changed unexpectedly.")
        singular_values = np.linalg.svd(features, compute_uv=False)
        if singular_values[feature_count - 1] <= 1e-10:
            continue
        condition_number = float(
            singular_values[0] / singular_values[feature_count - 1]
        )
        solve = np.linalg.pinv(features.T, rcond=1e-12)
        if best is None or condition_number < best[0]:
            best = condition_number, a, solve, bounds

    if best is None:
        raise RuntimeError(
            f"Could not construct a full-rank completion tableau for d={dimension}."
        )

    condition_number, a, solve, bounds = best
    best_features, _ = _feature_matrix(a, dimension)
    residual = best_features.T @ solve - np.eye(feature_count)
    if np.max(np.abs(residual)) > 1e-9:
        raise RuntimeError("Completion moment solve failed its residual check.")

    target_map, orders = _target_map(dimension, forests)
    return CompletionTableau(
        a=a,
        solve=solve,
        target_map=target_map,
        orders=orders,
        layer_bounds=bounds,
        feature_count=feature_count,
        condition_number=condition_number,
    )


def completion_stage_count(dimension: int) -> int:
    return int(completion_tableau(dimension).a.shape[0])


__all__ = [
    "CompletionTableau",
    "completion_stage_count",
    "completion_tableau",
]
