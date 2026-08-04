#!/usr/bin/env python3
"""Generate homogeneous Fer factors with Hofstaetter's ``bch`` executable.

The executable is built from https://github.com/HaraldHofstaetter/BCH. For a
maximum depth ``p``, the symbols ``A``, ``B``, ... represent the homogeneous
Magnus components Omega_1, Omega_2, ... and are assigned weights 1, 2, ....
Hofstaetter's tabular multidegrees are used to recover this weighted grading.

The generated factors satisfy, through weighted degree ``p``,

    exp(Omega_1 + ... + Omega_p) = exp(F_1) ... exp(F_p).

Generation is deliberately an offline development task. Roughrax imports only
the generated exact-rational recipes and has no runtime dependency on ``bch``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
from typing import TypeAlias

LieWord: TypeAlias = int | tuple["LieWord", "LieWord"]


@dataclass(frozen=True)
class LieTerm:
    coefficient: Fraction
    word: LieWord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bch", type=Path, help="Path to the built bch executable.")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Largest Fer depth to generate (default: 6).",
    )
    return parser.parse_args()


def _parse_word(text: str, generators: str) -> LieWord:
    def parse_at(position: int) -> tuple[LieWord, int]:
        token = text[position]
        if token in generators:
            return generators.index(token), position + 1
        if token != "[":
            raise ValueError(f"Expected a generator or '[', got {text[position:]!r}.")
        left, position = parse_at(position + 1)
        if text[position] != ",":
            raise ValueError(f"Expected ',' in {text!r} at position {position}.")
        right, position = parse_at(position + 1)
        if text[position] != "]":
            raise ValueError(f"Expected ']' in {text!r} at position {position}.")
        return (left, right), position + 1

    word, position = parse_at(0)
    if position != len(text):
        raise ValueError(f"Unexpected suffix in Lie word {text!r}.")
    return word


def _word_weight(word: LieWord) -> int:
    if isinstance(word, int):
        return word + 1
    return _word_weight(word[0]) + _word_weight(word[1])


def _run_bch(
    executable: Path,
    expression: str,
    maximum_depth: int,
    generators: str,
) -> list[LieTerm]:
    command = [
        str(executable),
        f"N={maximum_depth}",
        "basis=0",
        "table_output=1",
        "print_index=0",
        "print_multi_degree=1",
        "print_factors=0",
        "print_basis_element=1",
        f"expression={expression}",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    terms: list[LieTerm] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split("\t")
        if len(fields) != 4:
            continue
        _, multidegree_text, word_text, coefficient_text = fields
        coefficient = Fraction(coefficient_text)
        if coefficient == 0:
            continue
        multidegree = tuple(
            int(value) for value in multidegree_text.strip("()").split(",")
        )
        word = _parse_word(word_text, generators)
        table_weight = sum(
            (index + 1) * count for index, count in enumerate(multidegree)
        )
        if table_weight != _word_weight(word):
            raise RuntimeError(
                f"Inconsistent weighted degrees for {word_text}: "
                f"table gives {table_weight}, bracket tree gives {_word_weight(word)}."
            )
        terms.append(LieTerm(coefficient, word))
    return terms


def _word_expression(word: LieWord, generators: str) -> str:
    if isinstance(word, int):
        return generators[word]
    return (
        f"[{_word_expression(word[0], generators)},"
        f"{_word_expression(word[1], generators)}]"
    )


def _factor_expression(terms: list[LieTerm], generators: str) -> str:
    pieces = []
    for term in terms:
        sign = "+" if term.coefficient >= 0 else "-"
        magnitude = abs(term.coefficient)
        pieces.append(
            f"{sign}{magnitude.numerator}/{magnitude.denominator}*"
            f"{_word_expression(term.word, generators)}"
        )
    if not pieces:
        return "0/1*A"
    return "".join(pieces)


def _residual_expression(
    factors: list[list[LieTerm]],
    omega: str,
    generators: str,
) -> str:
    inverse_factors = "".join(
        f"exp(-({_factor_expression(factor, generators)}))*"
        for factor in reversed(factors)
    )
    return f"log({inverse_factors}exp({omega}))"


def _generate_factors(
    executable: Path,
    maximum_depth: int,
) -> tuple[str, list[list[LieTerm]]]:
    if maximum_depth < 1:
        raise ValueError("max-depth must be positive.")
    if maximum_depth > 26:
        raise ValueError("Hofstaetter's default executable has only 26 generators.")
    generators = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:maximum_depth]
    omega = "+".join(generators)
    factors: list[list[LieTerm]] = []

    for depth in range(1, maximum_depth + 1):
        residual = _run_bch(
            executable,
            _residual_expression(factors, omega, generators),
            maximum_depth,
            generators,
        )
        lower_terms = [term for term in residual if _word_weight(term.word) < depth]
        if lower_terms:
            raise RuntimeError(
                f"Residual before F_{depth} still contains lower-weight terms: "
                f"{lower_terms!r}."
            )
        factor = [term for term in residual if _word_weight(term.word) == depth]
        if not factor:
            raise RuntimeError(f"Generated factor F_{depth} is empty.")
        factors.append(factor)

    return generators, factors


def _verify_factorization(
    executable: Path,
    maximum_depth: int,
    generators: str,
    factors: list[list[LieTerm]],
) -> None:
    product = "*".join(
        f"exp({_factor_expression(factor, generators)})" for factor in factors
    )
    logarithm = _run_bch(
        executable,
        f"log({product})",
        maximum_depth,
        generators,
    )
    actual = {
        term.word: term.coefficient
        for term in logarithm
        if _word_weight(term.word) <= maximum_depth
    }
    expected: dict[LieWord, Fraction] = {
        index: Fraction(1) for index in range(maximum_depth)
    }
    if actual != expected:
        raise RuntimeError(
            "Generated Fer product failed exact BCH verification through weighted "
            f"degree {maximum_depth}: expected {expected!r}, got {actual!r}."
        )


def _source_revision(executable: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(executable.resolve().parent), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 else "unknown"


def _word_python(word: LieWord) -> str:
    if isinstance(word, int):
        return str(word)
    return f"({_word_python(word[0])}, {_word_python(word[1])})"


def _render_module(
    revision: str,
    maximum_depth: int,
    factors: list[list[LieTerm]],
) -> str:
    lines = [
        '"""Exact-rational homogeneous Fer recipes (generated; do not edit).',
        "",
        "Generated by ``tools/generate_fer_coefficients.py`` using Hofstaetter's",
        "BCH program: https://github.com/HaraldHofstaetter/BCH",
        f"Source revision: {revision}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TypeAlias",
        "",
        'LieWord: TypeAlias = int | tuple["LieWord", "LieWord"]',
        "FerTerm: TypeAlias = tuple[int, int, LieWord]",
        "",
        f"FER_MAX_DEPTH = {maximum_depth}",
        "FER_FACTORS: tuple[tuple[FerTerm, ...], ...] = (",
    ]
    for factor in factors:
        if len(factor) == 1:
            term = factor[0]
            lines.append(
                "    (("
                f"{term.coefficient.numerator}, {term.coefficient.denominator}, "
                f"{_word_python(term.word)}),),"
            )
            continue
        lines.append("    (")
        for term in factor:
            lines.append(
                "        "
                f"({term.coefficient.numerator}, {term.coefficient.denominator}, "
                f"{_word_python(term.word)}),"
            )
        lines.append("    ),")
    lines.extend([")", "", '__all__ = ["FER_FACTORS", "FER_MAX_DEPTH", "LieWord"]', ""])
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    executable = args.bch.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    generators, factors = _generate_factors(executable, args.max_depth)
    _verify_factorization(executable, args.max_depth, generators, factors)
    print(
        _render_module(
            _source_revision(executable),
            args.max_depth,
            factors,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
