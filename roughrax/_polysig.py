from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import product
from math import ceil

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pysiglib.jax_api as pysiglib
from jaxtyping import Array
from numpy.typing import ArrayLike, NDArray

from roughrax._bases import make_lyndon_basis


@contextmanager
def _jax_x64_enabled():
    previous = bool(jax.config.jax_enable_x64)
    if not previous:
        jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        if not previous:
            jax.config.update("jax_enable_x64", False)


@dataclass(frozen=True, slots=True)
class PolynomialLogsignatureRealisation:
    """A polynomial driver matching one truncated log-signature."""

    coefficients: NDArray[np.floating]
    residual_norm: float
    iterations: int
    converged: bool
    polynomial_degree: int


@dataclass(frozen=True, slots=True)
class BatchedPolynomialLogsignatureRealisation:
    """Polynomial drivers matching a batch of truncated log-signatures."""

    coefficients: NDArray[np.floating]
    residual_norms: NDArray[np.floating]
    iterations: NDArray[np.integer]
    converged: NDArray[np.bool_]
    polynomial_degree: int


@dataclass(frozen=True, slots=True)
class _StaticData:
    dimension: int
    depth: int
    polynomial_degree: int
    prefix_ids: tuple[int, ...]
    last_letters: tuple[int, ...]
    legendre_coefficients: NDArray[np.floating]


def _words(dimension: int, depth: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        word
        for length in range(1, depth + 1)
        for word in product(range(dimension), repeat=length)
    )


def _shifted_legendre_coefficients(
    polynomial_degree: int,
    width: int,
) -> NDArray[np.floating]:
    coeffs = np.zeros((polynomial_degree + 1, width), dtype=np.float64)
    coeffs[0, 0] = 1.0
    if polynomial_degree == 0:
        return coeffs

    coeffs[1, 0] = -1.0
    coeffs[1, 1] = 2.0
    two_u_minus_one = np.array([-1.0, 2.0], dtype=np.float64)
    for r in range(1, polynomial_degree):
        shifted = np.convolve(two_u_minus_one, coeffs[r, : r + 1])
        coeffs[r + 1, : r + 2] = (
            (2 * r + 1) * shifted[: r + 2] - r * coeffs[r - 1, : r + 2]
        ) / (r + 1)
    return coeffs


@lru_cache(maxsize=None)
def _degree_two_area_form(polynomial_degree: int) -> NDArray[np.floating]:
    pair_integrals = np.zeros(
        (polynomial_degree + 1, polynomial_degree + 1),
        dtype=np.float64,
    )
    pair_integrals[0, 0] = 0.5
    if polynomial_degree >= 1:
        pair_integrals[0, 1] = 1.0 / 6.0

    for r in range(1, polynomial_degree + 1):
        denominator = 2.0 * (2 * r + 1)
        lower = r - 1
        pair_integrals[r, lower] -= 1.0 / (denominator * (2 * lower + 1))
        upper = r + 1
        if upper <= polynomial_degree:
            pair_integrals[r, upper] += 1.0 / (denominator * (2 * upper + 1))
    return 0.5 * (pair_integrals - pair_integrals.T)


@lru_cache(maxsize=None)
def _prepare_log_sig(dimension: int, depth: int, method: int) -> None:
    device = "cuda" if jax.default_backend() == "gpu" else "cpu"
    pysiglib.prepare_log_sig(dimension, depth, method, device=device)


@lru_cache(maxsize=None)
def _static_data(
    dimension: int,
    depth: int,
    polynomial_degree: int,
) -> _StaticData:
    if dimension < 1:
        raise ValueError("`dimension` must be positive.")
    if depth < 1:
        raise ValueError("`depth` must be positive.")
    if polynomial_degree < 0:
        raise ValueError("`polynomial_degree` must be non-negative.")

    words = _words(dimension, depth)
    word_to_id = {word: index for index, word in enumerate(words)}
    prefix_ids = tuple(
        -1 if len(word) == 1 else word_to_id[word[:-1]] for word in words
    )
    last_letters = tuple(word[-1] for word in words)
    width = depth * (polynomial_degree + 1) + 1
    return _StaticData(
        dimension=dimension,
        depth=depth,
        polynomial_degree=polynomial_degree,
        prefix_ids=prefix_ids,
        last_letters=last_letters,
        legendre_coefficients=_shifted_legendre_coefficients(
            polynomial_degree,
            width,
        ),
    )


def shifted_legendre_values(u: ArrayLike, polynomial_degree: int) -> Array:
    """Evaluate shifted Legendre polynomials ``P_0, ..., P_M`` at ``u``."""

    u = jnp.asarray(u)
    values = [jnp.ones((), dtype=u.dtype)]
    if polynomial_degree == 0:
        return jnp.stack(values)

    x = 2 * u - 1
    values.append(x)
    for r in range(1, polynomial_degree):
        values.append(((2 * r + 1) * x * values[-1] - r * values[-2]) / (r + 1))
    return jnp.stack(values)


def evaluate_legendre_expansion(u: ArrayLike, coefficients: ArrayLike) -> Array:
    """Evaluate ``sum_r coefficients[:, r] P_r(u)``."""

    coefficients = jnp.asarray(coefficients)
    values = shifted_legendre_values(u, coefficients.shape[-1] - 1)
    return coefficients @ values


def _signature_from_coefficients(coefficients: Array, data: _StaticData) -> Array:
    dtype = coefficients.dtype
    width = data.legendre_coefficients.shape[-1]
    legendre = jnp.asarray(data.legendre_coefficients, dtype=dtype)
    derivative_polys = coefficients @ legendre
    denominators = jnp.arange(1, width, dtype=dtype)
    zero = jnp.zeros(1, dtype=dtype)
    empty_poly = jnp.concatenate(
        [jnp.ones(1, dtype=dtype), jnp.zeros(width - 1, dtype=dtype)]
    )

    word_polys = []
    values = []
    for prefix_id, letter in zip(data.prefix_ids, data.last_letters, strict=True):
        prefix_poly = empty_poly if prefix_id == -1 else word_polys[prefix_id]
        product_coefficients = jnp.convolve(
            prefix_poly,
            derivative_polys[letter],
            mode="full",
        )
        poly = jnp.concatenate(
            [zero, product_coefficients[: width - 1] / denominators]
        )
        word_polys.append(poly)
        values.append(jnp.sum(poly))
    return jnp.stack(values)


def _logsignature_from_coefficients(
    flat_coefficients: Array,
    data: _StaticData,
) -> Array:
    coefficients = flat_coefficients.reshape(
        data.dimension,
        data.polynomial_degree + 1,
    )
    if data.depth == 2:
        area_form = jnp.asarray(
            _degree_two_area_form(data.polynomial_degree),
            dtype=coefficients.dtype,
        )
        area = coefficients @ area_form @ coefficients.T
        area_coordinates = [
            area[i, j]
            for i in range(data.dimension)
            for j in range(i + 1, data.dimension)
        ]
        return jnp.concatenate([coefficients[:, 0], jnp.stack(area_coordinates)])

    signature = _signature_from_coefficients(coefficients, data)
    _prepare_log_sig(data.dimension, data.depth, 1)
    return pysiglib.sig_to_log_sig(signature, data.dimension, data.depth, method=1)


def _least_squares_step(
    residual: Array,
    jacobian: Array,
    damping: float,
) -> Array:
    width = jacobian.shape[1]
    dtype = jacobian.dtype
    augmented_operator = jnp.concatenate(
        [
            jacobian,
            jnp.sqrt(jnp.asarray(damping, dtype=dtype)) * jnp.eye(width, dtype=dtype),
        ],
        axis=0,
    )
    augmented_rhs = jnp.concatenate(
        [-residual, jnp.zeros(width, dtype=dtype)],
        axis=0,
    )
    return lx.linear_solve(
        lx.MatrixLinearOperator(augmented_operator),
        augmented_rhs,
        lx.QR(),
    ).value


def _target_size(dimension: int, depth: int) -> int:
    return len(make_lyndon_basis(depth, dimension).keys)


def _minimum_polynomial_degree(
    dimension: int,
    depth: int,
    expected_size: int,
) -> int:
    minimum_degree = max(0, ceil(expected_size / dimension) - 1)
    if depth == 2:
        minimum_degree = max(minimum_degree, dimension)
    return minimum_degree


def _initial_coefficients(
    targets: NDArray[np.floating],
    dimension: int,
    polynomial_degree: int,
    *,
    perturbation: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    coefficients = np.zeros(
        (len(targets), dimension, polynomial_degree + 1),
        dtype=targets.dtype,
    )
    coefficients[:, :, 0] = targets[:, :dimension]
    if perturbation > 0.0 and polynomial_degree > 0:
        coefficients[:, :, 1:] = perturbation * rng.standard_normal(
            (len(targets), dimension, polynomial_degree),
        )
    return coefficients


@partial(
    jax.jit,
    static_argnames=(
        "dimension",
        "depth",
        "polynomial_degree",
        "max_iterations",
        "max_backtracking",
    ),
)
def _batched_gauss_newton(
    flat: Array,
    targets: Array,
    *,
    dimension: int,
    depth: int,
    sig_tol: float = 1e-6,
    initial_damping: float = 1e-8,
    polynomial_degree: int,
    max_iterations: int = 16,
    max_backtracking: int = 12,
) -> tuple[Array, Array, Array, Array]:
    data = _static_data(dimension, depth, polynomial_degree)

    def residual_fn(x, target):
        return _logsignature_from_coefficients(x, data) - target

    def residual_and_jacobian(x, target):
        def f(z):
            r = residual_fn(z, target)
            return r, r

        jacobian, residual = jax.jacrev(f, has_aux=True)(x)
        return residual, jacobian

    def step_fn(carry, iteration):
        flat, residual_norms, damping, iterations = carry
        active = residual_norms > sig_tol
        iterations = jnp.where(active, iteration, iterations)

        residuals, jacobians = jax.vmap(residual_and_jacobian)(flat, targets)
        residual_norms = jnp.linalg.norm(residuals, axis=1)
        active = residual_norms > sig_tol
        steps = jax.vmap(
            lambda residual, jacobian, lambda_: _least_squares_step(
                residual,
                jacobian,
                lambda_,
            )
        )(residuals, jacobians, damping)

        accepted0 = jnp.zeros(flat.shape[0], dtype=jnp.bool_)
        alpha0 = jnp.ones(flat.shape[0], dtype=flat.dtype)

        def backtrack_fn(backtrack_carry, _):
            best_flat, best_norms, accepted, alpha = backtrack_carry
            candidate = flat + alpha[:, None] * steps
            candidate_residuals = jax.vmap(residual_fn)(candidate, targets)
            candidate_norms = jnp.linalg.norm(candidate_residuals, axis=1)
            accept = (candidate_norms < residual_norms) & active & ~accepted
            best_flat = jnp.where(accept[:, None], candidate, best_flat)
            best_norms = jnp.where(accept, candidate_norms, best_norms)
            return (
                best_flat,
                best_norms,
                accepted | accept,
                0.5 * alpha,
            ), None

        (flat, residual_norms, accepted, _), _ = jax.lax.scan(
            backtrack_fn,
            (flat, residual_norms, accepted0, alpha0),
            None,
            length=max_backtracking,
        )
        damping = jnp.where(
            active & accepted,
            jnp.maximum(0.3 * damping, jnp.asarray(1e-14, dtype=damping.dtype)),
            jnp.where(active, 10.0 * damping, damping),
        )
        return (flat, residual_norms, damping, iterations), None

    residuals = jax.vmap(lambda x, target: residual_fn(x, target))(flat, targets)
    residual_norms = jnp.linalg.norm(residuals, axis=1)
    damping = jnp.full(flat.shape[0], initial_damping, dtype=flat.dtype)
    iterations = jnp.zeros(flat.shape[0], dtype=jnp.int32)
    (flat, residual_norms, _, iterations), _ = jax.lax.scan(
        step_fn,
        (flat, residual_norms, damping, iterations),
        jnp.arange(1, max_iterations + 1, dtype=jnp.int32),
    )
    return flat, residual_norms, iterations, residual_norms <= sig_tol


def _validated_polynomial_degree(
    dimension: int,
    depth: int,
    polynomial_degree: int | None,
    max_polynomial_degree: int | None,
) -> int:
    expected_size = _target_size(dimension, depth)
    minimum_degree = _minimum_polynomial_degree(dimension, depth, expected_size)
    if polynomial_degree is None:
        polynomial_degree = minimum_degree
    if polynomial_degree < minimum_degree:
        raise ValueError(
            "`polynomial_degree` gives too few coefficients: "
            f"{dimension * (polynomial_degree + 1)} < {expected_size}."
        )
    if max_polynomial_degree not in {None, polynomial_degree}:
        raise ValueError("The batched solver uses one fixed `polynomial_degree`.")
    return polynomial_degree


def realise_polynomial_logsignatures(
    targets: ArrayLike,
    dimension: int,
    depth: int,
    *,
    polynomial_degree: int | None = None,
    max_polynomial_degree: int | None = None,
    sig_tol: float = 1e-6,
    max_iterations: int = 16,
    initial_damping: float = 1e-8,
    max_backtracking: int = 12,
    perturbation: float = 0.0,
    seed: int | None = None,
    progress: bool = False,
) -> BatchedPolynomialLogsignatureRealisation:
    """Realise a batch of truncated Lyndon log-signatures as polynomial drivers.

    The Gauss-Newton subproblem is solved as the damped augmented least-squares
    system ``[J; sqrt(lambda) I] delta ~= [-R; 0]``. This avoids forming
    normal equations and therefore avoids squaring the Jacobian condition number.
    """

    target_array = np.asarray(targets, dtype=np.float64)
    if target_array.ndim != 2:
        raise ValueError("`targets` must be a rank-2 array.")
    expected_size = _target_size(dimension, depth)
    if target_array.shape[1] != expected_size:
        raise ValueError(
            "`targets` must have shape "
            f"(batch, {expected_size}), got {target_array.shape} instead."
        )

    polynomial_degree = _validated_polynomial_degree(
        dimension,
        depth,
        polynomial_degree,
        max_polynomial_degree,
    )
    if progress:
        print(
            f"batch={len(target_array)} degree={polynomial_degree} "
            f"iterations={max_iterations}",
            flush=True,
        )
    rng = np.random.default_rng(seed)
    coefficients = _initial_coefficients(
        target_array,
        dimension,
        polynomial_degree,
        perturbation=perturbation,
        rng=rng,
    )
    with _jax_x64_enabled():
        flat, residual_norms, iterations, converged = _batched_gauss_newton(
            jnp.asarray(coefficients.reshape(len(target_array), -1)),
            jnp.asarray(target_array),
            dimension=dimension,
            depth=depth,
            polynomial_degree=polynomial_degree,
            sig_tol=sig_tol,
            initial_damping=initial_damping,
            max_iterations=max_iterations,
            max_backtracking=max_backtracking,
        )
        flat, residual_norms, iterations, converged = jax.block_until_ready(
            (flat, residual_norms, iterations, converged)
        )
    coefficients = np.asarray(flat).reshape(
        len(target_array),
        dimension,
        polynomial_degree + 1,
    )
    residual_norms = np.asarray(residual_norms)
    iterations = np.asarray(iterations)
    converged = np.asarray(converged)

    if progress:
        print(
            f"converged={np.count_nonzero(converged)}/{len(converged)} "
            f"max_residual={np.max(residual_norms):.3e}",
            flush=True,
        )
    return BatchedPolynomialLogsignatureRealisation(
        coefficients=coefficients,
        residual_norms=residual_norms,
        iterations=iterations,
        converged=converged,
        polynomial_degree=polynomial_degree,
    )


def realise_polynomial_logsignature(
    target: ArrayLike,
    dimension: int,
    depth: int,
    *,
    polynomial_degree: int | None = None,
    max_polynomial_degree: int | None = None,
    sig_tol: float = 1e-6,
    max_iterations: int = 16,
    initial_damping: float = 1e-8,
    max_backtracking: int = 12,
    perturbation: float = 0.0,
    seed: int | None = None,
    progress: bool = False,
) -> PolynomialLogsignatureRealisation:
    """Realise one truncated Lyndon log-signature as a polynomial smooth driver."""

    target_array = np.asarray(target, dtype=np.float64)
    expected_size = _target_size(dimension, depth)
    if target_array.shape != (expected_size,):
        raise ValueError(
            "`target` must have shape "
            f"({expected_size},), got {target_array.shape} instead."
        )

    result = realise_polynomial_logsignatures(
        target_array[None],
        dimension,
        depth,
        polynomial_degree=polynomial_degree,
        max_polynomial_degree=max_polynomial_degree,
        sig_tol=sig_tol,
        max_iterations=max_iterations,
        initial_damping=initial_damping,
        max_backtracking=max_backtracking,
        perturbation=perturbation,
        seed=seed,
        progress=progress,
    )
    return PolynomialLogsignatureRealisation(
        coefficients=result.coefficients[0],
        residual_norm=float(result.residual_norms[0]),
        iterations=int(result.iterations[0]),
        converged=bool(result.converged[0]),
        polynomial_degree=result.polynomial_degree,
    )


def polynomial_signature(
    coefficients: ArrayLike,
    depth: int,
) -> Array:
    """Return tensor signature word coordinates for a polynomial driver."""

    coefficients = jnp.asarray(coefficients)
    dimension, modes = coefficients.shape
    data = _static_data(dimension, depth, modes - 1)
    return _signature_from_coefficients(coefficients, data)


def polynomial_lyndon_logsignature(
    coefficients: ArrayLike,
    depth: int,
) -> Array:
    """Return Lyndon log-signature coordinates for a polynomial driver."""

    coefficients = jnp.asarray(coefficients)
    dimension, modes = coefficients.shape
    data = _static_data(dimension, depth, modes - 1)
    return _logsignature_from_coefficients(coefficients.reshape(-1), data)


__all__ = [
    "BatchedPolynomialLogsignatureRealisation",
    "PolynomialLogsignatureRealisation",
    "evaluate_legendre_expansion",
    "polynomial_lyndon_logsignature",
    "polynomial_signature",
    "realise_polynomial_logsignature",
    "realise_polynomial_logsignatures",
    "shifted_legendre_values",
]
