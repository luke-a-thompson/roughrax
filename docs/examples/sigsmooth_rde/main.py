"""Compare Log-ODE, loopy-path ODE, and polysig ODE against fine Wong-Zakai."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pysiglib
from fbm import fbm
from georax import Euclidean

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roughrax import (  # noqa: E402
    LogODE,
    RoughTerm,
    evaluate_legendre_expansion,
    nonstandard_wong_zakai,
    realise_polynomial_logsignatures,
)
from roughrax._bases import make_lyndon_basis  # noqa: E402

DEGREE = 3
PATH_DIM = 4
FINE_N = 2**10
COARSE_STRIDE = 4
T1 = 1.0
HURST = 0.25
SEED = 1
NOISE_SCALE = 0.25
DRIFT_0 = 0.12
DRIFT_1 = -0.05
DRIFT_SINE = 0.04
Y0 = jnp.asarray(0.3)
BENCHMARK_REPEATS = 1

REFERENCE_SOLVER = diffrax.Tsit5()
METHOD_SOLVER = diffrax.Heun()
ROUGHNESS_P = 1.0 / HURST + 0.1
POLYSIG_SIG_TOL = 2e-5
POLYSIG_MAX_ITERATIONS = 60
POLYSIG_POLYNOMIAL_DEGREE = 8
POLYSIG_ODE_ATOL_SAFETY = 1.0
POLYSIG_SEGMENT_MAX_STEPS = 128
PROGRESS_REFRESH_STEPS = 20


def progress_meter():
    return diffrax.TqdmProgressMeter(refresh_steps=PROGRESS_REFRESH_STEPS)


def path_length(xs: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xs, axis=0), axis=-1)))


def vector_field(y):
    indices = jnp.arange(PATH_DIM)
    frequencies = (indices // 2 + 1).astype(jnp.asarray(y).dtype)
    return jnp.where(
        indices % 2 == 0, jnp.cos(frequencies * y), jnp.sin(frequencies * y)
    )


def control_vector_field(t, y, args):
    del t, args
    return vector_field(y)


def sample_driver() -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(SEED)
    try:
        noise = np.stack(
            [
                fbm(n=FINE_N, hurst=HURST, length=T1, method="daviesharte")
                for _ in range(PATH_DIM)
            ],
            axis=-1,
        )
    finally:
        np.random.set_state(state)

    ts = np.linspace(0.0, T1, FINE_N + 1)
    channels = np.arange(1, PATH_DIM + 1, dtype=np.float64)
    slopes = np.linspace(DRIFT_0, DRIFT_1, PATH_DIM, dtype=np.float64)
    drift = ts[:, None] * slopes[None, :] + DRIFT_SINE * np.sin(
        2.0 * np.pi * channels[None, :] * ts[:, None] / T1
    )
    return ts, NOISE_SCALE * noise + drift


def make_loopy_driver(ts, xs, coarse_ts):
    loop_ts, loop_xs = [], []
    basepoint = xs[0]
    max_sig_error = 0.0

    for left, right in zip(coarse_ts[:-1], coarse_ts[1:], strict=True):
        left_index, right_index = np.searchsorted(ts, [left, right])
        interval = np.ascontiguousarray(xs[left_index : right_index + 1])
        interval_sig = pysiglib.sig(interval, DEGREE)
        loop = nonstandard_wong_zakai(
            interval_sig, PATH_DIM, DEGREE, basepoint=basepoint
        )
        max_sig_error = max(
            max_sig_error,
            float(np.max(np.abs(pysiglib.sig(loop, DEGREE) - interval_sig))),
        )

        interval_ts = np.linspace(left, right, len(loop))
        basepoint = loop[-1].copy()
        if loop_xs:
            interval_ts = interval_ts[1:]
            loop = loop[1:]
        loop_ts.append(interval_ts)
        loop_xs.append(loop)

    return np.concatenate(loop_ts), np.concatenate(loop_xs), max_sig_error


def make_polysig_targets(ts, xs, coarse_ts):
    pysiglib.prepare_log_sig(PATH_DIM, DEGREE, 1, device="cpu")
    indices = np.searchsorted(ts, coarse_ts)
    return np.stack(
        [
            pysiglib.log_sig(
                np.ascontiguousarray(xs[indices[j] : indices[j + 1] + 1]),
                DEGREE,
                method=1,
            )
            for j in range(len(indices) - 1)
        ]
    )


def polysig_global_vector_field(u, y, coefficients):
    window_count = coefficients.shape[0]
    window = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, window_count - 1)
    local_u = u - window.astype(u.dtype)
    dot_x = evaluate_legendre_expansion(local_u, coefficients[window])
    return jnp.tensordot(dot_x, vector_field(y), axes=1)


def polysig_segment_vector_field(u, y, coefficients):
    dot_x = evaluate_legendre_expansion(u, coefficients)
    return jnp.tensordot(dot_x, vector_field(y), axes=1)


def estimate_truncation_scales(logsignatures):
    basis_degrees = np.asarray(make_lyndon_basis(DEGREE, PATH_DIM).degree)
    scales = np.zeros(len(logsignatures))
    for index, logsig in enumerate(logsignatures):
        omega = 0.0
        for level in range(1, DEGREE + 1):
            level_norm = np.linalg.norm(logsig[basis_degrees == level])
            if level_norm > 0:
                omega = max(omega, level_norm ** (ROUGHNESS_P / level))
        scales[index] = omega ** ((DEGREE + 1) / ROUGHNESS_P)
    return scales


def format_stats(stats) -> str:
    def n(key):
        return int(np.asarray(stats[key]))

    return f"steps={n('num_steps')}, accepted={n('num_accepted_steps')}, rejected={n('num_rejected_steps')}"


def benchmark(fn):
    start = time.perf_counter()
    compiled = jax.jit(fn).lower().compile()
    compile_time = time.perf_counter() - start

    value, stats = jax.block_until_ready(compiled())
    times = []
    for _ in range(BENCHMARK_REPEATS):
        start = time.perf_counter()
        value, _ = jax.block_until_ready(compiled())
        times.append(time.perf_counter() - start)
    return float(value), stats, compile_time, min(times), float(np.mean(times))


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def step_to_solve(term, grid, solver, *, args=None):
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=float(grid[0]),
        t1=float(grid[-1]),
        dt0=None,
        y0=Y0,
        args=args,
        stepsize_controller=diffrax.StepTo(jnp.asarray(grid)),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=len(grid) + 16,
        progress_meter=progress_meter(),
    )
    return sol.ys[-1], sol.stats


def main() -> None:
    (ts, xs), sample_time = timed(sample_driver)
    coarse_ts = ts[::COARSE_STRIDE]
    fine_driver = diffrax.LinearInterpolation(ts=jnp.asarray(ts), ys=jnp.asarray(xs))

    rough_term, logode_pp = timed(
        lambda: RoughTerm(
            vector_field,
            fine_driver,
            Euclidean(),
            depth=DEGREE,
            interval_ts=jnp.asarray(coarse_ts),
            solution="stratonovich",
        )
    )

    def build_loopy():
        lts, lxs, sig_err = make_loopy_driver(ts, xs, coarse_ts)
        driver = diffrax.LinearInterpolation(ts=jnp.asarray(lts), ys=jnp.asarray(lxs))
        return lts, lxs, driver, sig_err

    (loop_ts, loop_xs, loop_driver, sig_error), loopy_pp = timed(build_loopy)

    def build_polysig():
        targets = make_polysig_targets(ts, xs, coarse_ts)
        scales = estimate_truncation_scales(targets)
        realisation = realise_polynomial_logsignatures(
            targets,
            PATH_DIM,
            DEGREE,
            polynomial_degree=POLYSIG_POLYNOMIAL_DEGREE,
            sig_tol=POLYSIG_SIG_TOL,
            max_iterations=POLYSIG_MAX_ITERATIONS,
            progress=True,
        )
        floor = np.finfo(np.asarray(Y0).dtype).eps
        atols = np.maximum.reduce(
            [
                POLYSIG_ODE_ATOL_SAFETY * scales,
                realisation.residual_norms,
                np.full_like(scales, floor),
            ]
        )
        return targets, scales, atols, realisation

    (polysig_targets, polysig_scales, polysig_atols, polysig_realisation), polysig_pp = (
        timed(build_polysig)
    )
    polysig_coefficients = jnp.asarray(polysig_realisation.coefficients, dtype=Y0.dtype)
    polysig_ode_atols = jnp.asarray(polysig_atols, dtype=Y0.dtype)
    polysig_rtol = 0.0

    print(
        f"polysig ODE tolerance: p={ROUGHNESS_P:.3f}, rtol={polysig_rtol:.3e}, "
        f"segment atol min={np.min(polysig_atols):.3e}, "
        f"median={np.median(polysig_atols):.3e}, max={np.max(polysig_atols):.3e}",
        flush=True,
    )

    def solve_fine_wz():
        return step_to_solve(
            diffrax.ControlTerm(control_vector_field, fine_driver), ts, REFERENCE_SOLVER
        )

    def solve_logode():
        return step_to_solve(rough_term, coarse_ts, LogODE(METHOD_SOLVER))

    def solve_loopy_ode():
        return step_to_solve(
            diffrax.ControlTerm(control_vector_field, loop_driver),
            loop_ts,
            METHOD_SOLVER,
        )

    def solve_polysig_ode():
        def segment_step(y, inputs):
            coefficients, atol = inputs
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(polysig_segment_vector_field),
                diffrax.Heun(),
                t0=jnp.asarray(0.0, dtype=Y0.dtype),
                t1=jnp.asarray(1.0, dtype=Y0.dtype),
                dt0=None,
                y0=y,
                args=coefficients,
                stepsize_controller=diffrax.PIDController(
                    rtol=polysig_rtol,
                    atol=atol,
                ),
                saveat=diffrax.SaveAt(t1=True),
                max_steps=POLYSIG_SEGMENT_MAX_STEPS * 12,
                progress_meter=diffrax.NoProgressMeter(),
            )
            stats = {
                "num_steps": sol.stats["num_steps"],
                "num_accepted_steps": sol.stats["num_accepted_steps"],
                "num_rejected_steps": sol.stats["num_rejected_steps"],
            }
            return sol.ys[-1], stats

        y, segment_stats = jax.lax.scan(
            segment_step,
            Y0,
            (polysig_coefficients, polysig_ode_atols),
        )
        stats = {key: jnp.sum(value) for key, value in segment_stats.items()}
        return y, stats

    solvers = [
        ("fine WZ", solve_fine_wz, sample_time),
        ("log-ode", solve_logode, logode_pp),
        ("loopy ODE", solve_loopy_ode, loopy_pp),
        ("polysig ODE", solve_polysig_ode, polysig_pp),
    ]
    results = {name: benchmark(fn) for name, fn, _ in solvers}
    y_ref = results["fine WZ"][0]

    print("\nconfiguration")
    print(f"  degree={DEGREE}, path_dim={PATH_DIM}, fine_n={FINE_N}")
    print(f"  coarse_stride={COARSE_STRIDE}, coarse_steps={len(coarse_ts) - 1}")
    print(f"  benchmark_repeats={BENCHMARK_REPEATS}")
    print(f"  polysig sig_tol={POLYSIG_SIG_TOL:.1e}, roughness p={ROUGHNESS_P:.3f}")
    print(
        f"  polysig ODE rtol/atol={polysig_rtol:.3e} / "
        f"{np.min(polysig_atols):.3e}..{np.max(polysig_atols):.3e}"
    )
    print(f"  jax backend={jax.default_backend()}, devices={jax.devices()}")

    print("\npreprocessing")
    for name, _, t in solvers:
        print(f"  {name}: {t:.3f}s")

    print("\npath stats")
    print(f"  fine driver: {len(ts)} points, length {path_length(xs):.6f}")
    print(
        f"  loopy ODE driver: {len(loop_ts)} points, length {path_length(loop_xs):.6f}"
    )
    print(f"  max interval signature error: {sig_error:.3e}")
    print(f"  polysig polynomial degree: {polysig_realisation.polynomial_degree}")
    print(
        f"  polysig residual: max={np.max(polysig_realisation.residual_norms):.3e}, "
        f"mean={np.mean(polysig_realisation.residual_norms):.3e}"
    )
    print(
        f"  polysig truncation scale: min={np.min(polysig_scales):.3e}, "
        f"max={np.max(polysig_scales):.3e}"
    )
    print(
        f"  polysig ODE atol: min={np.min(polysig_atols):.3e}, "
        f"median={np.median(polysig_atols):.3e}, max={np.max(polysig_atols):.3e}"
    )
    print(
        f"  polysig intervals converged: "
        f"{np.count_nonzero(polysig_realisation.converged)}/{len(polysig_realisation.converged)}"
    )

    print("\nterminal values")
    for name, _, _ in solvers:
        y = results[name][0]
        suffix = (
            "" if name == "fine WZ" else f"   error vs fine WZ={abs(y - y_ref):.3e}"
        )
        print(f"  {name} y(T)={y:.12f}{suffix}")

    print("\ntiming, compiled solve only")
    for name, _, _ in solvers:
        _, stats, compile_t, best, mean = results[name]
        print(
            f"  {name}: compile={compile_t:.3f}s, best={best * 1e3:.3f}ms, "
            f"mean={mean * 1e3:.3f}ms, {format_stats(stats)}"
        )


if __name__ == "__main__":
    main()
