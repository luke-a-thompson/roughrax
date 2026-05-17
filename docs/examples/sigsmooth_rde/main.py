"""Compare Log-ODE and loopy-path ODE errors against fine Wong-Zakai."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

platforms = [p.strip() for p in os.environ.get("JAX_PLATFORMS", "cuda,cpu").split(",")]
if "cuda" in platforms and "cpu" not in platforms:
    platforms.append("cpu")
os.environ["JAX_PLATFORMS"] = ",".join(platforms)

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

from roughrax import LogODE, RoughTerm, signature_to_loopy_path

DEGREE = 2
PATH_DIM = 24
FINE_N = 2**10
COARSE_STRIDE = 4
T1 = 1.0
HURST = 0.25
SEED = 1
NOISE_SCALE = 0.25
DRIFT_0 = 0.12
DRIFT_1 = -0.05
DRIFT_SINE = 0.04
Y0 = 0.3
BENCHMARK_REPEATS = 5

REFERENCE_SOLVER = diffrax.Tsit5()
METHOD_SOLVER = diffrax.Heun()


def path_length(xs: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xs, axis=0), axis=-1)))


def vector_field(y):
    indices = jnp.arange(PATH_DIM)
    frequencies = (indices // 2 + 1).astype(jnp.asarray(y).dtype)
    return jnp.where(
        indices % 2 == 0,
        jnp.cos(frequencies * y),
        jnp.sin(frequencies * y),
    )


def control_vector_field(t, y, args):
    del t, args
    return vector_field(y)


def sample_driver() -> tuple[np.ndarray, np.ndarray]:
    if PATH_DIM < 1:
        raise ValueError("PATH_DIM must be positive.")

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
    drift = (
        ts[:, None] * slopes[None, :]
        + DRIFT_SINE * np.sin(2.0 * np.pi * channels[None, :] * ts[:, None] / T1)
    )
    return ts, NOISE_SCALE * noise + drift


def make_loopy_driver(
    ts: np.ndarray,
    xs: np.ndarray,
    coarse_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if xs.shape[-1] != PATH_DIM:
        raise ValueError(f"Expected driver dimension {PATH_DIM}, got {xs.shape[-1]}.")

    loop_ts = []
    loop_xs = []
    basepoint = xs[0]
    max_sig_error = 0.0

    for left, right in zip(coarse_ts[:-1], coarse_ts[1:], strict=True):
        left_index, right_index = np.searchsorted(ts, [left, right])
        interval = np.array(xs[left_index : right_index + 1], copy=True, order="C")
        interval_sig = pysiglib.sig(interval, DEGREE)
        loop = signature_to_loopy_path(
            interval_sig,
            PATH_DIM,
            DEGREE,
            basepoint=basepoint,
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


def benchmark_lowered(fn) -> tuple[float, float, float, float]:
    start = time.perf_counter()
    compiled = jax.jit(fn).lower().compile()
    compile_time = time.perf_counter() - start

    value = jax.block_until_ready(compiled())
    times = []
    for _ in range(BENCHMARK_REPEATS):
        start = time.perf_counter()
        value = jax.block_until_ready(compiled())
        times.append(time.perf_counter() - start)

    return float(value), compile_time, min(times), float(np.mean(times))


def main() -> None:
    y0 = jnp.asarray(Y0)

    start = time.perf_counter()
    ts, xs = sample_driver()
    sample_time = time.perf_counter() - start
    coarse_ts = ts[::COARSE_STRIDE]
    fine_driver = diffrax.LinearInterpolation(ts=jnp.asarray(ts), ys=jnp.asarray(xs))

    start = time.perf_counter()
    rough_term = RoughTerm(
        vector_field,
        fine_driver,
        Euclidean(),
        depth=DEGREE,
        interval_ts=jnp.asarray(coarse_ts),
        solution="stratonovich",
    )
    logode_preprocess_time = time.perf_counter() - start

    start = time.perf_counter()
    loop_ts, loop_xs, sig_error = make_loopy_driver(ts, xs, coarse_ts)
    loop_driver = diffrax.LinearInterpolation(
        ts=jnp.asarray(loop_ts),
        ys=jnp.asarray(loop_xs),
    )
    loopy_preprocess_time = time.perf_counter() - start

    def solve_fine_wz():
        sol = diffrax.diffeqsolve(
            diffrax.ControlTerm(control_vector_field, fine_driver),
            REFERENCE_SOLVER,
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(jnp.asarray(ts)),
            saveat=diffrax.SaveAt(t1=True),
        )
        return sol.ys[-1]

    def solve_logode():
        sol = diffrax.diffeqsolve(
            rough_term,
            LogODE(METHOD_SOLVER),
            t0=float(coarse_ts[0]),
            t1=float(coarse_ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(jnp.asarray(coarse_ts)),
            saveat=diffrax.SaveAt(t1=True),
        )
        return sol.ys[-1]

    def solve_loopy_ode():
        sol = diffrax.diffeqsolve(
            diffrax.ControlTerm(control_vector_field, loop_driver),
            METHOD_SOLVER,
            t0=float(loop_ts[0]),
            t1=float(loop_ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(jnp.asarray(loop_ts)),
            saveat=diffrax.SaveAt(t1=True),
        )
        return sol.ys[-1]

    y_wz, wz_compile, wz_best, wz_mean = benchmark_lowered(solve_fine_wz)
    y_logode, logode_compile, logode_best, logode_mean = benchmark_lowered(solve_logode)
    y_loop, loop_compile, loop_best, loop_mean = benchmark_lowered(solve_loopy_ode)

    print("configuration")
    print(f"degree: {DEGREE}")
    print(f"path_dim: {PATH_DIM}")
    print(f"fine_n: {FINE_N}")
    print(f"coarse_stride: {COARSE_STRIDE}")
    print(f"coarse steps: {len(coarse_ts) - 1}")
    print(f"benchmark repeats: {BENCHMARK_REPEATS}")
    print(f"jax platforms: {os.environ['JAX_PLATFORMS']}")
    print(f"jax default backend: {jax.default_backend()}")
    print(f"jax devices: {jax.devices()}")

    print()
    print("preprocessing")
    print(f"sample driver: {sample_time:.3f}s")
    print(f"log-ode rough term: {logode_preprocess_time:.3f}s")
    print(f"loopy driver: {loopy_preprocess_time:.3f}s")

    print()
    print("path stats")
    print(f"fine driver points: {len(ts)}")
    print(f"fine driver length: {path_length(xs):.6f}")
    print(f"loopy ODE driver points: {len(loop_ts)}")
    print(f"loopy ODE driver length: {path_length(loop_xs):.6f}")
    print(f"max interval signature error: {sig_error:.3e}")

    print()
    print("terminal values")
    print(f"fine WZ y(T): {y_wz:.12f}")
    print(f"log-ode y(T): {y_logode:.12f}")
    print(f"loopy ODE y(T): {y_loop:.12f}")
    print(f"log-ode error vs fine WZ: {abs(y_logode - y_wz):.3e}")
    print(f"loopy ODE error vs fine WZ: {abs(y_loop - y_wz):.3e}")
    print(f"log-ode vs loopy difference: {abs(y_loop - y_logode):.3e}")

    print()
    print("timing, compiled solve only")
    print(
        "fine WZ timing: "
        f"compile={wz_compile:.3f}s, best={wz_best * 1e3:.3f}ms, "
        f"mean={wz_mean * 1e3:.3f}ms"
    )
    print(
        "log-ode timing: "
        f"compile={logode_compile:.3f}s, best={logode_best * 1e3:.3f}ms, "
        f"mean={logode_mean * 1e3:.3f}ms"
    )
    print(
        "loopy ODE timing: "
        f"compile={loop_compile:.3f}s, best={loop_best * 1e3:.3f}ms, "
        f"mean={loop_mean * 1e3:.3f}ms"
    )


if __name__ == "__main__":
    main()
