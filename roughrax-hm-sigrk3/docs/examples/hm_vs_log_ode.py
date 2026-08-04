"""Compare HMSigRK3 and LogODE on the same degree-three log-signatures."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from georax import Euclidean

from roughrax import HMSigRK3, LogODE, RoughTerm, SignatureInterpolation


matplotlib.use("Agg")
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)


def to_numpy(value) -> np.ndarray:
    result = np.asarray(jax.block_until_ready(value))
    gc.collect()
    return result


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def control_vector_field(t, y, args):
    del t, args
    return rough_vector_field(y)


def brownian_paths(
    *, exponent: int, num_paths: int, seed: int, t1: float
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = 2**exponent
    ts = np.linspace(0.0, t1, num_steps + 1, dtype=np.float64)
    generator = np.random.default_rng(seed)
    increments = generator.normal(
        scale=np.sqrt(t1 / num_steps),
        size=(num_paths, num_steps, 2),
    )
    paths = np.concatenate(
        [
            np.zeros((num_paths, 1, 2), dtype=np.float64),
            np.cumsum(increments, axis=1),
        ],
        axis=1,
    )
    return ts, paths


@jax.jit
def solve_reference_batch(ts, paths, y0):
    def solve_one(path):
        term = diffrax.ControlTerm(
            control_vector_field,
            diffrax.LinearInterpolation(ts=ts, ys=path),
        )
        solution = diffrax.diffeqsolve(
            term,
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(ts),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=ts.shape[0] + 4,
        )
        return solution.ys[-1]

    return jax.vmap(solve_one)(paths)


def materialise_log_signatures(
    ts: np.ndarray,
    paths: np.ndarray,
    coarse_ts: np.ndarray,
) -> tuple[RoughTerm, jax.Array]:
    template = None
    coefficients = []
    ts_jax = jnp.asarray(ts)
    coarse_ts_jax = jnp.asarray(coarse_ts)

    for path in paths:
        control = SignatureInterpolation(
            diffrax.LinearInterpolation(ts=ts_jax, ys=jnp.asarray(path)),
            coarse_ts_jax,
            depth=3,
            solution="stratonovich",
        )
        term = RoughTerm(rough_vector_field, control, Euclidean())
        if template is None:
            template = term
        coefficients.append(np.asarray(term.control.coeffs))

    if template is None:
        raise ValueError("At least one path is required.")
    return template, jnp.asarray(np.stack(coefficients))


@eqx.filter_jit
def solve_from_coefficients_batch(
    term_template: RoughTerm,
    coefficients_batch: jax.Array,
    coarse_ts: jax.Array,
    y0: jax.Array,
    solver,
):
    def solve_one(coefficients):
        term = eqx.tree_at(
            lambda candidate: candidate.control.coeffs,
            term_template,
            coefficients,
        )
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=coarse_ts[0],
            t1=coarse_ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(coarse_ts),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=coarse_ts.shape[0] + 4,
        )
        return solution.ys[-1]

    return jax.vmap(solve_one)(coefficients_batch)


def time_compiled_solve(
    term_template,
    coefficients_batch,
    coarse_ts,
    y0,
    solver,
    repeats: int,
):
    start = time.perf_counter()
    values = solve_from_coefficients_batch(
        term_template,
        coefficients_batch,
        coarse_ts,
        y0,
        solver,
    )
    jax.block_until_ready(values)
    compile_and_first = time.perf_counter() - start

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        values = solve_from_coefficients_batch(
            term_template,
            coefficients_batch,
            coarse_ts,
            y0,
            solver,
        )
        jax.block_until_ready(values)
        timings.append(time.perf_counter() - start)
    return values, compile_and_first, float(np.median(timings))


def make_inner_solver(name: str):
    match name:
        case "tsit5":
            return diffrax.Tsit5()
        case "dopri5":
            return diffrax.Dopri5()
        case "heun":
            return diffrax.Heun()
    raise ValueError(f"Unknown LogODE inner solver {name!r}.")


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, float | int | str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in ("HMSigRK3", "LogODE"):
        selected = [row for row in rows if row["method"] == method]
        runtimes = np.asarray([row["median_seconds_per_path"] for row in selected])
        errors = np.asarray([row["endpoint_rmse"] for row in selected])
        steps = np.asarray([row["num_steps"] for row in selected])
        axes[0].loglog(runtimes, errors, marker="o", label=method)
        axes[1].loglog(steps, errors, marker="o", label=method)

    axes[0].set_xlabel("median solve time per path [s]")
    axes[0].set_ylabel("endpoint RMSE")
    axes[1].set_xlabel("number of rough steps")
    axes[1].set_ylabel("endpoint RMSE")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--fine-exponent", type=int, default=12)
    parser.add_argument(
        "--coarse-exponents",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8],
    )
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--y0", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--logode-inner",
        choices=("tsit5", "dopri5", "heun"),
        default="tsit5",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "hm_vs_log_ode.csv",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "hm_vs_log_ode.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_paths < 1:
        raise ValueError("--num-paths must be positive.")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if any(exponent >= args.fine_exponent for exponent in args.coarse_exponents):
        raise ValueError("Every coarse exponent must be below --fine-exponent.")

    ts, paths = brownian_paths(
        exponent=args.fine_exponent,
        num_paths=args.num_paths,
        seed=args.seed,
        t1=args.t1,
    )
    print("computing fine Wong--Zakai references")
    references = to_numpy(
        solve_reference_batch(
            jnp.asarray(ts),
            jnp.asarray(paths),
            jnp.asarray(args.y0),
        )
    )

    solvers = {
        "HMSigRK3": HMSigRK3(),
        "LogODE": LogODE(make_inner_solver(args.logode_inner)),
    }
    rows: list[dict[str, float | int | str]] = []

    for exponent in args.coarse_exponents:
        stride = 2 ** (args.fine_exponent - exponent)
        coarse_ts = ts[::stride]
        print(
            f"materialising degree-three log-signatures for {2**exponent} steps"
        )
        start = time.perf_counter()
        template, coefficients_batch = materialise_log_signatures(
            ts,
            paths,
            coarse_ts,
        )
        preprocessing_seconds = time.perf_counter() - start
        coarse_ts_jax = jnp.asarray(coarse_ts)

        for method, solver in solvers.items():
            values, compile_seconds, median_seconds = time_compiled_solve(
                template,
                coefficients_batch,
                coarse_ts_jax,
                jnp.asarray(args.y0),
                solver,
                args.repeats,
            )
            values_np = to_numpy(values)
            rmse = float(np.sqrt(np.mean((values_np - references) ** 2)))
            row = {
                "method": method,
                "num_steps": 2**exponent,
                "step_size": args.t1 / 2**exponent,
                "endpoint_rmse": rmse,
                "compile_and_first_seconds": compile_seconds,
                "median_batch_seconds": median_seconds,
                "median_seconds_per_path": median_seconds / args.num_paths,
                "signature_preprocessing_seconds": preprocessing_seconds,
                "num_paths": args.num_paths,
            }
            rows.append(row)
            print(
                f"{method:10s} steps={2**exponent:5d} "
                f"rmse={rmse:.4e} "
                f"steady={median_seconds / args.num_paths:.4e}s/path"
            )

    write_csv(rows, args.csv)
    plot_results(rows, args.plot)
    print(f"wrote {args.csv}")
    print(f"wrote {args.plot}")
    print(f"HMSigRK3 stages for d=2: {HMSigRK3.num_stages(2)}")


if __name__ == "__main__":
    main()
