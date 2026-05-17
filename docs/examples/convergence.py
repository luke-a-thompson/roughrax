"""Log-ODE and SO(3) RDE convergence against fine Wong-Zakai references."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diffrax
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from georax import CFEES25, Euclidean, GeometricTerm, RKMK, SO

from roughrax import LogODE, RoughTerm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def to_numpy_and_clear_cache(x) -> np.ndarray:
    out = np.asarray(jax.block_until_ready(x))
    jax.clear_caches()
    gc.collect()
    return out


def diffrax_vector_field(t, y, args):
    del t, args
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def so3_vector_field(y):
    return jnp.eye(3, dtype=y.dtype)


def brownian_samples(
    *, exponent: int, seed: int, t1: float, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    n = 2**exponent
    ts = jnp.linspace(0.0, t1, n + 1)
    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=t1,
        tol=t1 / (4 * n),
        shape=(dim,),
        key=jax.random.PRNGKey(seed),
    )
    ys = jax.vmap(lambda t: brownian.evaluate(t))(ts)
    return np.asarray(ts), np.asarray(ys)


def solve_fine_wong_zakai(ts: np.ndarray, xs: np.ndarray, y0: float) -> np.ndarray:
    ts_jax = jnp.asarray(ts)
    term = diffrax.ControlTerm(
        diffrax_vector_field,
        diffrax.LinearInterpolation(ts=ts_jax, ys=jnp.asarray(xs)),
    )
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Dopri5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=None,
        y0=jnp.asarray(y0),
        stepsize_controller=diffrax.StepTo(ts_jax),
        saveat=diffrax.SaveAt(ts=ts_jax),
        max_steps=len(ts) + 4,
    )
    return to_numpy_and_clear_cache(sol.ys)


def solve_so3_fine_wong_zakai(ts: np.ndarray, xs: np.ndarray) -> np.ndarray:
    ts_jax = jnp.asarray(ts)
    xs_jax = jnp.asarray(xs)
    velocities = (xs_jax[1:] - xs_jax[:-1]) / (ts_jax[1:] - ts_jax[:-1])[:, None]

    def coeffs(t, y, args):
        del y, args
        index = jnp.searchsorted(ts_jax, t, side="right") - 1
        index = jnp.clip(index, 0, velocities.shape[0] - 1)
        return velocities[index]

    sol = diffrax.diffeqsolve(
        GeometricTerm(coeffs, SO(3)),
        CFEES25(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=None,
        y0=jnp.eye(3),
        stepsize_controller=diffrax.StepTo(ts_jax),
        saveat=diffrax.SaveAt(ts=ts_jax),
        max_steps=len(ts) + 4,
    )
    return to_numpy_and_clear_cache(sol.ys)


def solve_log_ode(
    ts: np.ndarray,
    xs: np.ndarray,
    *,
    depth: int,
    coarse_exponent: int,
    fine_exponent: int,
    y0: float,
    solution: str,
) -> np.ndarray:
    coarse_ts = jnp.asarray(ts[:: 2 ** (fine_exponent - coarse_exponent)])
    driver = diffrax.LinearInterpolation(ts=jnp.asarray(ts), ys=jnp.asarray(xs))
    term = RoughTerm(
        rough_vector_field,
        driver,
        Euclidean(),
        depth=depth,
        interval_ts=coarse_ts,
        solution=solution,
    )

    sol = diffrax.diffeqsolve(
        term,
        LogODE(diffrax.Tsit5()),
        t0=float(coarse_ts[0]),
        t1=float(coarse_ts[-1]),
        dt0=None,
        y0=jnp.asarray(y0),
        stepsize_controller=diffrax.StepTo(coarse_ts),
        saveat=diffrax.SaveAt(ts=coarse_ts),
        max_steps=len(coarse_ts) + 4,
    )
    return to_numpy_and_clear_cache(sol.ys)


def solve_so3_rde(
    ts: np.ndarray,
    xs: np.ndarray,
    *,
    depth: int,
    coarse_exponent: int,
    fine_exponent: int,
    solution: str,
) -> np.ndarray:
    coarse_ts = jnp.asarray(ts[:: 2 ** (fine_exponent - coarse_exponent)])
    driver = diffrax.LinearInterpolation(ts=jnp.asarray(ts), ys=jnp.asarray(xs))
    term = RoughTerm(
        so3_vector_field,
        driver,
        SO(3),
        depth=depth,
        interval_ts=coarse_ts,
        solution=solution,
    )

    sol = diffrax.diffeqsolve(
        term,
        LogODE(CFEES25()),
        t0=float(coarse_ts[0]),
        t1=float(coarse_ts[-1]),
        dt0=None,
        y0=jnp.eye(3),
        stepsize_controller=diffrax.StepTo(coarse_ts),
        saveat=diffrax.SaveAt(ts=coarse_ts),
        max_steps=len(coarse_ts) + 4,
    )
    return to_numpy_and_clear_cache(sol.ys)


def expected_line(h: np.ndarray, errors: np.ndarray, rate: float) -> np.ndarray:
    offset = np.mean(np.log(errors) - rate * np.log(h))
    return np.exp(offset) * h**rate


def plot_convergence(
    h: np.ndarray,
    rows: list[tuple[str, str, dict[int, np.ndarray]]],
    *,
    output: Path,
) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.4 * len(rows)), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    for row_axes, (title_prefix, ylabel, mean_errors_by_depth) in zip(
        axes, rows, strict=True
    ):
        for ax, depth in zip(row_axes, (1, 2, 3), strict=True):
            errors = mean_errors_by_depth[depth]
            rate = 0.5 * depth
            ax.loglog(h, errors, "x", color="red", markersize=7, label="sampled")
            ax.loglog(
                h,
                expected_line(h, errors, rate),
                color="black",
                linestyle="--",
                label=rf"$h^{{{rate:g}}}$",
            )
            ax.set_title(f"{title_prefix} order {depth}")
            ax.set_xlabel("step size")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
        row_axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--y0", type=float, default=1.0)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--fine-exponent", type=int, default=14)
    parser.add_argument(
        "--coarse-exponents",
        type=int,
        nargs="+",
        default=list(range(4, 11)),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "log_ode_convergence.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_paths < 1:
        raise ValueError("--num-paths must be at least 1.")
    if any(k >= args.fine_exponent for k in args.coarse_exponents):
        raise ValueError("Every coarse exponent must be smaller than --fine-exponent.")

    h = np.asarray([args.t1 / 2**k for k in args.coarse_exponents], dtype=np.float64)
    stratonovich_errors_by_depth = {
        depth: np.zeros(len(args.coarse_exponents), dtype=np.float64)
        for depth in (1, 2, 3)
    }
    ito_errors_by_depth = {
        depth: np.zeros(len(args.coarse_exponents), dtype=np.float64)
        for depth in (1, 2, 3)
    }
    for path_index in range(args.num_paths):
        path_seed = args.seed + path_index
        print(
            f"sampling Brownian path {path_index + 1}/{args.num_paths} "
            f"on {2**args.fine_exponent} fine steps"
        )
        ts, xs = brownian_samples(
            exponent=args.fine_exponent,
            seed=path_seed,
            t1=args.t1,
            dim=2,
        )

        print(f"solving fine Wong-Zakai reference for path {path_index + 1}")
        y_ref = solve_fine_wong_zakai(ts, xs, args.y0)
        print(f"path {path_index + 1} reference y({args.t1}) = {y_ref[-1]:.8e}")

        for depth in (1, 2, 3):
            for i, k in enumerate(args.coarse_exponents):
                print(
                    f"solving Log-ODE Stratonovich order {depth} on {2**k} steps "
                    f"for path {path_index + 1}"
                )
                y = solve_log_ode(
                    ts,
                    xs,
                    depth=depth,
                    coarse_exponent=k,
                    fine_exponent=args.fine_exponent,
                    y0=args.y0,
                    solution="stratonovich",
                )
                step = 2 ** (args.fine_exponent - k)
                stratonovich_errors_by_depth[depth][i] += float(
                    np.max(np.abs(y - y_ref[::step]))
                )

                print(
                    f"solving Log-ODE branched Ito order {depth} on {2**k} steps "
                    f"for path {path_index + 1}"
                )
                y = solve_log_ode(
                    ts,
                    xs,
                    depth=depth,
                    coarse_exponent=k,
                    fine_exponent=args.fine_exponent,
                    y0=args.y0,
                    solution="ito",
                )
                ito_errors_by_depth[depth][i] += float(
                    np.max(np.abs(y - y_ref[::step]))
                )

    mean_stratonovich_errors_by_depth = {
        depth: np.maximum(errors / args.num_paths, np.finfo(np.float64).tiny)
        for depth, errors in stratonovich_errors_by_depth.items()
    }
    mean_ito_errors_by_depth = {
        depth: np.maximum(errors / args.num_paths, np.finfo(np.float64).tiny)
        for depth, errors in ito_errors_by_depth.items()
    }

    so3_stratonovich_errors_by_depth = {
        depth: np.zeros(len(args.coarse_exponents), dtype=np.float64)
        for depth in (1, 2, 3)
    }
    so3_ito_errors_by_depth = {
        depth: np.zeros(len(args.coarse_exponents), dtype=np.float64)
        for depth in (1, 2, 3)
    }
    for path_index in range(args.num_paths):
        path_seed = args.seed + path_index
        print(
            f"sampling SO(3) Brownian path {path_index + 1}/{args.num_paths} "
            f"on {2**args.fine_exponent} fine steps"
        )
        ts, xs = brownian_samples(
            exponent=args.fine_exponent,
            seed=path_seed,
            t1=args.t1,
            dim=3,
        )

        print(f"solving fine SO(3) Wong-Zakai reference for path {path_index + 1}")
        y_ref = solve_so3_fine_wong_zakai(ts, xs)
        print(
            f"path {path_index + 1} reference det(y({args.t1})) = "
            f"{np.linalg.det(y_ref[-1]):.8e}"
        )

        for depth in (1, 2, 3):
            for i, k in enumerate(args.coarse_exponents):
                print(
                    f"solving SO(3) RDE Stratonovich order {depth} on {2**k} steps "
                    f"for path {path_index + 1}"
                )
                y = solve_so3_rde(
                    ts,
                    xs,
                    depth=depth,
                    coarse_exponent=k,
                    fine_exponent=args.fine_exponent,
                    solution="stratonovich",
                )
                step = 2 ** (args.fine_exponent - k)
                path_error = np.linalg.norm(y - y_ref[::step], axis=(-2, -1))
                so3_stratonovich_errors_by_depth[depth][i] += float(np.max(path_error))

                print(
                    f"solving SO(3) RDE planar branched Ito order {depth} "
                    f"on {2**k} steps for path {path_index + 1}"
                )
                y = solve_so3_rde(
                    ts,
                    xs,
                    depth=depth,
                    coarse_exponent=k,
                    fine_exponent=args.fine_exponent,
                    solution="ito",
                )
                path_error = np.linalg.norm(y - y_ref[::step], axis=(-2, -1))
                so3_ito_errors_by_depth[depth][i] += float(np.max(path_error))

    mean_so3_stratonovich_errors_by_depth = {
        depth: np.maximum(errors / args.num_paths, np.finfo(np.float64).tiny)
        for depth, errors in so3_stratonovich_errors_by_depth.items()
    }
    mean_so3_ito_errors_by_depth = {
        depth: np.maximum(errors / args.num_paths, np.finfo(np.float64).tiny)
        for depth, errors in so3_ito_errors_by_depth.items()
    }
    plot_convergence(
        h,
        [
            (
                "Log-ODE Stratonovich",
                "max absolute path error",
                mean_stratonovich_errors_by_depth,
            ),
            (
                "Log-ODE branched Ito",
                "max absolute path error",
                mean_ito_errors_by_depth,
            ),
            (
                "SO(3) RDE Stratonovich",
                "max Frobenius path error",
                mean_so3_stratonovich_errors_by_depth,
            ),
            (
                "SO(3) RDE planar branched Ito",
                "max Frobenius path error",
                mean_so3_ito_errors_by_depth,
            ),
        ],
        output=args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
