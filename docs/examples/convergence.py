"""HMSigRK3, Log-ODE, and SO(3) convergence against Wong-Zakai references."""

# ruff: noqa: E402

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
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from georax import CFEES25, Euclidean, GeometricTerm, SO

from roughrax import HMSigRK3, LogODE, RoughTerm, SignatureInterpolation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def to_numpy(x) -> np.ndarray:
    out = np.asarray(jax.block_until_ready(x))
    gc.collect()
    return out


def diffrax_vector_field(t, y, args):
    del t, args
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def rough_vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


def so3_vector_field(y):
    return jnp.eye(3, dtype=y.dtype)


def brownian_samples_batch(
    *, exponent: int, seed: int, num_paths: int, t1: float, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    n = 2**exponent
    ts = jnp.linspace(0.0, t1, n + 1)
    keys = jnp.stack([jax.random.PRNGKey(seed + i) for i in range(num_paths)])

    @jax.jit
    def sample_all(keys):
        def sample_one(key):
            brownian = diffrax.VirtualBrownianTree(
                t0=0.0,
                t1=t1,
                tol=t1 / (4 * n),
                shape=(dim,),
                key=key,
            )
            return jax.vmap(lambda t: brownian.evaluate(t))(ts)

        return jax.vmap(sample_one)(keys)

    return np.asarray(ts), to_numpy(sample_all(keys))


@jax.jit
def _solve_fine_wong_zakai_batch(
    ts: jax.Array, xs_batch: jax.Array, y0: jax.Array
) -> jax.Array:
    def solve_one(xs):
        term = diffrax.ControlTerm(
            diffrax_vector_field,
            diffrax.LinearInterpolation(ts=ts, ys=xs),
        )
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(ts),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=ts.shape[0] + 4,
        )
        return sol.ys

    return jax.vmap(solve_one)(xs_batch)


def solve_fine_wong_zakai_batch(
    ts: np.ndarray, xs_batch: np.ndarray, y0: float
) -> np.ndarray:
    return to_numpy(
        _solve_fine_wong_zakai_batch(
            jnp.asarray(ts), jnp.asarray(xs_batch), jnp.asarray(y0)
        )
    )


@jax.jit
def _solve_so3_fine_wong_zakai_batch(ts: jax.Array, xs_batch: jax.Array) -> jax.Array:
    def solve_one(xs):
        velocities = (xs[1:] - xs[:-1]) / (ts[1:] - ts[:-1])[:, None]

        def coeffs(t, y, args):
            del y, args
            index = jnp.searchsorted(ts, t, side="right") - 1
            index = jnp.clip(index, 0, velocities.shape[0] - 1)
            return velocities[index]

        sol = diffrax.diffeqsolve(
            GeometricTerm(coeffs, SO(3)),
            CFEES25(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=jnp.eye(3, dtype=xs.dtype),
            stepsize_controller=diffrax.StepTo(ts),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=ts.shape[0] + 4,
        )
        return sol.ys

    return jax.vmap(solve_one)(xs_batch)


def solve_so3_fine_wong_zakai_batch(ts: np.ndarray, xs_batch: np.ndarray) -> np.ndarray:
    return to_numpy(
        _solve_so3_fine_wong_zakai_batch(jnp.asarray(ts), jnp.asarray(xs_batch))
    )


def rough_term_template_and_coeffs(
    ts: np.ndarray,
    xs_batch: np.ndarray,
    *,
    vector_field,
    geometry,
    depth: int,
    coarse_exponent: int,
    fine_exponent: int,
    solution: str,
) -> tuple[jax.Array, RoughTerm, jax.Array]:
    step = 2 ** (fine_exponent - coarse_exponent)
    ts_jax = jnp.asarray(ts)
    coarse_ts = jnp.asarray(ts[::step])
    template = None
    coeffs = []

    for xs in xs_batch:
        driver = diffrax.LinearInterpolation(ts=ts_jax, ys=jnp.asarray(xs))
        control = SignatureInterpolation(
            driver,
            coarse_ts,
            depth,
            solution,
        )
        term = RoughTerm(
            vector_field,
            control,
            geometry,
        )
        if template is None:
            template = term
        coeffs.append(np.asarray(term.control.coeffs))

    if template is None:
        raise ValueError("At least one path is required.")
    return coarse_ts, template, jnp.asarray(np.stack(coeffs))


@eqx.filter_jit
def solve_rough_from_coeffs_batch(
    term_template: RoughTerm,
    coeffs_batch: jax.Array,
    coarse_ts: jax.Array,
    y0: jax.Array,
    solver,
) -> jax.Array:
    def solve_one(coeffs):
        term = eqx.tree_at(lambda term: term.control.coeffs, term_template, coeffs)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=coarse_ts[0],
            t1=coarse_ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(coarse_ts),
            saveat=diffrax.SaveAt(ts=coarse_ts),
            max_steps=coarse_ts.shape[0] + 4,
        )
        return sol.ys

    return jax.vmap(solve_one)(coeffs_batch)


def solve_log_ode_batch(
    ts: np.ndarray,
    xs_batch: np.ndarray,
    *,
    depth: int,
    coarse_exponent: int,
    fine_exponent: int,
    y0: float,
    solution: str,
) -> np.ndarray:
    coarse_ts, template, coeffs_batch = rough_term_template_and_coeffs(
        ts,
        xs_batch,
        vector_field=rough_vector_field,
        geometry=Euclidean(),
        depth=depth,
        coarse_exponent=coarse_exponent,
        fine_exponent=fine_exponent,
        solution=solution,
    )
    return to_numpy(
        solve_rough_from_coeffs_batch(
            template,
            coeffs_batch,
            coarse_ts,
            jnp.asarray(y0),
            LogODE(diffrax.Tsit5()),
        )
    )


def solve_hm_sigrk3_batch(
    ts: np.ndarray,
    xs_batch: np.ndarray,
    *,
    coarse_exponent: int,
    fine_exponent: int,
    y0: float,
    rho_scale: float,
    beta_scale: float,
    brownian_chain_correction: bool,
) -> np.ndarray:
    coarse_ts, template, coeffs_batch = rough_term_template_and_coeffs(
        ts,
        xs_batch,
        vector_field=rough_vector_field,
        geometry=Euclidean(),
        depth=3,
        coarse_exponent=coarse_exponent,
        fine_exponent=fine_exponent,
        solution="stratonovich",
    )
    return to_numpy(
        solve_rough_from_coeffs_batch(
            template,
            coeffs_batch,
            coarse_ts,
            jnp.asarray(y0),
            HMSigRK3(
                rho_scale=rho_scale,
                beta_scale=beta_scale,
                brownian_chain_correction=brownian_chain_correction,
            ),
        )
    )


def solve_so3_rde_batch(
    ts: np.ndarray,
    xs_batch: np.ndarray,
    *,
    depth: int,
    coarse_exponent: int,
    fine_exponent: int,
    solution: str,
) -> np.ndarray:
    coarse_ts, template, coeffs_batch = rough_term_template_and_coeffs(
        ts,
        xs_batch,
        vector_field=so3_vector_field,
        geometry=SO(3),
        depth=depth,
        coarse_exponent=coarse_exponent,
        fine_exponent=fine_exponent,
        solution=solution,
    )
    return to_numpy(
        solve_rough_from_coeffs_batch(
            template,
            coeffs_batch,
            coarse_ts,
            jnp.eye(3),
            LogODE(CFEES25()),
        )
    )


def expected_line(h: np.ndarray, errors: np.ndarray, rate: float) -> np.ndarray:
    offset = np.mean(np.log(errors) - rate * np.log(h))
    return np.exp(offset) * h**rate


def plot_convergence(
    h: np.ndarray,
    rows: list[
        tuple[
            str,
            str,
            dict[int, np.ndarray],
            dict[int, tuple[str, np.ndarray]],
        ]
    ],
    *,
    output: Path,
) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.4 * len(rows)), sharex=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    for row_axes, (
        title_prefix,
        ylabel,
        mean_errors_by_depth,
        comparisons_by_depth,
    ) in zip(axes, rows, strict=True):
        for ax, depth in zip(row_axes, (1, 2, 3), strict=True):
            errors = mean_errors_by_depth[depth]
            rate = 0.5 * depth
            ax.loglog(h, errors, "x", color="red", markersize=7, label="Log-ODE")
            ax.loglog(
                h,
                expected_line(h, errors, rate),
                color="black",
                linestyle="--",
                label=rf"$h^{{{rate:g}}}$",
            )
            if depth in comparisons_by_depth:
                label, comparison_errors = comparisons_by_depth[depth]
                ax.loglog(
                    h,
                    comparison_errors,
                    "+",
                    color="tab:blue",
                    markersize=7,
                    label=label,
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


def plot_hm_sigrk3_comparison(
    h: np.ndarray,
    *,
    log_ode_order_two_path_errors: np.ndarray,
    log_ode_path_errors: np.ndarray,
    hm_sigrk3_path_errors: np.ndarray,
    log_ode_order_two_endpoint_errors: np.ndarray,
    log_ode_endpoint_errors: np.ndarray,
    hm_sigrk3_endpoint_errors: np.ndarray,
    rho_scale: float,
    beta_scale: float,
    brownian_chain_correction: bool,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    metrics = (
        (
            log_ode_order_two_path_errors,
            log_ode_path_errors,
            hm_sigrk3_path_errors,
            "mean max absolute path error",
        ),
        (
            log_ode_order_two_endpoint_errors,
            log_ode_endpoint_errors,
            hm_sigrk3_endpoint_errors,
            "mean absolute endpoint error",
        ),
    )
    for ax, (
        log_ode_order_two_errors,
        log_ode_errors,
        hm_sigrk3_errors,
        ylabel,
    ) in zip(axes, metrics, strict=True):
        ax.loglog(
            h,
            log_ode_order_two_errors,
            "o-",
            color="tab:orange",
            label="Log-ODE order 2",
        )
        ax.loglog(
            h,
            hm_sigrk3_errors,
            "+-",
            color="tab:blue",
            label="HMSigRK3",
        )
        ax.loglog(
            h,
            expected_line(h, log_ode_order_two_errors, 1.0),
            color="gray",
            linestyle=":",
            label=r"$h^1$",
        )
        ax.loglog(
            h,
            expected_line(h, hm_sigrk3_errors, 1.5),
            color="black",
            linestyle="--",
            label=r"$h^{1.5}$",
        )
        ax.set_xlabel("step size")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Third-order Stratonovich RDE convergence "
        f"(HMSigRK3 rho={rho_scale:.4g}, beta={beta_scale:.4g}, "
        f"Brownian correction={brownian_chain_correction})"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def fitted_convergence_rate(h: np.ndarray, errors: np.ndarray) -> float:
    if h.size < 2:
        return float("nan")
    return float(np.polyfit(np.log(h), np.log(errors), deg=1)[0])


def print_hm_sigrk3_comparison(
    h: np.ndarray,
    *,
    log_ode_order_two_path_errors: np.ndarray,
    log_ode_path_errors: np.ndarray,
    hm_sigrk3_path_errors: np.ndarray,
    log_ode_order_two_endpoint_errors: np.ndarray,
    log_ode_endpoint_errors: np.ndarray,
    hm_sigrk3_endpoint_errors: np.ndarray,
    rho_scale: float,
    beta_scale: float,
    brownian_chain_correction: bool,
) -> None:
    print(
        "\nHMSigRK3 versus Log-ODE order 3 "
        f"(rho_scale={rho_scale:.8g}, beta_scale={beta_scale:.8g}, "
        f"brownian_chain_correction={brownian_chain_correction})"
    )
    print("\npath errors")
    print("step size    LogODE-2    LogODE-3     HMSigRK3  HM/LogODE-2  HM/LogODE-3")
    for values in zip(
        h,
        log_ode_order_two_path_errors,
        log_ode_path_errors,
        hm_sigrk3_path_errors,
        strict=True,
    ):
        (
            step_size,
            log_ode_order_two_path,
            log_ode_path,
            hm_sigrk3_path,
        ) = values
        print(
            f"{step_size:9.3e}  "
            f"{log_ode_order_two_path:11.4e}  "
            f"{log_ode_path:11.4e}  "
            f"{hm_sigrk3_path:11.4e}  "
            f"{hm_sigrk3_path / log_ode_order_two_path:11.2f}  "
            f"{hm_sigrk3_path / log_ode_path:10.2f}"
        )

    print("\nendpoint errors")
    print("step size    LogODE-2    LogODE-3     HMSigRK3  HM/LogODE-2  HM/LogODE-3")
    for values in zip(
        h,
        log_ode_order_two_endpoint_errors,
        log_ode_endpoint_errors,
        hm_sigrk3_endpoint_errors,
        strict=True,
    ):
        (
            step_size,
            log_ode_order_two_endpoint,
            log_ode_endpoint,
            hm_sigrk3_endpoint,
        ) = values
        print(
            f"{step_size:9.3e}  "
            f"{log_ode_order_two_endpoint:11.4e}  "
            f"{log_ode_endpoint:11.4e}  "
            f"{hm_sigrk3_endpoint:11.4e}  "
            f"{hm_sigrk3_endpoint / log_ode_order_two_endpoint:11.2f}  "
            f"{hm_sigrk3_endpoint / log_ode_endpoint:10.2f}"
        )

    finest = slice(-min(3, h.size), None)
    print("\nfitted convergence rates (expected: Log-ODE-2 1.0; order 3 1.5)")
    print("method       metric       all points  finest 3")
    for method, metric, errors in (
        ("Log-ODE-2", "path", log_ode_order_two_path_errors),
        ("Log-ODE-3", "path", log_ode_path_errors),
        ("HMSigRK3", "path", hm_sigrk3_path_errors),
        ("Log-ODE-2", "endpoint", log_ode_order_two_endpoint_errors),
        ("Log-ODE-3", "endpoint", log_ode_endpoint_errors),
        ("HMSigRK3", "endpoint", hm_sigrk3_endpoint_errors),
    ):
        print(
            f"{method:10s}   {metric:8s}   "
            f"{fitted_convergence_rate(h, errors):10.3f}  "
            f"{fitted_convergence_rate(h[finest], errors[finest]):8.3f}"
        )
    print()


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
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "hm_sigrk3_vs_log_ode_convergence.png",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only run the Euclidean HMSigRK3 versus Log-ODE comparison.",
    )
    parser.add_argument("--hm-rho-scale", type=float, default=1.0)
    parser.add_argument("--hm-beta-scale", type=float, default=1.0)
    parser.add_argument(
        "--hm-brownian-chain-correction",
        action="store_true",
        help="Enable HMSigRK3's Brownian-specific fourth-degree chain correction.",
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
    hm_sigrk3_errors = np.zeros(len(args.coarse_exponents), dtype=np.float64)
    log_ode_order_two_endpoint_errors = np.zeros(
        len(args.coarse_exponents), dtype=np.float64
    )
    log_ode_order_three_endpoint_errors = np.zeros(
        len(args.coarse_exponents), dtype=np.float64
    )
    hm_sigrk3_endpoint_errors = np.zeros(len(args.coarse_exponents), dtype=np.float64)
    print(
        f"sampling {args.num_paths} Brownian paths on "
        f"{2**args.fine_exponent} fine steps"
    )
    ts, xs_batch = brownian_samples_batch(
        exponent=args.fine_exponent,
        seed=args.seed,
        num_paths=args.num_paths,
        t1=args.t1,
        dim=2,
    )

    print(f"solving {args.num_paths} fine Wong-Zakai references")
    y_refs = solve_fine_wong_zakai_batch(ts, xs_batch, args.y0)
    print(
        f"reference y({args.t1}) mean = {np.mean(y_refs[:, -1]):.8e}, "
        f"std = {np.std(y_refs[:, -1]):.8e}"
    )

    depths = (2, 3) if args.comparison_only else (1, 2, 3)
    for depth in depths:
        for i, k in enumerate(args.coarse_exponents):
            print(
                f"solving Log-ODE Stratonovich order {depth} on {2**k} steps "
                f"for {args.num_paths} paths"
            )
            y = solve_log_ode_batch(
                ts,
                xs_batch,
                depth=depth,
                coarse_exponent=k,
                fine_exponent=args.fine_exponent,
                y0=args.y0,
                solution="stratonovich",
            )
            step = 2 ** (args.fine_exponent - k)
            path_errors = np.max(np.abs(y - y_refs[:, ::step]), axis=1)
            stratonovich_errors_by_depth[depth][i] = float(np.mean(path_errors))
            if depth in (2, 3):
                endpoint_errors = np.abs(y[:, -1] - y_refs[:, -1])
                target = (
                    log_ode_order_two_endpoint_errors
                    if depth == 2
                    else log_ode_order_three_endpoint_errors
                )
                target[i] = float(np.mean(endpoint_errors))

            if args.comparison_only:
                continue
            print(
                f"solving Log-ODE branched Ito order {depth} on {2**k} steps "
                f"for {args.num_paths} paths"
            )
            y = solve_log_ode_batch(
                ts,
                xs_batch,
                depth=depth,
                coarse_exponent=k,
                fine_exponent=args.fine_exponent,
                y0=args.y0,
                solution="ito",
            )
            path_errors = np.max(np.abs(y - y_refs[:, ::step]), axis=1)
            ito_errors_by_depth[depth][i] = float(np.mean(path_errors))

    for i, k in enumerate(args.coarse_exponents):
        print(f"solving HMSigRK3 on {2**k} steps for {args.num_paths} paths")
        y = solve_hm_sigrk3_batch(
            ts,
            xs_batch,
            coarse_exponent=k,
            fine_exponent=args.fine_exponent,
            y0=args.y0,
            rho_scale=args.hm_rho_scale,
            beta_scale=args.hm_beta_scale,
            brownian_chain_correction=args.hm_brownian_chain_correction,
        )
        step = 2 ** (args.fine_exponent - k)
        path_errors = np.max(np.abs(y - y_refs[:, ::step]), axis=1)
        hm_sigrk3_errors[i] = float(np.mean(path_errors))
        endpoint_errors = np.abs(y[:, -1] - y_refs[:, -1])
        hm_sigrk3_endpoint_errors[i] = float(np.mean(endpoint_errors))

    mean_stratonovich_errors_by_depth = {
        depth: np.maximum(errors, np.finfo(np.float64).tiny)
        for depth, errors in stratonovich_errors_by_depth.items()
    }
    mean_ito_errors_by_depth = {
        depth: np.maximum(errors, np.finfo(np.float64).tiny)
        for depth, errors in ito_errors_by_depth.items()
    }
    mean_hm_sigrk3_errors = np.maximum(hm_sigrk3_errors, np.finfo(np.float64).tiny)
    mean_log_ode_order_three_endpoint_errors = np.maximum(
        log_ode_order_three_endpoint_errors,
        np.finfo(np.float64).tiny,
    )
    mean_log_ode_order_two_endpoint_errors = np.maximum(
        log_ode_order_two_endpoint_errors,
        np.finfo(np.float64).tiny,
    )
    mean_hm_sigrk3_endpoint_errors = np.maximum(
        hm_sigrk3_endpoint_errors,
        np.finfo(np.float64).tiny,
    )
    print_hm_sigrk3_comparison(
        h,
        log_ode_order_two_path_errors=mean_stratonovich_errors_by_depth[2],
        log_ode_path_errors=mean_stratonovich_errors_by_depth[3],
        hm_sigrk3_path_errors=mean_hm_sigrk3_errors,
        log_ode_order_two_endpoint_errors=mean_log_ode_order_two_endpoint_errors,
        log_ode_endpoint_errors=mean_log_ode_order_three_endpoint_errors,
        hm_sigrk3_endpoint_errors=mean_hm_sigrk3_endpoint_errors,
        rho_scale=args.hm_rho_scale,
        beta_scale=args.hm_beta_scale,
        brownian_chain_correction=args.hm_brownian_chain_correction,
    )
    plot_hm_sigrk3_comparison(
        h,
        log_ode_order_two_path_errors=mean_stratonovich_errors_by_depth[2],
        log_ode_path_errors=mean_stratonovich_errors_by_depth[3],
        hm_sigrk3_path_errors=mean_hm_sigrk3_errors,
        log_ode_order_two_endpoint_errors=mean_log_ode_order_two_endpoint_errors,
        log_ode_endpoint_errors=mean_log_ode_order_three_endpoint_errors,
        hm_sigrk3_endpoint_errors=mean_hm_sigrk3_endpoint_errors,
        rho_scale=args.hm_rho_scale,
        beta_scale=args.hm_beta_scale,
        brownian_chain_correction=args.hm_brownian_chain_correction,
        output=args.comparison_output,
    )
    print(f"saved {args.comparison_output}")
    stages = HMSigRK3.num_stages(
        2,
        brownian_chain_correction=args.hm_brownian_chain_correction,
    )
    print(f"HMSigRK3 stages for d=2: {stages}")
    if args.comparison_only:
        return

    so3_stratonovich_errors_by_depth = {
        depth: np.zeros(len(args.coarse_exponents), dtype=np.float64)
        for depth in (1, 2, 3)
    }
    so3_ito_errors_by_depth = {
        depth: np.zeros(len(args.coarse_exponents), dtype=np.float64)
        for depth in (1, 2, 3)
    }
    print(
        f"sampling {args.num_paths} SO(3) Brownian paths on "
        f"{2**args.fine_exponent} fine steps"
    )
    ts, xs_batch = brownian_samples_batch(
        exponent=args.fine_exponent,
        seed=args.seed,
        num_paths=args.num_paths,
        t1=args.t1,
        dim=3,
    )

    print(f"solving {args.num_paths} fine SO(3) Wong-Zakai references")
    y_refs = solve_so3_fine_wong_zakai_batch(ts, xs_batch)
    dets = np.linalg.det(y_refs[:, -1])
    print(
        f"reference det(y({args.t1})) mean = {np.mean(dets):.8e}, "
        f"std = {np.std(dets):.8e}"
    )

    for depth in (1, 2, 3):
        for i, k in enumerate(args.coarse_exponents):
            print(
                f"solving SO(3) RDE Stratonovich order {depth} on {2**k} steps "
                f"for {args.num_paths} paths"
            )
            y = solve_so3_rde_batch(
                ts,
                xs_batch,
                depth=depth,
                coarse_exponent=k,
                fine_exponent=args.fine_exponent,
                solution="stratonovich",
            )
            step = 2 ** (args.fine_exponent - k)
            path_errors = np.linalg.norm(y - y_refs[:, ::step], axis=(-2, -1))
            so3_stratonovich_errors_by_depth[depth][i] = float(
                np.mean(np.max(path_errors, axis=1))
            )

            print(
                f"solving SO(3) RDE planar branched Ito order {depth} "
                f"on {2**k} steps for {args.num_paths} paths"
            )
            y = solve_so3_rde_batch(
                ts,
                xs_batch,
                depth=depth,
                coarse_exponent=k,
                fine_exponent=args.fine_exponent,
                solution="ito",
            )
            path_errors = np.linalg.norm(y - y_refs[:, ::step], axis=(-2, -1))
            so3_ito_errors_by_depth[depth][i] = float(
                np.mean(np.max(path_errors, axis=1))
            )

    mean_so3_stratonovich_errors_by_depth = {
        depth: np.maximum(errors, np.finfo(np.float64).tiny)
        for depth, errors in so3_stratonovich_errors_by_depth.items()
    }
    mean_so3_ito_errors_by_depth = {
        depth: np.maximum(errors, np.finfo(np.float64).tiny)
        for depth, errors in so3_ito_errors_by_depth.items()
    }
    plot_convergence(
        h,
        [
            (
                "Log-ODE Stratonovich",
                "max absolute path error",
                mean_stratonovich_errors_by_depth,
                {3: ("HMSigRK3", mean_hm_sigrk3_errors)},
            ),
            (
                "Log-ODE branched Ito",
                "max absolute path error",
                mean_ito_errors_by_depth,
                {},
            ),
            (
                "SO(3) RDE Stratonovich",
                "max Frobenius path error",
                mean_so3_stratonovich_errors_by_depth,
                {},
            ),
            (
                "SO(3) RDE planar branched Ito",
                "max Frobenius path error",
                mean_so3_ito_errors_by_depth,
                {},
            ),
        ],
        output=args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
