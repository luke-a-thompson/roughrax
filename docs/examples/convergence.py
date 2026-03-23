"""Convergence experiment mirroring the original rough EES script."""

from __future__ import annotations

import argparse
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
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
import numpy as np
from fbm import FBM

from roughrax import RoughRK
from ees import EES25, EES27

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)


def vector_field(t, y, args):
    del t, args
    return jnp.stack([jnp.cos(y), jnp.sin(y)], axis=-1)


def get_2d_fbm(n: int, hurst: float, length: float) -> np.ndarray:
    fbm = FBM(n=n, hurst=hurst, length=length, method="daviesharte")
    x1 = fbm.fbm()
    x2 = fbm.fbm()
    return np.stack([x1, x2], axis=-1)


def method_run(
    base_solver: diffrax.AbstractERK, x: np.ndarray, y0: float = 1.0
) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    term = diffrax.ControlTerm(
        vector_field=vector_field,
        control=diffrax.LinearInterpolation(
            ts=jnp.asarray(ts, dtype=jnp.float64),
            ys=jnp.asarray(x, dtype=jnp.float64),
        ),
    )
    solution = diffrax.diffeqsolve(
        term,
        RoughRK(base_solver),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=float(ts[1] - ts[0]),
        y0=jnp.asarray(y0, dtype=jnp.float64),
        saveat=diffrax.SaveAt(ts=jnp.asarray(ts, dtype=jnp.float64)),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=len(ts) + 4,
    )
    return np.asarray(solution.ys).flatten()


def get_error(y_exact: np.ndarray, y_vals: np.ndarray, length: float) -> float:
    t = np.linspace(0.0, length, len(y_vals), dtype=np.float64)
    t_exact = np.linspace(0.0, length, len(y_exact), dtype=np.float64)
    true_vals = np.interp(t, t_exact, y_exact)
    return float(np.max(np.abs(true_vals - y_vals)))


def plot(
    base_solver: diffrax.AbstractERK,
    name: str,
    hurst: float,
    length: float,
    rate,
    ax,
    *,
    num_paths: int = 10,
    backward: bool = False,
) -> None:
    n = int(length * 2**16)
    h = [2 ** (-i) for i in range(4, 14)]
    y = np.zeros(len(h), dtype=np.float64)

    for _ in range(num_paths):
        x = get_2d_fbm(n, hurst=hurst, length=length)
        y_exact = method_run(base_solver, x)

        error = []
        for h_ in h:
            step = int(n * h_)
            x_coarse = x[::step]
            y_vals = method_run(base_solver, x_coarse)
            if not backward:
                error.append(get_error(y_exact, y_vals, length))
            else:
                y_vals = method_run(base_solver, x_coarse[::-1], y0=float(y_vals[-1]))
                error.append(abs(float(y_exact[0]) - float(y_vals[-1])))

        y += np.log10(np.maximum(error, np.finfo(np.float64).tiny))

    y /= num_paths
    x = np.log10(np.asarray(h, dtype=np.float64))
    dx = np.array([x[0], x[-1]], dtype=np.float64)

    err_label = (
        r"$\log_{10}(\mathcal{E}(h))$"
        if not backward
        else r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"
    )
    slope = rate(hurst)
    intercept = float(np.mean(y) - slope * np.mean(x))
    print(
        f"{name} H={hurst:.1f} {'backward' if backward else 'forward'} intercept: {intercept:.6f}"
    )

    ax.scatter(
        x,
        y,
        marker=mkr.MarkerStyle("x", fillstyle="none"),
        color="crimson",
    )
    ax.plot(dx, slope * dx + intercept, color="mediumblue")
    ax.legend([err_label, f"{np.round(slope, 1)}$x + c$"])
    ax.set_xlabel(r"$\log_{10}(h)$")
    ax.set_ylabel(err_label)


def plot_grid(hurst: float, *, length: float, num_paths: int, output_dir: Path) -> Path:
    _, ax = plt.subplots(2, 2, figsize=(10 * (2 / 3), 10 * (2 / 3)))
    methods = [("EES(2,5)", EES25()), ("EES(2,7)", EES27())]
    titles = [
        r"$\mathcal{E}(h)$ for $\mathrm{EES}_\mathcal{R}(2,5)$",
        r"$\overleftarrow{\mathcal{E}}(h)$ for $\mathrm{EES}_\mathcal{R}(2,5)$",
        r"$\mathcal{E}(h)$ for $\mathrm{EES}_\mathcal{R}(2,7)$",
        r"$\overleftarrow{\mathcal{E}}(h)$ for $\mathrm{EES}_\mathcal{R}(2,7)$",
    ]
    rates = [
        lambda x: 2 * x - 0.5,
        lambda x: 6 * x - 1.0,
        lambda x: 2 * x - 0.5,
        lambda x: 8 * x - 1.0,
    ]

    for i, (name, solver) in enumerate(methods):
        for j in range(2):
            plot(
                solver,
                name,
                hurst,
                length,
                rates[i * 2 + j],
                ax[i][j],
                num_paths=num_paths,
                backward=bool(j),
            )
            ax[i][j].set_title(titles[i * 2 + j])

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rde_example_ees{int(hurst * 100)}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hurst", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--num-paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    for hurst in args.hurst:
        output_path = plot_grid(
            hurst,
            length=args.length,
            num_paths=args.num_paths,
            output_dir=args.output_dir,
        )
        print(f"H={hurst:.1f} plot: {output_path}")


if __name__ == "__main__":
    main()
