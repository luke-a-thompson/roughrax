"""The geometric controlled RDE dY = X dX, Y_0 = 0.

Run: python docs/examples/controlled_brownian.py
Writes a path comparison and mesh-error figure to docs/examples/outputs/.

The exact solution is (X_t**2 - X_0**2) / 2. Freezing X at each step loses
half the discrete quadratic variation, which tends to T/2 for Brownian X.
Supplying V' = 1 restores the geometric correction. Heun integrates the
augmented local ODE exactly here, so every mesh is exact up to roundoff.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/roughrax-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from roughrax import ControlledVectorField, LogODE, RoughTerm, SignatureInterpolation

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@eqx.filter_jit
def integrate(ts, xs, *, controlled):
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs[:, None])

    def vf(t, y, args):
        return driver.evaluate(t)  # Shape (driver_dim,) for scalar y.

    def vf_prime(t, y, args):
        return jnp.ones((1, 1), dtype=xs.dtype)

    control = SignatureInterpolation(
        driver, signature_knots=ts, depth=2, solution="stratonovich"
    )
    coefficient = ControlledVectorField(vf, vf_prime if controlled else None)
    # Equivalently, infer the derivatives of F(x, y, args) = x:
    # coefficient = ControlledVectorField.from_driver(
    #     lambda x, y, args: x, driver, depth=2
    # )
    solution = diffrax.diffeqsolve(
        RoughTerm(coefficient, control),
        LogODE(diffrax.Heun()),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=jnp.asarray(0.0, dtype=xs.dtype),
        stepsize_controller=diffrax.StepTo(ts),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=len(ts),
    )
    return solution.ys


def plot_results(ts, xs, frozen, corrected, results, *, seed, output_dir):
    """Plot one shared Brownian path and errors on its nested meshes."""
    exact = 0.5 * (xs**2 - xs[0] ** 2)
    ink, teal, orange, purple = "#253746", "#008779", "#cf623b", "#8573a6"
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": ink,
        "text.color": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.edgecolor": "#bec6c9",
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), layout="constrained")
        fig.set_facecolor("#fafbf9")
        fig.suptitle(
            "A rough coefficient changes the answer\n"
            r"$dY_t=X_t\,d\mathbf{X}_t$, geometric lift, $Y_0=0$",
            fontsize=18,
        )
        for ax in axes.flat:
            ax.set_facecolor("#fafbf9")
            ax.grid(alpha=0.35, color="#d5dddf", linewidth=0.7)
            ax.set_axisbelow(True)

        ax = axes[0, 0]
        ax.plot(ts, xs, color=ink, linewidth=1)
        ax.axhline(0, color="#bec6c9", linewidth=0.8)
        ax.set(title=f"One Brownian path · seed {seed}", xlabel="Time", ylabel=r"$X_t$")

        ax = axes[0, 1]
        ax.plot(ts, exact, color=ink, linewidth=1.4, label=r"Exact: $(X_t^2-X_0^2)/2$")
        ax.plot(
            ts,
            corrected,
            color=teal,
            linewidth=1,
            linestyle="--",
            label="Controlled log-ODE",
        )
        # Sparse markers make the coincident controlled and exact paths visible.
        indices = np.linspace(0, len(ts) - 1, 25, dtype=int)
        ax.plot(
            ts[indices],
            corrected[indices],
            "o",
            color=teal,
            markersize=3,
            markerfacecolor="#fafbf9",
            markeredgewidth=0.9,
        )
        ax.plot(ts, frozen, color=orange, linewidth=1, label="Frozen coefficient")
        ax.set(title="Solution paths · finest mesh", xlabel="Time", ylabel=r"$Y_t$")
        ax.legend(loc="upper left", fontsize=9)

        ax = axes[1, 0]
        ax.fill_between(ts, frozen - exact, 0, color=orange, alpha=0.08)
        ax.plot(
            ts, corrected - exact, color=teal, linewidth=1.8, label="Controlled error"
        )
        ax.plot(ts, frozen - exact, color=orange, linewidth=1.5, label="Frozen error")
        ax.plot(
            ts,
            -(ts - ts[0]) / 2,
            color=purple,
            linestyle="--",
            label=r"Brownian limit: $-t/2$",
        )
        ax.set(
            title="The missing quadratic-variation correction",
            xlabel="Time",
            ylabel="Numerical − exact",
        )
        ax.legend(loc="lower left", fontsize=9)

        ax = axes[1, 1]
        steps, frozen_errors, controlled_errors = np.asarray(results).T
        floor = np.finfo(float).eps / 10
        ax.loglog(steps, frozen_errors, "o-", color=orange, label="Frozen coefficient")
        ax.loglog(
            steps,
            np.maximum(controlled_errors, floor),
            "o-",
            color=teal,
            label="Controlled log-ODE",
        )
        ax.axhline(
            (ts[-1] - ts[0]) / 2,
            color=purple,
            linestyle="--",
            linewidth=1,
            label=r"Brownian limit: $T/2$",
        )
        ax.set(
            title="Mesh refinement · same Brownian path",
            xlabel="Number of steps",
            ylabel="Maximum absolute error at mesh points",
            ylim=(1e-17, 2),
        )
        ax.set_xticks(steps, labels=[str(int(step)) for step in steps])
        ax.minorticks_off()
        ax.legend(loc="center right", fontsize=9)
        fig.supxlabel(
            "Heun inner solver · nested meshes · controlled solution equals the exact solution up to roundoff",
            fontsize=10,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            path = output_dir / f"controlled_brownian.{suffix}"
            fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
            print(f"Saved {path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=12,
        help="Finest mesh has 2**max_power steps (minimum 4).",
    )
    options = parser.parse_args()
    if options.max_power < 4:
        parser.error("--max-power must be at least 4")

    jax.config.update("jax_enable_x64", True)
    n = 2**options.max_power
    t1 = 1.0
    rng = np.random.default_rng(options.seed)
    xs = np.concatenate([[0.0], np.cumsum(rng.normal(size=n) * np.sqrt(t1 / n))])
    ts = np.linspace(0.0, t1, n + 1)
    exact_terminal = 0.5 * (xs[-1] ** 2 - xs[0] ** 2)
    print(
        f"Exact Y_T: {exact_terminal:.8f}; frozen-coefficient bias tends to {-t1 / 2:.3f}"
    )
    print(" steps   frozen error    -quadratic variation/2   controlled max error")

    powers = sorted(set(range(4, options.max_power + 1, 2)) | {options.max_power})
    results = []
    for power in powers:
        stride = 2 ** (options.max_power - power)
        mesh_ts, mesh_xs = jnp.asarray(ts[::stride]), jnp.asarray(xs[::stride])
        frozen = np.asarray(integrate(mesh_ts, mesh_xs, controlled=False))
        corrected = np.asarray(integrate(mesh_ts, mesh_xs, controlled=True))
        exact = 0.5 * (np.asarray(mesh_xs) ** 2 - xs[0] ** 2)
        bias = -0.5 * np.sum(np.diff(np.asarray(mesh_xs)) ** 2)
        error = np.max(np.abs(corrected - exact))
        results.append((2**power, np.max(np.abs(frozen - exact)), error))
        np.testing.assert_allclose(corrected, exact, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(
            frozen[-1] - exact_terminal, bias, atol=1e-11, rtol=1e-11
        )
        print(
            f"{2**power:6d}   {frozen[-1] - exact_terminal: .8f}       {bias: .8f}                  {error:.2e}"
        )
    plot_results(
        ts,
        xs,
        frozen,
        corrected,
        results,
        seed=options.seed,
        output_dir=options.output_dir,
    )


if __name__ == "__main__":
    main()
