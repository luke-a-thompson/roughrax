"""Animate a nonlinear controlled RDE driven by two Brownian channels.

Run: python docs/examples/controlled_nonlinear.py

Y is two-dimensional and dY = F(X, Y) dX (geometric / Stratonovich), where

    F_1(x, y) = 0.7 * (1 + 0.4 sin(x_1 + y_2), 0.45 cos(x_2 - y_1))
    F_2(x, y) = 0.7 * (0.45 sin(x_1 - y_2), 1 + 0.4 sin(x_2 + y_1)).

Rows are driving fields. These bounded, smooth fields depend nonlinearly on
both X and Y and do not commute. The controlled coefficient derivatives are
generated with respect to X; spatial derivatives are handled by the lift.

The frozen comparison deliberately sets all rough coefficient derivatives to
zero, but retains the spatial derivatives and the same driver signatures.
Both use LogODE(Dopri5()). A separate reference integrates the ordinary ODE
along every linear segment of the sampled Brownian path, evaluating F at the
evolving X and Y. Splitting each segment again checks reference accuracy.
This is a numerical Wong--Zakai reference, not a claimed exact Brownian solution.

Two nested comparison meshes are checked. Only paths on the finest comparison
mesh are animated, synchronised by physical time. Outputs are a GIF and PNG in
docs/examples/outputs/.
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
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def vector_field(x, y, args):
    del args
    return 0.7 * jnp.array(
        [
            [1 + 0.4 * jnp.sin(x[0] + y[1]), 0.45 * jnp.cos(x[1] - y[0])],
            [0.45 * jnp.sin(x[0] - y[1]), 1 + 0.4 * jnp.sin(x[1] + y[0])],
        ]
    )


@eqx.filter_jit
def solve_log_ode(ts, xs, knots, *, depth, controlled):
    driver = diffrax.LinearInterpolation(ts=ts, ys=xs)
    control = SignatureInterpolation(
        driver, signature_knots=knots, depth=depth, solution="stratonovich"
    )
    if controlled:
        coefficient = ControlledVectorField.from_driver(
            vector_field, driver, depth=depth
        )
    else:
        coefficient = ControlledVectorField(
            lambda t, y, args: vector_field(driver.evaluate(t), y, args),
            *((None,) * (depth - 1)),
        )
    return diffrax.diffeqsolve(
        RoughTerm(coefficient, control),
        LogODE(diffrax.Dopri5()),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=jnp.zeros(2, dtype=xs.dtype),
        stepsize_controller=diffrax.StepTo(knots),
        saveat=diffrax.SaveAt(ts=knots),
        max_steps=len(knots),
    ).ys


@eqx.filter_jit
def solve_reference(xs, *, substeps):
    """Integrate each linear driver segment directly, without roughrax's lift."""
    solver = diffrax.Dopri5()

    def advance(y, segment):
        start, increment = segment

        def rhs(s, current_y, args):
            return vector_field(start + s * increment, current_y, args).T @ increment

        term = diffrax.ODETerm(rhs)
        state = solver.init(term, 0.0, 1.0, y, None)
        for k in range(substeps):
            y, _, _, state, _ = solver.step(
                term, k / substeps, (k + 1) / substeps, y, None, state, False
            )
        return y, y

    y0 = jnp.zeros(2, dtype=xs.dtype)
    _, ys = jax.lax.scan(advance, y0, (xs[:-1], jnp.diff(xs, axis=0)))
    return jnp.concatenate([y0[None], ys])


def max_path_error(path, reference):
    return float(np.linalg.norm(path - reference, axis=-1).max())


def animate_paths(
    ts, frozen, controlled, reference, *, depth, seed, fps, seconds, output_dir
):
    background, ink = "#fafbf9", "#253746"
    colors = ("#cf623b", "#008779")
    paths = (frozen, controlled)
    titles = ("Frozen rough dependence", "Controlled log-ODE")
    all_points = np.concatenate([reference, *paths])
    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    radius = max(float(np.ptp(all_points, axis=0).max()) * 0.60, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.1))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.78, wspace=0.22)
    fig.set_facecolor(background)
    fig.suptitle(
        "Nonlinear rough coefficients · two Brownian channels",
        fontsize=17,
        y=0.97,
        color=ink,
    )
    fig.text(
        0.5,
        0.90,
        r"$dY_t=F(X_t,Y_t)\,d\mathbf{X}_t$  ·  same driver, same mesh, same inner solver",
        ha="center",
        color=ink,
        fontsize=11,
    )
    clock = fig.text(0.5, 0.84, "", ha="center", fontsize=11, color=ink)
    fig.text(
        0.5,
        0.025,
        f"{len(ts) - 1:,} comparison steps · depth {depth} · seed {seed} · grey: fine piecewise-linear reference",
        ha="center",
        fontsize=10,
        color=ink,
    )
    artists = []

    for ax, path, title, color in zip(axes, paths, titles, colors, strict=True):
        ax.set_facecolor(background)
        ax.set_aspect("equal", adjustable="box")
        ax.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            xlabel=r"$Y^1$",
            ylabel=r"$Y^2$",
        )
        ax.set_title(title, color=color, fontsize=14, pad=12, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c8d1d1")
        ax.tick_params(colors=ink)
        ax.grid(color="#e0e6e4", linewidth=0.7)
        ax.set_axisbelow(True)
        (reference_line,) = ax.plot(
            [], [], color="#9ba9ae", linewidth=2.7, alpha=0.8, label="Reference"
        )
        (path_line,) = ax.plot([], [], color=color, linewidth=1.35, label="Solution")
        (reference_head,) = ax.plot([], [], "o", color="#9ba9ae", markersize=6)
        (head,) = ax.plot(
            [],
            [],
            "o",
            color=color,
            markersize=6,
            markeredgecolor=background,
            markeredgewidth=1,
        )
        ax.plot(path[0, 0], path[0, 1], "o", color=ink, markersize=4)
        error_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            color=ink,
            fontsize=10,
            bbox={"facecolor": background, "edgecolor": "none", "alpha": 0.9, "pad": 4},
        )
        ax.text(
            0.5,
            -0.17,
            f"Max path error: {max_path_error(path, reference):.3e}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color=ink,
        )
        artists.append((reference_line, path_line, reference_head, head, error_text))

    rollout_frames = max(2, round(seconds * fps))
    indices = np.rint(np.linspace(0, len(ts) - 1, rollout_frames)).astype(int)
    indices = np.concatenate([indices, np.full(2 * fps, len(ts) - 1, dtype=int)])

    def update(frame):
        index = indices[frame]
        clock.set_text(f"t = {ts[index]:.2f} / {ts[-1]:.2f}")
        changed = [clock]
        for path, items in zip(paths, artists, strict=True):
            ref_line, path_line, ref_head, head, error_text = items
            ref_line.set_data(reference[: index + 1].T)
            path_line.set_data(path[: index + 1].T)
            ref_head.set_data(reference[index : index + 1].T)
            head.set_data(path[index : index + 1].T)
            error_text.set_text(
                f"Current error: {np.linalg.norm(path[index] - reference[index]):.3e}"
            )
            changed.extend(items)
        return changed

    movie = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=1000 / fps, blit=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gif = output_dir / "controlled_nonlinear.gif"
    print(f"Rendering {gif}", flush=True)
    movie.save(gif, writer=animation.PillowWriter(fps=fps), dpi=100)
    update(len(indices) - 1)
    png = output_dir / "controlled_nonlinear.png"
    fig.savefig(png, dpi=160, facecolor=background)
    plt.close(fig)
    print(f"Saved {gif}\nSaved {png}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t1", type=float, default=4.0)
    parser.add_argument("--fine-steps", type=int, default=8192)
    parser.add_argument(
        "--steps",
        type=int,
        default=1024,
        help="Finest comparison mesh; also checks steps/4.",
    )
    parser.add_argument("--depth", type=int, choices=(2, 3), default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    options = parser.parse_args()
    if (
        options.steps < 4
        or options.steps % 4
        or options.fine_steps < options.steps
        or options.fine_steps % options.steps
    ):
        parser.error("--steps must be a positive multiple of 4 and divide --fine-steps")
    if options.t1 <= 0 or options.fps < 1 or options.seconds <= 0:
        parser.error("--t1, --fps, and --seconds must be positive")

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(options.seed)
    ts = jnp.linspace(0, options.t1, options.fine_steps + 1)
    increments = rng.normal(size=(options.fine_steps, 2)) * np.sqrt(
        options.t1 / options.fine_steps
    )
    xs = jnp.asarray(np.concatenate([np.zeros((1, 2)), increments.cumsum(axis=0)]))
    print("Computing independent piecewise-linear reference...", flush=True)
    reference_one = np.asarray(solve_reference(xs, substeps=1))
    reference = np.asarray(solve_reference(xs, substeps=2))
    reference_change = max_path_error(reference_one, reference)
    print(
        f"Reference change after doubling substeps: {reference_change:.3e}", flush=True
    )
    np.testing.assert_allclose(reference_one, reference, atol=1e-8, rtol=1e-8)

    print(" steps    frozen max error    controlled max error", flush=True)
    for steps in (options.steps // 4, options.steps):
        stride = options.fine_steps // steps
        knots = ts[::stride]
        frozen = np.asarray(
            solve_log_ode(ts, xs, knots, depth=options.depth, controlled=False)
        )
        controlled = np.asarray(
            solve_log_ode(ts, xs, knots, depth=options.depth, controlled=True)
        )
        sampled_reference = reference[::stride]
        if not np.isfinite(frozen).all() or not np.isfinite(controlled).all():
            raise FloatingPointError("A log-ODE solve produced non-finite states.")
        print(
            f"{steps:6d}    {max_path_error(frozen, sampled_reference):.6e}        {max_path_error(controlled, sampled_reference):.6e}",
            flush=True,
        )

    animate_paths(
        np.asarray(knots),
        frozen,
        controlled,
        sampled_reference,
        depth=options.depth,
        seed=options.seed,
        fps=options.fps,
        seconds=options.seconds,
        output_dir=options.output_dir,
    )


if __name__ == "__main__":
    main()
