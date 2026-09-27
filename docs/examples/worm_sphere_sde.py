"""A smooth Stratonovich SDE on the sphere via SO(3).

This evolves an SO(3)-valued SDE, then displays the rotation applied to the
north pole as a path on S^2. The same sampled Brownian path drives all solves:

* Georax `GeometricEuler` uses all fine Brownian increments.
* Roughrax `LogODE(RKMK(Heun()))` uses a coarser log-signature grid.
* Roughrax `Davie()` uses full signatures on the same coarse grid.
* Georax `SRKMK(GeneralShARK())` on the fine grid is used as a reference.

Playback speed reflects warmed solve time, excluding signature construction and
plotting. Display paths interpolate the saved rotations on SO(3).

The drift attracts the point towards the camera; two projected constant fields
drive tangent noise and a third noise field spins the frame. All fields are
smooth on SO(3), with no clipping or hard boundary confinement.

Run with:

    uv run python docs/examples/worm_sphere_sde.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
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
from georax import RKMK, SO, SRKMK, GeometricEuler, GeometricTerm
from scipy.spatial.transform import Rotation, Slerp

from roughrax import (
    Davie,
    LogODE,
    LogSignatureInterpolation,
    RoughTerm,
    SignatureInterpolation,
)

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

VIEW_ELEV = 24.0
VIEW_AZIM = -58.0
VISIBLE_DOT_MIN = 0.08
CAP_ATTRACTION = 1.0
TARGET_VIDEO_SECONDS = 15.0
END_PAUSE_SECONDS = 5.0
BACKGROUND = "#fafbf9"
INK = "#253746"
MUTED = "#61736e"
WARMUP_BEFORE_TIMING = True
DEFAULT_SEED = 7
DEFAULT_T1 = 5.0
DEFAULT_FINE_STEPS = 1024
DEFAULT_COARSE_STEPS = 64
DEFAULT_DEPTH = 2
DEFAULT_DIFFUSION_SCALE = 0.48
DEFAULT_FPS = 20
DEFAULT_DPI = 120
DEFAULT_FORMATS = "gif"


@dataclass(frozen=True)
class SphereSolve:
    name: str
    ts: np.ndarray
    rotations: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class TimedSolve:
    solve: SphereSolve
    elapsed_seconds: float
    finish_seconds: float


class PiecewiseLinearLevyPath(diffrax.AbstractPath):
    interpolation: diffrax.LinearInterpolation

    @property
    def t0(self):
        return self.interpolation.t0

    @property
    def t1(self):
        return self.interpolation.t1

    def evaluate(self, t0, t1=None, left=True, use_levy=False):
        if t1 is None:
            return self.interpolation.evaluate(t0, left=left)
        increment = self.interpolation.evaluate(t0, t1, left=left)
        if use_levy:
            return diffrax.SpaceTimeLevyArea(
                dt=t1 - t0,
                W=increment,
                H=jnp.zeros_like(increment),
            )
        return increment


def sample_brownian_path(
    *, seed: int, t1: float, steps: int, dim: int
) -> tuple[jax.Array, jax.Array]:
    ts = jnp.linspace(0.0, t1, steps + 1)
    dt = ts[1:] - ts[:-1]
    key = jax.random.PRNGKey(seed)
    increments = jax.random.normal(key, (steps, dim)) * jnp.sqrt(dt)[:, None]
    xs = jnp.concatenate(
        [jnp.zeros((1, dim), dtype=ts.dtype), jnp.cumsum(increments, axis=0)],
        axis=0,
    )
    return ts, xs


def sphere_points(rotations: np.ndarray) -> np.ndarray:
    north = np.array([0.0, 0.0, 1.0])
    return np.einsum("tij,j->ti", rotations, north)


def camera_direction(*, elev: float = VIEW_ELEV, azim: float = VIEW_AZIM) -> np.ndarray:
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    return np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )


def camera_frame(dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    center = camera_direction()
    up = np.array([0.0, 0.0, 1.0])
    first = np.cross(up, center)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    return (
        jnp.asarray(center, dtype=dtype),
        jnp.asarray(first, dtype=dtype),
        jnp.asarray(second, dtype=dtype),
    )


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def visible_sector_y0(dtype) -> jax.Array:
    north = np.array([0.0, 0.0, 1.0])
    rotation = rotation_between(north, camera_direction())
    return jnp.asarray(rotation, dtype=dtype)


def _tangent_to_so3_coords(y: jax.Array, tangent: jax.Array) -> jax.Array:
    body = y.T @ tangent
    return jnp.asarray([0.0, body[0], body[1]], dtype=y.dtype)


def cap_vector_fields(
    y: jax.Array, *, diffusion_scale: float
) -> tuple[jax.Array, jax.Array]:
    """Lift kappa P_p c and sigma P_p e_i to SO(3), retaining frame-spin noise."""
    center, e1, e2 = camera_frame(y.dtype)
    point = y[:, 2]
    spin_field = jnp.asarray([0.18 * diffusion_scale, 0.0, 0.0], dtype=y.dtype)

    def project_tangent(vector):
        return vector - jnp.dot(vector, point) * point

    drift_frame = _tangent_to_so3_coords(y, CAP_ATTRACTION * project_tangent(center))
    diffusion_rows = jnp.stack(
        [
            _tangent_to_so3_coords(y, diffusion_scale * project_tangent(e1)),
            _tangent_to_so3_coords(y, diffusion_scale * project_tangent(e2)),
            spin_field,
        ]
    )
    return drift_frame, diffusion_rows


def _make_sphere_solve(
    name: str,
    terms,
    solver: diffrax.AbstractSolver,
    ts: jax.Array,
    y0: jax.Array,
) -> Callable[[], SphereSolve]:
    stepsize_controller = diffrax.StepTo(ts)
    saveat = diffrax.SaveAt(ts=ts)

    def solve() -> SphereSolve:
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=ts.shape[0] + 4,
            throw=True,
        )
        rotations = np.asarray(jax.block_until_ready(sol.ys))
        return SphereSolve(name, np.asarray(ts), rotations, sphere_points(rotations))

    return solve


def make_geometric_euler_solve(
    fine_ts: jax.Array,
    brownian: jax.Array,
    *,
    y0: jax.Array,
    diffusion_scale: float,
) -> Callable[[], SphereSolve]:
    geometry = SO(3)

    def drift_fn(t, y, args):
        del t, args
        drift, _ = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return drift

    def diffusion_fn(t, y, args):
        del t, args
        _, diffusion_rows = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return diffusion_rows.T

    terms = diffrax.MultiTerm(
        GeometricTerm(drift_fn, geometry),
        diffrax.ControlTerm(
            diffusion_fn,
            diffrax.LinearInterpolation(ts=fine_ts, ys=brownian),
        ),
    )
    return _make_sphere_solve("GeometricEuler", terms, GeometricEuler(), fine_ts, y0)


def make_srkmk_reference_solve(
    fine_ts: jax.Array,
    brownian: jax.Array,
    *,
    y0: jax.Array,
    diffusion_scale: float,
) -> Callable[[], SphereSolve]:
    geometry = SO(3)

    def drift_fn(t, y, args):
        del t, args
        drift, _ = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return drift

    def diffusion_fn(t, y, args):
        del t, args
        _, diffusion_rows = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return diffusion_rows.T

    control = PiecewiseLinearLevyPath(
        diffrax.LinearInterpolation(ts=fine_ts, ys=brownian)
    )
    terms = diffrax.MultiTerm(
        GeometricTerm(drift_fn, geometry),
        diffrax.ControlTerm(diffusion_fn, control),
    )
    return _make_sphere_solve(
        "SRKMK(GeneralShARK)", terms, SRKMK(diffrax.GeneralShARK()), fine_ts, y0
    )


def make_rough_solve(
    fine_ts: jax.Array,
    brownian: jax.Array,
    coarse_ts: jax.Array,
    *,
    y0: jax.Array,
    diffusion_scale: float,
    depth: int,
    solver: Davie | LogODE,
) -> Callable[[], SphereSolve]:
    driver_ys = jnp.concatenate([fine_ts[:, None], brownian], axis=1)
    driver = diffrax.LinearInterpolation(ts=fine_ts, ys=driver_ys)
    interpolation = (
        SignatureInterpolation
        if isinstance(solver, Davie)
        else LogSignatureInterpolation
    )
    control = interpolation(
        driver,
        coarse_ts,
        depth=depth,
        solution="stratonovich",
    )

    def vector_field(y):
        drift, diffusion_rows = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return jnp.concatenate([drift[None, :], diffusion_rows], axis=0)

    term = RoughTerm(vector_field, control, SO(3))
    return _make_sphere_solve(
        "Davie"
        if isinstance(solver, Davie)
        else f"LogODE + RKMK({type(solver.solver.solver).__name__})",
        term,
        solver,
        coarse_ts,
        y0,
    )


def time_solve(solve_fn) -> tuple[SphereSolve, float]:
    if WARMUP_BEFORE_TIMING:
        solve_fn()
    start = time.perf_counter()
    solve = solve_fn()
    elapsed = time.perf_counter() - start
    return solve, elapsed


def normalise_finish_times(
    solves: list[tuple[SphereSolve, float]],
) -> list[TimedSolve]:
    slowest = max(elapsed for _, elapsed in solves)
    return [
        TimedSolve(solve, elapsed, TARGET_VIDEO_SECONDS * max(elapsed, 1e-12) / slowest)
        for solve, elapsed in solves
    ]


def validate_rotation_solve(solve: SphereSolve) -> None:
    eye = np.eye(3)
    orthogonality_error = np.linalg.norm(
        np.swapaxes(solve.rotations, -1, -2) @ solve.rotations - eye,
        axis=(-2, -1),
    ).max()
    min_det = np.linalg.det(solve.rotations).min()
    if not np.isfinite(orthogonality_error) or not np.isfinite(min_det):
        raise FloatingPointError(f"{solve.name} produced non-finite rotations.")
    if orthogonality_error > 2e-5 or min_det < 0.999:
        raise RuntimeError(
            f"{solve.name} left SO(3): "
            f"max ||R^T R - I||={orthogonality_error:.3e}, "
            f"min det(R)={min_det:.6f}."
        )


def validate_visible_sector(solves: list[SphereSolve]) -> None:
    direction = camera_direction()
    dots = [solve.points @ direction for solve in solves]
    min_dot = min(float(dot.min()) for dot in dots)
    if min_dot < VISIBLE_DOT_MIN:
        raise RuntimeError(
            "Path left the visible sphere sector. "
            f"Minimum camera-facing dot product was {min_dot:.3f}; "
            "try reducing --t1 or --diffusion-scale."
        )


def configure_axis(ax) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#dfe7e2", alpha=0.25, linewidth=0, shade=False)
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=4,
        cstride=4,
        color="#8ea39a",
        alpha=0.22,
        linewidth=0.55,
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_zlim(-1.02, 1.02)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_axis_off()


def final_point_error(solve: SphereSolve, reference: SphereSolve) -> float:
    return float(np.linalg.norm(solve.points[-1] - reference.points[-1]))


def make_animation(
    solves: list[TimedSolve],
    reference: SphereSolve,
    *,
    video_seconds: float,
    end_pause_seconds: float,
    fps: int,
    seed: int,
    depth: int,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    total_seconds = video_seconds + end_pause_seconds
    if total_seconds <= 0.0:
        raise ValueError("Total animation duration must be positive.")
    frames = max(2, int(round(total_seconds * fps)))
    frame_seconds = np.linspace(0.0, total_seconds, frames)
    fig = plt.figure(figsize=(13.5, 6.4), facecolor=BACKGROUND)
    fig.text(
        0.5,
        0.955,
        "One Brownian path on the sphere",
        ha="center",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.905,
        "Geometric Euler · LogODE · Manifold Davie",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    colors = ["#5479ab", "#bd7160", "#008779"]
    artists = []

    for i, (timed, color) in enumerate(zip(solves, colors, strict=True)):
        solve = timed.solve
        center = (i + 0.5) / len(solves)
        ax = fig.add_axes(
            [i / len(solves), 0.225, 1 / len(solves), 0.61],
            projection="3d",
            facecolor=BACKGROUND,
        )
        configure_axis(ax)
        fig.text(
            center,
            0.825,
            solve.name,
            ha="center",
            fontsize=13,
            weight="bold",
            color=INK,
        )
        fig.text(
            center,
            0.788,
            f"{len(solve.ts) - 1} steps · {timed.elapsed_seconds * 1000:.1f} ms",
            ha="center",
            fontsize=11,
            color=MUTED,
        )
        (path_line,) = ax.plot([], [], [], color=color, alpha=0.95, linewidth=2.2)
        (head,) = ax.plot(
            [],
            [],
            [],
            "o",
            color=color,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
        )
        ax.plot(*solve.points[0], "o", color=INK, markersize=4)
        clock = fig.text(center, 0.235, "", ha="center", fontsize=12, color=INK)
        fig.text(
            center,
            0.195,
            f"Playback finishes at {timed.finish_seconds:.1f} s",
            ha="center",
            fontsize=10,
            color=MUTED,
        )
        fig.text(
            center,
            0.145,
            f"Final point error  {final_point_error(solve, reference):.3e}",
            ha="center",
            fontsize=11,
            color=INK,
        )

        # Display-only geodesic interpolation: preserve every solver knot and keep
        # the moving head on the sphere. This work is outside the solve timing.
        interpolate = Slerp(solve.ts, Rotation.from_matrix(solve.rotations))
        display_ts = np.unique(
            np.concatenate([solve.ts, np.linspace(solve.ts[0], solve.ts[-1], 2049)])
        )
        display_points = sphere_points(interpolate(display_ts).as_matrix())
        artists.append(
            (path_line, head, clock, interpolate, display_ts, display_points)
        )

    fig.text(
        0.5,
        0.09,
        f"Depth {depth} · Seed {seed} · Reference: {reference.name}, {len(reference.ts) - 1} steps",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.045,
        "Playback scaled by warmed solve time · Signature construction and plotting excluded",
        ha="center",
        fontsize=10,
        color=MUTED,
    )

    def update(frame: int):
        display_second = frame_seconds[frame]
        changed = []
        for timed, (line, head, clock, interpolate, ts, points) in zip(
            solves, artists, strict=True
        ):
            solve = timed.solve
            progress = min(display_second / timed.finish_seconds, 1.0)
            t = solve.ts[0] + progress * (solve.ts[-1] - solve.ts[0])
            current = interpolate(float(t)).as_matrix()[:, 2]
            index = np.searchsorted(ts, t, side="left")
            path = np.concatenate([points[:index], current[None, :]])
            line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
            head.set_data_3d(*current[:, None])
            clock.set_text(
                f"t = {t:.2f} / {solve.ts[-1]:.2f}"
                + (" · complete" if progress == 1 else "")
            )
            changed.extend([line, head, clock])
        return changed

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )
    return fig, anim


def save_animation(anim, output: Path, *, fps: int, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib ffmpeg writer is not available.")
        writer = animation.FFMpegWriter(fps=fps, bitrate=2200)
    elif suffix == ".gif":
        if not animation.writers.is_available("pillow"):
            raise RuntimeError("Matplotlib pillow writer is not available.")
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported animation format: {output.suffix!r}.")
    anim.save(output, writer=writer, dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--t1", type=float, default=DEFAULT_T1)
    parser.add_argument("--fine-steps", type=int, default=DEFAULT_FINE_STEPS)
    parser.add_argument("--coarse-steps", type=int, default=DEFAULT_COARSE_STEPS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--diffusion-scale", type=float, default=DEFAULT_DIFFUSION_SCALE
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="Comma-separated animation formats to save. Supported: mp4,gif.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "docs/examples/outputs/worm_sphere_sde_side_by_side",
        help="Output path without extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fine_steps % args.coarse_steps != 0:
        raise ValueError("--fine-steps must be divisible by --coarse-steps.")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1.")

    fine_ts, brownian = sample_brownian_path(
        seed=args.seed,
        t1=args.t1,
        steps=args.fine_steps,
        dim=3,
    )
    stride = args.fine_steps // args.coarse_steps
    coarse_ts = fine_ts[::stride]
    y0 = visible_sector_y0(brownian.dtype)

    solve_euler = make_geometric_euler_solve(
        fine_ts,
        brownian,
        y0=y0,
        diffusion_scale=args.diffusion_scale,
    )
    solve_reference = make_srkmk_reference_solve(
        fine_ts,
        brownian,
        y0=y0,
        diffusion_scale=args.diffusion_scale,
    )
    rough_solves = [
        make_rough_solve(
            fine_ts,
            brownian,
            coarse_ts,
            y0=y0,
            diffusion_scale=args.diffusion_scale,
            depth=args.depth,
            solver=solver,
        )
        for solver in (LogODE(RKMK(diffrax.Heun())), Davie())
    ]
    timed_solves = normalise_finish_times(
        [time_solve(solve) for solve in [solve_euler, *rough_solves]]
    )
    reference = solve_reference()
    solutions = [timed.solve for timed in timed_solves] + [reference]
    for solve in solutions:
        validate_rotation_solve(solve)
    validate_visible_sector(solutions)
    for timed in timed_solves:
        print(
            f"{timed.solve.name}: {len(timed.solve.ts) - 1} steps, "
            f"warmed solve {timed.elapsed_seconds:.6f}s "
            f"(playback finishes at {timed.finish_seconds:.1f}s), "
            f"final point error vs {reference.name} "
            f"{final_point_error(timed.solve, reference):.3e}",
            flush=True,
        )

    fig, anim = make_animation(
        timed_solves,
        reference,
        video_seconds=TARGET_VIDEO_SECONDS,
        end_pause_seconds=END_PAUSE_SECONDS,
        fps=args.fps,
        seed=args.seed,
        depth=args.depth,
    )
    for format_name in [item.strip() for item in args.formats.split(",") if item]:
        output = args.output_stem.with_suffix(f".{format_name}")
        print(f"saving {output}")
        save_animation(anim, output, fps=args.fps, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
