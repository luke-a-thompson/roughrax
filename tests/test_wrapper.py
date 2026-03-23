import diffrax
import jax.numpy as jnp
import pytest

from roughrax._wrapper import RoughRK


def test_wrapped_rough_rk_matches_base_solver_on_ode_step():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    y0 = jnp.array(1.0)
    t0 = 0.0
    t1 = 0.1

    base_solver = diffrax.Tsit5()
    wrapped_solver = RoughRK(base_solver)

    base_state = base_solver.init(term, t0, t1, y0, None)
    wrapped_state = wrapped_solver.init(term, t0, t1, y0, None)

    base_y1, _, _, _, _ = base_solver.step(
        term, t0, t1, y0, None, base_state, made_jump=False
    )
    wrapped_y1, wrapped_error, dense_info, _, _ = wrapped_solver.step(
        term, t0, t1, y0, None, wrapped_state, made_jump=False
    )

    assert wrapped_error is None
    assert jnp.allclose(wrapped_y1, base_y1)

    interpolation = wrapped_solver.interpolation_cls(t0=t0, t1=t1, **dense_info)
    assert jnp.allclose(interpolation.evaluate(t0), y0)
    assert jnp.allclose(interpolation.evaluate(t1), wrapped_y1)


def test_wrapped_rough_rk_runs_in_diffeqsolve():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = RoughRK(diffrax.Tsit5())

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=0.2,
        dt0=0.1,
        y0=jnp.array(1.0),
        stepsize_controller=diffrax.ConstantStepSize(),
    )

    assert solution.result == diffrax.RESULTS.successful
    assert solution.ys.shape == (1,)


def test_wrapped_rough_rk_rejects_non_erk_solver():
    with pytest.raises(TypeError, match="explicit diffrax RK solver"):
        RoughRK(object())
