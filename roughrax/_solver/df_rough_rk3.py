from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array


class SignatureIncrement(eqx.Module):
    x1: Array
    x2: Array
    x3: Array


class SignatureRDETerm(diffrax.AbstractTerm):
    vector_field: Callable = eqx.field(static=True)
    signature_path: Any

    def vf(self, t, y, args):
        return self.vector_field(t, y, args)

    def contr(self, t0, t1, **kwargs):
        return self.signature_path.evaluate(t0, t1, **kwargs)

    def prod(self, vf, control):
        return tree_contract_last_axis(vf, control.x1)

    def vf_color(self, t, y, args, a):
        return tree_select_last_axis(self.vf(t, y, args), a)


def tree_zero_like(y):
    return jtu.tree_map(jnp.zeros_like, y)


def tree_add(x, y):
    return jtu.tree_map(lambda xi, yi: xi + yi, x, y)


def tree_sub(x, y):
    return jtu.tree_map(lambda xi, yi: xi - yi, x, y)


def tree_mul(a, x):
    return jtu.tree_map(lambda xi: a * xi, x)


def tree_axpy(a, x, y):
    return jtu.tree_map(lambda xi, yi: yi + a * xi, x, y)


def tree_div(x, a):
    return jtu.tree_map(lambda xi: xi / a, x)


def tree_select_last_axis(vf_all, a):
    return jtu.tree_map(lambda z: z[..., a], vf_all)


def tree_contract_last_axis(vf_all, coeff):
    return jtu.tree_map(
        lambda z: jnp.tensordot(z, coeff, axes=([-1], [0])),
        vf_all,
    )


def validate_signature_shapes(sig):
    m = sig.x1.shape[0]
    if sig.x2.shape != (m, m):
        raise ValueError(f"x2 must have shape {(m, m)}, got {sig.x2.shape}")
    if sig.x3.shape != (m, m, m):
        raise ValueError(f"x3 must have shape {(m, m, m)}, got {sig.x3.shape}")


def signature_radius(sig, rho_floor):
    r1 = jnp.max(jnp.abs(sig.x1))
    r2 = jnp.sqrt(jnp.max(jnp.abs(sig.x2)))
    r3 = jnp.cbrt(jnp.max(jnp.abs(sig.x3)))
    rho_raw = jnp.maximum(r1, jnp.maximum(r2, r3))
    rho = jnp.maximum(rho_raw, rho_floor)
    return rho, rho_raw


class DFRoughRK3(diffrax.AbstractSolver[None]):
    """Derivative-free finite-difference extended rough Runge-Kutta method.

    This solver consumes true word-signature coordinates through depth 3. It is
    not a classical ``AbstractERK``/``ButcherTableau`` method, not a derivative
    RK method, not a Lie/RKMK/log-ODE method, and not a post-hoc B-series
    correction.
    """

    term_structure: ClassVar = SignatureRDETerm
    interpolation_cls: ClassVar = diffrax.LocalLinearInterpolation

    rho_floor: float = eqx.field(static=True, default=0.0)
    validate_shapes: bool = eqx.field(static=True, default=False)

    def order(self, terms):
        del terms
        return 3

    def strong_order(self, terms):
        del terms
        return None

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        sig = terms.contr(t0, t1)

        if self.validate_shapes:
            validate_signature_shapes(sig)

        rho, rho_raw = signature_radius(sig, self.rho_floor)
        rho_safe = jnp.where(rho_raw <= 0, jnp.ones_like(rho), rho)
        two_rho = 2.0 * rho_safe
        four_rho2 = 4.0 * rho_safe * rho_safe

        m = sig.x1.shape[0]
        dy = tree_zero_like(y0)

        k_all = terms.vf(t0, y0, args)
        dy = tree_add(dy, tree_contract_last_axis(k_all, sig.x1))

        q2_by_a = []

        for a in range(m):
            k_a = tree_select_last_axis(k_all, a)

            y_plus = tree_axpy(rho_safe, k_a, y0)
            y_minus = tree_axpy(-rho_safe, k_a, y0)

            v_plus_all = terms.vf(t0, y_plus, args)
            v_minus_all = terms.vf(t0, y_minus, args)

            q2_all = tree_div(tree_sub(v_plus_all, v_minus_all), two_rho)
            q2_by_a.append(q2_all)

            dy = tree_add(
                dy,
                tree_contract_last_axis(q2_all, sig.x2[a, :]),
            )

        for a in range(m):
            k_a = tree_select_last_axis(k_all, a)

            for b in range(m):
                k_b = tree_select_last_axis(k_all, b)
                q_ab = tree_select_last_axis(q2_by_a[a], b)

                y_chain_plus = tree_axpy(rho_safe, q_ab, y0)
                y_chain_minus = tree_axpy(-rho_safe, q_ab, y0)

                v_chain_plus_all = terms.vf(t0, y_chain_plus, args)
                v_chain_minus_all = terms.vf(t0, y_chain_minus, args)

                q_chain_all = tree_div(
                    tree_sub(v_chain_plus_all, v_chain_minus_all),
                    two_rho,
                )

                dy = tree_add(
                    dy,
                    tree_contract_last_axis(q_chain_all, sig.x3[a, b, :]),
                )

                k_a_plus_k_b = tree_add(k_a, k_b)
                k_a_minus_k_b = tree_sub(k_a, k_b)
                minus_k_a_plus_k_b = tree_sub(k_b, k_a)

                y_pp = tree_axpy(rho_safe, k_a_plus_k_b, y0)
                y_pm = tree_axpy(rho_safe, k_a_minus_k_b, y0)
                y_mp = tree_axpy(rho_safe, minus_k_a_plus_k_b, y0)
                y_mm = tree_axpy(-rho_safe, k_a_plus_k_b, y0)

                v_pp_all = terms.vf(t0, y_pp, args)
                v_pm_all = terms.vf(t0, y_pm, args)
                v_mp_all = terms.vf(t0, y_mp, args)
                v_mm_all = terms.vf(t0, y_mm, args)

                numerator = tree_add(
                    tree_sub(v_pp_all, v_pm_all),
                    tree_sub(v_mm_all, v_mp_all),
                )

                q_bush_all = tree_div(numerator, four_rho2)

                dy = tree_add(
                    dy,
                    tree_contract_last_axis(q_bush_all, sig.x3[a, b, :]),
                )

        y1_candidate = tree_add(y0, dy)
        y1 = jtu.tree_map(
            lambda old, new: jnp.where(rho_raw <= 0, old, new),
            y0,
            y1_candidate,
        )

        return y1, None, dict(y0=y0, y1=y1), None, diffrax.RESULTS.successful


__all__ = [
    "DFRoughRK3",
    "SignatureIncrement",
    "SignatureRDETerm",
    "signature_radius",
    "tree_add",
    "tree_axpy",
    "tree_contract_last_axis",
    "tree_div",
    "tree_mul",
    "tree_select_last_axis",
    "tree_sub",
    "tree_zero_like",
    "validate_signature_shapes",
]
