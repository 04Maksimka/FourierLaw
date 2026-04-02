"""
Численное интегрирование СДУ схемой Эйлера–Маруямы.

СДУ в физическом времени t:
    da_{k,σ}/dt = ε·exp(−iσω_k t)·NL_k − γ·a_{k,σ} + √γ·dβ_{k,σ}/dt

Схема:
    a_{n+1} = a_n + rhs(a_n, t_n)·dt + √(γ·dt)·ξ_n,   ξ_n ~ CN(0,1)
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from .physics import compute_rhs, hamiltonian
from .fourier import apply_symmetry, a_to_qp


def euler_maruyama(
    a0: np.ndarray,
    omega: np.ndarray,
    mk: np.ndarray,
    varepsilon: float,
    gamma: float,
    T_phys: float,
    dt: float,
    n_store: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Интегрирует СДУ и возвращает словарь с траекторией.

    Parameters
    ----------
    a0        : (L, 2) complex — начальные условия
    omega     : (L,)  float   — частоты ω_k
    mk        : (L,)  int     — индексы (−k) mod L
    varepsilon: float         — сила нелинейности ε
    gamma     : float         — параметр термостата γ
    T_phys    : float         — полное физическое время
    dt        : float         — шаг по времени
    n_store   : int           — число сохраняемых точек
    seed      : int | None    — зерно ГПСЧ

    Returns
    -------
    dict с ключами: tau, a, q, p, H, H2, H4
    """
    rng = np.random.default_rng(seed)
    L   = len(omega)

    n_steps  = max(1, int(np.ceil(T_phys / dt)))
    dt_exact = T_phys / n_steps
    store_ev = max(1, n_steps // n_store)
    sigma    = float(np.sqrt(gamma * dt_exact)) if gamma > 0.0 else 0.0

    a = a0.copy()

    tau_buf: List[float]      = []
    a_buf  : List[np.ndarray] = []
    q_buf  : List[np.ndarray] = []
    p_buf  : List[np.ndarray] = []
    H_buf  : List[float]      = []
    H2_buf : List[float]      = []
    H4_buf : List[float]      = []

    t = 0.0
    print(f"  [integrator] шагов={n_steps}, dt={dt_exact:.4g}, "
          f"T_phys={T_phys:.4g}, σ={sigma:.4g}")

    for step in range(n_steps + 1):
        if step % store_ev == 0:
            q, p      = a_to_qp(a, omega, mk, t)
            H, H2, H4 = hamiltonian(q, p, varepsilon)
            tau        = gamma * t if gamma > 0.0 else t

            tau_buf.append(tau)
            a_buf.append(a.copy())
            q_buf.append(q)
            p_buf.append(p)
            H_buf.append(H);  H2_buf.append(H2);  H4_buf.append(H4)

        if step == n_steps:
            break

        # Детерминированный шаг
        a += compute_rhs(a, t, omega, mk, varepsilon, gamma) * dt_exact

        # Шум (только a[:,0], a[:,1] восстанавливается из симметрии)
        if sigma > 0.0:
            a[:, 0] += sigma * (
                rng.standard_normal(L) + 1j * rng.standard_normal(L)
            )

        apply_symmetry(a, mk)
        t += dt_exact

    print(f"  [integrator] {len(tau_buf)} точек. "
          f"H₀={H_buf[0]:.5g}, H_fin={H_buf[-1]:.5g}")

    return dict(
        tau=np.array(tau_buf),
        a  =np.array(a_buf),
        q  =np.array(q_buf),
        p  =np.array(p_buf),
        H  =np.array(H_buf),
        H2 =np.array(H2_buf),
        H4 =np.array(H4_buf),
    )
