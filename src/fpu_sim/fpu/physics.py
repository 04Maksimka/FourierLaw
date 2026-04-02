"""
Физика системы:
  - Спектральные параметры (ω_k)
  - Гамильтониан H = H₂ + ε H₄
  - Правая часть СДУ (детерминированная часть)
  - Производные величины: энергия мод, энергия частицы
"""
from __future__ import annotations
import numpy as np
from .fourier import a_to_qp


def make_frequencies(L: int) -> np.ndarray:
    """ω_k² = 1 + 4 sin²(πk/L), k = 0,…,L−1."""
    k = np.arange(L)
    return np.sqrt(1.0 + 4.0 * np.sin(np.pi * k / L) ** 2)


def hamiltonian(q, p, varepsilon):
    """H = H₂ + ε H₄, возвращает (H, H₂, H₄)."""
    dp = np.roll(q, -1) - q   # q_{j+1} − qⱼ
    dm = q - np.roll(q, +1)   # qⱼ − q_{j-1}
    H2 = 0.5 * float(p @ p + q @ q + dp @ dp + dm @ dm)
    H4 = 0.25 * float(np.sum(dp**4) + np.sum(dm**4))
    return H2 + varepsilon * H4, H2, H4


def compute_rhs(a, t, omega, mk, varepsilon, gamma):
    """
    Детерминированная часть da/dt в физическом времени:
        da_{k,±}/dt = ε·exp(∓iω_k t)·NL_k  −  γ·a_{k,±}

    NL_k — Фурье-образ нелинейной силы fⱼ = (q_{j+1}−qⱼ)³ − (qⱼ−q_{j-1})³.
    """
    q, _ = a_to_qp(a, omega, mk, t)
    dp = np.roll(q, -1) - q
    dm = q - np.roll(q, +1)
    f  = dp**3 - dm**3

    # NL_k = (1/√L) Σⱼ fⱼ exp(+i2πkj/L) = FFT(f)[(−k)%L] / √L
    NL = np.fft.fft(f)[mk] / np.sqrt(len(omega))

    ph = omega * t
    da = np.empty_like(a)
    da[:, 0] = varepsilon * np.exp(-1j * ph) * NL - gamma * a[:, 0]
    da[:, 1] = varepsilon * np.exp( 1j * ph) * NL - gamma * a[:, 1]
    return da


def mode_energies(a_arr):
    """
    E_k(τ) = ¼(|a_{k,+}|² + |a_{k,−}|²),  форма (N, L).
    Из статьи: H₂ = ¼ Σ_{k,σ} |a_{k,σ}|² = Σ_k E_k.
    """
    return 0.25 * (np.abs(a_arr[:, :, 0])**2 + np.abs(a_arr[:, :, 1])**2)


def particle_energy(q_arr, p_arr, i, varepsilon):
    """
    E_i(τ) — локальная энергия частицы i, форма (N,).
    Разбиение: каждая связь делится поровну. Σᵢ Eᵢ = H.
    """
    L  = q_arr.shape[1]
    ip = (i + 1) % L
    im = (i - 1) % L
    dp = q_arr[:, ip] - q_arr[:, i]
    dm = q_arr[:, i]  - q_arr[:, im]
    return (
        0.5 * p_arr[:, i]**2
      + 0.5 * q_arr[:, i]**2
      + 0.5 * dp**2
      + 0.5 * dm**2
      + varepsilon * 0.25 * (dp**4 + dm**4)
    )
