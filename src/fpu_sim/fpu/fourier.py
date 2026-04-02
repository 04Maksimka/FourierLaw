"""
Преобразования Фурье: a_{k,σ} ↔ (qⱼ, pⱼ).

Соглашения:
  q̂_k = L^{-1/2} Σⱼ qⱼ exp(−i2πkj/L)   [знак «−»]
  p̂_k = L^{-1/2} Σⱼ pⱼ exp(+i2πkj/L)   [знак «+»]
  v_{k,σ} = p̂_k + iσω_k q̂_{−k}
  a_{k,σ} = exp(−iσω_k t) · v_{k,σ}
  Симметрия: a_{k,−1} = conj(a_{−k,+1})
"""
from __future__ import annotations
import numpy as np


def make_neg_k_index(L: int) -> np.ndarray:
    """mk[k] = (−k) mod L."""
    return (-np.arange(L, dtype=int)) % L


def a_to_qp(a, omega, mk, t=0.0):
    """(L,2) complex × params → q, p : (L,) float."""
    v_plus  = np.exp( 1j * omega * t) * a[:, 0]
    v_minus = np.exp(-1j * omega * t) * a[:, 1]
    hat_q = (v_plus[mk] - v_minus[mk]) / (2j * omega)
    hat_p = (v_plus + v_minus) * 0.5
    L = len(omega)
    q = np.real(np.sqrt(L) * np.fft.ifft(hat_q))
    p = np.real(np.sqrt(L) * np.fft.ifft(hat_p[mk]))
    return q, p


def qp_to_a(q, p, omega, mk, t=0.0):
    """q, p : (L,) float → a : (L,2) complex."""
    L = len(omega)
    hat_q     = np.fft.fft(q) / np.sqrt(L)
    hat_p     = np.fft.fft(p)[mk] / np.sqrt(L)
    hat_q_neg = hat_q[mk]
    v_plus  = hat_p + 1j * omega * hat_q_neg
    v_minus = hat_p - 1j * omega * hat_q_neg
    a = np.empty((L, 2), dtype=complex)
    a[:, 0] = np.exp(-1j * omega * t) * v_plus
    a[:, 1] = np.exp( 1j * omega * t) * v_minus
    return a


def apply_symmetry(a, mk):
    """a_{k,−1} ← conj(a_{−k,+1})."""
    a[:, 1] = np.conj(a[mk, 0])
    return a
