"""
chain.py
════════
Тонкий класс-оркестратор OscillatorChain.
Хранит параметры и данные траектории; делегирует всю математику
в подмодули пакета fpu/.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")


from fpu.fourier import make_neg_k_index, qp_to_a
from fpu.physics import make_frequencies
from fpu.integrator import euler_maruyama
import fpu.plots as _plots


class OscillatorChain:
    """
    Одномерная цепочка L осцилляторов FPU-β с термостатом Ланжевена.

    Параметры
    ---------
    L          : число осцилляторов (≥ 3)
    gamma      : параметр термостата γ  (0 ≤ γ ≤ 0.5)
    varepsilon : сила нелинейности ε
    T_slow     : длительность в медленном времени τ = γt
    dt         : шаг по физическому времени (авто, если None)
    seed       : зерно ГПСЧ
    """

    def __init__(self, L, gamma, varepsilon, T_slow, dt=None, seed=None):
        if L < 3:
            raise ValueError("Нужно не менее трёх осцилляторов.")
        self.L          = int(L)
        self.gamma      = float(gamma)
        self.varepsilon = float(varepsilon)
        self.T_slow     = float(T_slow)
        self.seed       = seed

        # Спектральные параметры
        self.omega = make_frequencies(self.L)
        self.mk    = make_neg_k_index(self.L)

        # Шаг и физическое время
        self.dt  = (float(dt) if dt is not None
                    else min(0.05, 0.15 / float(self.omega.max())))
        self._T  = T_slow / gamma if gamma > 0.0 else T_slow

        # Данные траектории (после solve)
        self.tau_arr = self.a_arr = self.q_arr = None
        self.p_arr   = self.H_arr = self.H2_arr = self.H4_arr = None

    # Интегрирование

    def solve(self, q0=None, p0=None, a0=None, n_store=1000):
        """Запускает интегрирование и сохраняет траекторию."""
        if a0 is not None:
            a_init = np.asarray(a0, complex).reshape(self.L, 2).copy()
        elif q0 is not None:
            p_ = np.zeros(self.L) if p0 is None else np.asarray(p0, float)
            a_init = qp_to_a(np.asarray(q0, float), p_, self.omega, self.mk)
        else:
            j = np.arange(self.L)
            a_init = qp_to_a(
                0.5 * np.cos(2.0 * np.pi * j / self.L),
                np.zeros(self.L), self.omega, self.mk
            )

        data = euler_maruyama(
            a_init, self.omega, self.mk,
            self.varepsilon, self.gamma,
            self._T, self.dt, n_store, self.seed
        )
        for k, v in data.items():
            setattr(self, f"{k}_arr" if k not in ("tau",) else "tau_arr", v)

        self.tau_arr = data["tau"]
        self.a_arr   = data["a"]
        self.q_arr   = data["q"]
        self.p_arr   = data["p"]
        self.H_arr   = data["H"]
        self.H2_arr  = data["H2"]
        self.H4_arr  = data["H4"]
        return self


    def plot_hamiltonian(self, **kw):
        return _plots.plot_hamiltonian(
            self.tau_arr, self.H_arr, self.H2_arr, self.H4_arr,
            self.varepsilon, self.L, self.gamma, self.T_slow, **kw)

    def plot_mode_energies(self, modes=None, **kw):
        return _plots.plot_mode_energies(
            self.tau_arr, self.a_arr, self.L,
            self.gamma, self.varepsilon, modes, **kw)

    def plot_particle_energy(self, i=0, **kw):
        return _plots.plot_particle_energy(
            self.tau_arr, self.q_arr, self.p_arr,
            i, self.varepsilon, self.gamma, **kw)

    def plot_phase_portrait(self, i=0, **kw):
        return _plots.plot_phase_portrait(
            self.tau_arr, self.q_arr, self.p_arr,
            i, self.varepsilon, self.gamma, **kw)

    def plot_summary(self, particle_i=0, modes=None, **kw):
        return _plots.plot_summary(
            self.tau_arr, self.a_arr, self.q_arr, self.p_arr,
            self.H_arr, self.H2_arr, self.H4_arr,
            self.varepsilon, self.gamma, self.L, self.T_slow,
            particle_i, modes, **kw)