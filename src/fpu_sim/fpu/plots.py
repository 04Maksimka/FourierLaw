"""
Визуализация результатов симуляции.

Все функции принимают данные явно (не объект класса).
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from .physics import mode_energies, particle_energy


def _signed_k(L: int) -> np.ndarray:
    """k > L//2 → k − L (знаковые волновые числа)."""
    k = np.arange(L)
    k[k > L // 2] -= L
    return k


def _time_label(gamma: float) -> str:
    return (r"Медленное время $\tau=\gamma t$"
            if gamma > 0.0 else r"Время $t$")


def plot_hamiltonian(
        tau, H, H2, H4, varepsilon, L, gamma, T_slow, figsize=(12, 4)
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    xl = _time_label(gamma)

    ax1.plot(tau, H, 'k', lw=1.8, label='$H$')
    ax1.set(xlabel=xl, ylabel='Энергия', title='Полный гамильтониан $H(\\tau)$')
    ax1.legend(fontsize=10);  ax1.grid(alpha=0.3)

    ax2.plot(tau, H2,              lw=1.5, label='$H_2$ (квадратичный)')
    ax2.plot(tau, varepsilon * H4, lw=1.5, linestyle='--',
             label=r'$\varepsilon H_4$ (нелинейный)')
    ax2.set(xlabel=xl, ylabel='Энергия', title='Компоненты гамильтониана')
    ax2.legend(fontsize=9);  ax2.grid(alpha=0.3)

    fig.suptitle(rf'$L={L}$,  $\gamma={gamma}$,  '
                 rf'$\varepsilon={varepsilon}$,  $T_{{\rm slow}}={T_slow}$',
                 fontsize=11)
    plt.tight_layout()
    return fig


def plot_mode_energies(tau, a_arr, L, gamma, varepsilon,
                       modes=None, figsize=(14, 9)):
    Ek   = mode_energies(a_arr)
    ks   = _signed_k(L)
    sidx = np.argsort(ks);  ks_s = ks[sidx];  Ek_s = Ek[:, sidx]
    xl   = _time_label(gamma)
    c10  = plt.get_cmap('tab10')

    if modes is None:
        modes = list(range(min(8, L // 2 + 1)))

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    for ci, k in enumerate(modes):
        kid = int(k) % L
        ax.plot(tau, Ek[:, kid], label=rf'$k={ks[kid]}$',
                color=c10(ci % 10), lw=1.3)
    ax.set(xlabel=xl, ylabel='$E_k(\\tau)$',
           title='Эволюция энергии мод (FPU-стиль)')
    ax.legend(fontsize=8, ncol=2);  ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    w = 0.37
    ax.bar(ks_s - w/2, Ek_s[0],  width=w, alpha=0.72,
           color='steelblue', edgecolor='k', label='Начало (τ=0)')
    ax.bar(ks_s + w/2, Ek_s[-1], width=w, alpha=0.72,
           color='tomato',    edgecolor='k', label=f'Финал (τ={tau[-1]:.2g})')
    ax.set(xlabel='Волновое число $k$', ylabel='$E_k$',
           title='Распределение энергии: начало vs финал')
    ax.legend(fontsize=8);  ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, :])
    pc = ax.pcolormesh(tau, ks_s, Ek_s.T, cmap='hot', shading='auto')
    plt.colorbar(pc, ax=ax, label='$E_k$', fraction=0.02, pad=0.01)
    ax.set(xlabel=xl, ylabel='Волновое число $k$',
           title=r'Тепловая карта $E_k(\tau)$')
    ax.set_yticks(ks_s[::max(1, L // 10)])

    fig.suptitle(rf'FPU-анализ:  $L={L}$,  $\gamma={gamma}$,  '
                 rf'$\varepsilon={varepsilon}$',
                 fontsize=12, fontweight='bold')
    return fig


def plot_particle_energy(tau, q_arr, p_arr, i, varepsilon, gamma,
                          figsize=(10, 4)):
    L  = q_arr.shape[1]
    i  = int(i) % L
    Ei = particle_energy(q_arr, p_arr, i, varepsilon)
    xl = _time_label(gamma)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(tau, Ei, color='darkorange', lw=1.4)
    ax.set(xlabel=xl, ylabel=rf'$E_{{{i}}}(\tau)$',
           title=rf'Энергия частицы $j={i}$   ($\varepsilon={varepsilon}$)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase_portrait(tau, q_arr, p_arr, i, varepsilon, gamma,
                         figsize=(6, 6)):
    L  = q_arr.shape[1]
    i  = int(i) % L
    qi = q_arr[:, i];  pi = p_arr[:, i]
    N  = len(qi)
    xl = _time_label(gamma)

    fig, ax = plt.subplots(figsize=figsize)
    cols = plt.cm.viridis(np.linspace(0.0, 1.0, max(1, N - 1)))
    for n in range(N - 1):
        ax.plot(qi[n:n+2], pi[n:n+2], color=cols[n], lw=0.9, alpha=0.85)
    ax.scatter(qi[0],  pi[0],  color='lime', s=90, zorder=6,
               edgecolors='k', lw=0.5, label='Начало')
    ax.scatter(qi[-1], pi[-1], color='red',  s=90, zorder=6,
               edgecolors='k', lw=0.5, label='Конец')
    ax.set(xlabel=rf'$q_{{{i}}}$', ylabel=rf'$p_{{{i}}}$',
           title=rf'Фазовый портрет частицы $j={i}$   ($\varepsilon={varepsilon}$)')
    ax.legend(fontsize=9);  ax.grid(alpha=0.3)

    sm = ScalarMappable(cmap='viridis',
                        norm=Normalize(vmin=tau[0], vmax=tau[-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label(xl, fontsize=9)
    plt.tight_layout()
    return fig


def plot_summary(tau, a_arr, q_arr, p_arr, H, H2, H4,
                 varepsilon, gamma, L, T_slow,
                 particle_i=0, modes=None, figsize=(17, 10)):
    Ek   = mode_energies(a_arr)
    ks   = _signed_k(L)
    sidx = np.argsort(ks);  ks_s = ks[sidx];  Ek_s = Ek[:, sidx]
    pi_i = int(particle_i) % L
    xl   = _time_label(gamma)
    c10  = plt.get_cmap('tab10')
    w    = 0.37

    if modes is None:
        modes = list(range(min(6, L // 2 + 1)))

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.37)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(tau, H,  'k',  lw=1.8, label='$H$')
    ax.plot(tau, H2, '--', lw=1.2, label='$H_2$')
    ax.plot(tau, varepsilon * H4, ':', lw=1.2, label=r'$\varepsilon H_4$')
    ax.set(xlabel=xl, ylabel='Энергия', title='Гамильтониан')
    ax.legend(fontsize=8);  ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1:])
    for ci, k in enumerate(modes):
        kid = int(k) % L
        ax.plot(tau, Ek[:, kid], label=rf'$k={ks[kid]}$',
                color=c10(ci % 10), lw=1.3)
    ax.set(xlabel=xl, ylabel='$E_k(\\tau)$',
           title='Энергия нормальных мод (FPU)')
    ax.legend(fontsize=8, ncol=2);  ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    Ei = particle_energy(q_arr, p_arr, pi_i, varepsilon)
    ax.plot(tau, Ei, color='darkorange', lw=1.3)
    ax.set(xlabel=xl, ylabel=rf'$E_{{{pi_i}}}$',
           title=f'Энергия частицы $j={pi_i}$')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    qi  = q_arr[:, pi_i];  p_i = p_arr[:, pi_i]
    N   = len(qi)
    cc  = plt.cm.viridis(np.linspace(0.0, 1.0, max(1, N - 1)))
    for n in range(N - 1):
        ax.plot(qi[n:n+2], p_i[n:n+2], color=cc[n], lw=0.75, alpha=0.85)
    ax.scatter(qi[0],  p_i[0],  color='lime', s=60, zorder=5,
               edgecolors='k', lw=0.4)
    ax.scatter(qi[-1], p_i[-1], color='red',  s=60, zorder=5,
               edgecolors='k', lw=0.4)
    ax.set(xlabel=rf'$q_{{{pi_i}}}$', ylabel=rf'$p_{{{pi_i}}}$',
           title=f'Фазовый портрет (ч. {pi_i})')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(ks_s - w/2, Ek_s[0],  width=w, alpha=0.72,
           color='steelblue', edgecolor='k', label='Нач.')
    ax.bar(ks_s + w/2, Ek_s[-1], width=w, alpha=0.72,
           color='tomato',    edgecolor='k', label='Фин.')
    ax.set(xlabel='$k$', ylabel='$E_k$', title='Спектр: начало vs финал')
    ax.legend(fontsize=8);  ax.grid(alpha=0.3, axis='y')

    fig.suptitle(rf'Цепочка осцилляторов:  $L={L}$,  '
                 rf'$\gamma={gamma}$,  $\varepsilon={varepsilon}$,  '
                 rf'$T_{{\rm slow}}={T_slow}$',
                 fontsize=13, fontweight='bold')
    return fig
