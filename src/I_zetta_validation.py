#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Численная проверка уравнения для действия

    dI_zeta = (-2 I_zeta + 1) dτ + sqrt(2 I_zeta) dW_zeta,

полученного из эффективного уравнения в a-переменных

    da_zeta = (i ρ Ω_eff_zeta(I) a_zeta - a_zeta) dτ + dβ_zeta,

где β_zeta = β^1_zeta + i β^2_zeta и
    dβ_zeta d\bar{β}_zeta = 2 dτ.

Главная проверяемая идея: если Ω_eff_zeta(I) вещественна, то фазовый член
    i ρ Ω_eff_zeta(I) a_zeta
не входит в уравнение для |a_zeta|^2/2. Поэтому действие должно иметь тот же
закон, что и CIR-процесс выше.

Запуск:
    python validate_action_equation.py

После запуска в папке --outdir будут созданы рисунки:
    1) a_phase_portrait.png
    2) actions_from_a_heatmap.png
    3) action_path_comparison.png
    4) ensemble_mean_validation.png
    5) ito_residual_convergence.png

Зависимости:
    numpy, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    """Параметры численного эксперимента."""

    L: int = 7                  # нечётное число узлов/независимых представителей ζ=(k,+)
    rho: float = 6.0            # ρ = ε / γ
    T: float = 20.0             # финальное медленное время τ
    dt: float = 1.0e-3          # шаг по τ
    seed: int = 20260512
    alpha_phase: float = 0.35   # сила искусственной нелинейной фазовой зависимости Ω_eff(I)
    store_stride: int = 1       # можно поставить >1 для экономии памяти

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))


@dataclass
class APathResult:
    times: Array
    k: Array
    sigma: Array
    a: Array                    # shape: (n_store, n_modes), complex
    I_from_a: Array             # shape: (n_store, n_modes), real
    ito_residual: Array         # shape: (n_steps, n_modes), local discrete residual
    ito_residual_cumsum: Array  # shape: (n_steps + 1, n_modes), cumulative residual


@dataclass
class ActionPathResult:
    times: Array
    I: Array                    # shape: (n_store, n_modes)


def build_independent_modes(L: int) -> tuple[Array, Array]:
    """
    Выбираем независимые представители из пар ζ и -ζ.

    В курсовой переменные удовлетворяют a_ζ = conjugate(a_{-ζ}). Поэтому достаточно
    симулировать по одному представителю из каждой пары. Удобный выбор: все ζ=(k,+).
    Тогда число независимых комплексных мод равно L.
    """
    if L % 2 != 1:
        raise ValueError("В этой реализации L должен быть нечётным: L = 2 ell + 1.")
    ell = (L - 1) // 2
    k = np.arange(-ell, ell + 1, dtype=float)
    sigma = np.ones_like(k)
    return k, sigma


def omega_lattice(k: Array, L: int) -> Array:
    """Дисперсионное соотношение ω_k = sqrt(1 + 4 sin^2(π k/L))."""
    return np.sqrt(1.0 + 4.0 * np.sin(np.pi * k / L) ** 2)


def default_omega_eff(I: Array, k: Array, sigma: Array, L: int, alpha: float = 0.35) -> Array:
    """
    Пример вещественной эффективной частоты Ω_eff(I).

    Важно: для проверки уравнения на действия точная формула Ω_eff несущественна;
    существенна только вещественность Ω_eff. Если у вас уже есть явная формула
    из резонансного гамильтониана H_4^res, замените тело этой функции на неё.
    """
    base = omega_lattice(k, L)
    mean_action = np.mean(I)
    return sigma * (base + alpha * (I + mean_action))


def simulate_effective_a(
    cfg: SimulationConfig,
    omega_eff: Callable[[Array, Array, Array, int, float], Array] = default_omega_eff,
    a0: complex | Array = 0.0 + 0.0j,
) -> APathResult:
    """
    Интегрирует эффективное уравнение в a-переменных методом Эйлера--Маруямы.

    Схема:
        a_{n+1} = a_n + (i ρ Ω_eff(I_n) a_n - a_n) dt + Δβ_n,
        Δβ_n = sqrt(dt) (ξ^1_n + i ξ^2_n).

    Одновременно считается дискретный остаток формулы Ито для действия:
        R_n = ΔI_n - [(-2 I_n + 1) dt + Re(conj(a_n) Δβ_n)].

    Для точного непрерывного процесса сумма R_n стремится к нулю при dt -> 0.
    В дискретной схеме основной вклад R_n даёт замена квадратической вариации
    |Δβ_n|^2 / 2 на её среднее dt.
    """
    rng = np.random.default_rng(cfg.seed)
    k, sigma = build_independent_modes(cfg.L)
    n_modes = len(k)
    n_steps = cfg.n_steps
    dt = cfg.dt
    sqrt_dt = np.sqrt(dt)

    a = np.empty(n_modes, dtype=np.complex128)
    if np.isscalar(a0):
        a[:] = complex(a0)
    else:
        a0_arr = np.asarray(a0, dtype=np.complex128)
        if a0_arr.shape != (n_modes,):
            raise ValueError(f"a0 должен иметь shape {(n_modes,)}, получено {a0_arr.shape}")
        a[:] = a0_arr

    n_store = n_steps // cfg.store_stride + 1
    times_store = np.empty(n_store)
    a_store = np.empty((n_store, n_modes), dtype=np.complex128)
    I_store = np.empty((n_store, n_modes), dtype=float)
    residual = np.empty((n_steps, n_modes), dtype=float)
    residual_cumsum = np.zeros((n_steps + 1, n_modes), dtype=float)

    store_idx = 0
    times_store[store_idx] = 0.0
    a_store[store_idx] = a
    I_store[store_idx] = 0.5 * np.abs(a) ** 2
    store_idx += 1

    for n in range(n_steps):
        I_old = 0.5 * np.abs(a) ** 2
        dB1 = sqrt_dt * rng.normal(size=n_modes)
        dB2 = sqrt_dt * rng.normal(size=n_modes)
        dB = dB1 + 1j * dB2

        Om = omega_eff(I_old, k, sigma, cfg.L, cfg.alpha_phase)
        drift = (1j * cfg.rho * Om - 1.0) * a
        a_new = a + drift * dt + dB
        I_new = 0.5 * np.abs(a_new) ** 2

        # sqrt(2I_n) ΔW_n = Re(conj(a_n) Δβ_n), без деления на sqrt(I_n).
        martingale_increment = np.real(np.conjugate(a) * dB)
        action_sde_increment = (-2.0 * I_old + 1.0) * dt + martingale_increment
        residual[n] = (I_new - I_old) - action_sde_increment
        residual_cumsum[n + 1] = residual_cumsum[n] + residual[n]

        a = a_new

        if (n + 1) % cfg.store_stride == 0:
            times_store[store_idx] = (n + 1) * dt
            a_store[store_idx] = a
            I_store[store_idx] = I_new
            store_idx += 1

    return APathResult(
        times=times_store,
        k=k,
        sigma=sigma,
        a=a_store,
        I_from_a=I_store,
        ito_residual=residual,
        ito_residual_cumsum=residual_cumsum,
    )


def simulate_action_exact_cir(
    times: Array,
    I0: float | Array,
    seed: int,
) -> ActionPathResult:
    """
    Симулирует непосредственно уравнение действия
        dI = (1 - 2I) dτ + sqrt(2I) dW
    по точному переходному распределению CIR-процесса.

    Для CIR dX = κ(θ-X)dt + σ sqrt(X)dW имеем:
        κ = 2, θ = 1/2, σ^2 = 2.
    Тогда
        X_{t+dt} = c * χ'^2_df(λ),
        c = σ^2(1-exp(-κdt))/(4κ),
        df = 4κθ/σ^2 = 2,
        λ = 4κ exp(-κdt) X_t / (σ^2(1-exp(-κdt))).
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    n_store = len(times)

    I0_arr = np.asarray(I0, dtype=float)
    if I0_arr.ndim == 0:
        I = np.empty((n_store, 1), dtype=float)
        I[0, 0] = float(I0_arr)
    else:
        I = np.empty((n_store, I0_arr.size), dtype=float)
        I[0] = I0_arr.ravel()

    kappa = 2.0
    theta = 0.5
    sigma2 = 2.0
    df = 4.0 * kappa * theta / sigma2  # = 2

    for n in range(n_store - 1):
        dt = times[n + 1] - times[n]
        if dt <= 0:
            raise ValueError("times должен быть строго возрастающим")
        exp_kdt = np.exp(-kappa * dt)
        one_minus = 1.0 - exp_kdt
        c = sigma2 * one_minus / (4.0 * kappa)
        noncentrality = 4.0 * kappa * exp_kdt * I[n] / (sigma2 * one_minus)
        I[n + 1] = c * rng.noncentral_chisquare(df=df, nonc=noncentrality, size=I[n].shape)

    return ActionPathResult(times=times, I=I)


def theoretical_mean_action(times: Array, I0_mean: float = 0.0) -> Array:
    """E I(τ) = 1/2 + (E I(0) - 1/2) exp(-2τ)."""
    return 0.5 + (I0_mean - 0.5) * np.exp(-2.0 * times)


def ensemble_mean_validation(
    T: float = 6.0,
    dt: float = 2.0e-3,
    n_paths: int = 3000,
    seed: int = 20260512,
) -> tuple[Array, Array, Array, Array]:
    """
    Сравнивает E[I(τ)] из a-переменных и из точного CIR-перехода.

    Здесь берётся одна мода. Для a-уравнения используется Эйлер--Маруяма с
    произвольной вещественной фазовой частотой; фазовый член не должен влиять на среднее действия.
    """
    rng_a = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)

    n_steps = int(round(T / dt))
    stride = max(1, n_steps // 400)
    n_store = n_steps // stride + 1
    times = np.empty(n_store)
    mean_from_a = np.empty(n_store)
    mean_direct_I = np.empty(n_store)

    a = np.zeros(n_paths, dtype=np.complex128)
    I = np.zeros(n_paths, dtype=float)

    sqrt_dt = np.sqrt(dt)
    store_idx = 0
    times[store_idx] = 0.0
    mean_from_a[store_idx] = 0.0
    mean_direct_I[store_idx] = 0.0
    store_idx += 1

    # параметры точного CIR-перехода
    kappa = 2.0
    theta = 0.5
    sigma2 = 2.0
    df = 4.0 * kappa * theta / sigma2
    exp_kdt = np.exp(-kappa * dt)
    one_minus = 1.0 - exp_kdt
    c = sigma2 * one_minus / (4.0 * kappa)

    for n in range(n_steps):
        I_a_old = 0.5 * np.abs(a) ** 2
        # Нелинейная вещественная фазовая частота: только для демонстрации вращения.
        omega_eff = 1.0 + 0.35 * I_a_old
        # Экспоненциальный шаг для замороженной вещественной частоты.
        # Он устраняет искусственный рост амплитуды, который явная схема Эйлера
        # даёт для быстро вращающейся фазы.
        noise_scale = np.sqrt((1.0 - np.exp(-2.0 * dt)) / 2.0)
        dB = noise_scale * (rng_a.normal(size=n_paths) + 1j * rng_a.normal(size=n_paths))
        a = np.exp((-1.0 + 1j * 6.0 * omega_eff) * dt) * a + dB

        noncentrality = 4.0 * kappa * exp_kdt * I / (sigma2 * one_minus)
        I = c * rng_i.noncentral_chisquare(df=df, nonc=noncentrality, size=n_paths)

        if (n + 1) % stride == 0:
            times[store_idx] = (n + 1) * dt
            mean_from_a[store_idx] = np.mean(0.5 * np.abs(a) ** 2)
            mean_direct_I[store_idx] = np.mean(I)
            store_idx += 1

    return times, mean_from_a, mean_direct_I, theoretical_mean_action(times, I0_mean=0.0)


def residual_convergence_table(
    dts: list[float],
    T: float = 2.0,
    L: int = 3,
    seed: int = 20260512,
) -> tuple[Array, Array, Array]:
    """Считает убывание накопленного остатка формулы Ито при измельчении сетки."""
    rms_final = []
    rms_path = []
    for j, dt in enumerate(dts):
        cfg = SimulationConfig(L=L, T=T, dt=dt, seed=seed + 100 * j, store_stride=1)
        res = simulate_effective_a(cfg)
        final_cum = res.ito_residual_cumsum[-1]
        rms_final.append(np.sqrt(np.mean(final_cum ** 2)))
        rms_path.append(np.sqrt(np.mean(res.ito_residual_cumsum ** 2)))
    return np.asarray(dts), np.asarray(rms_final), np.asarray(rms_path)


def save_plots(outdir: Path, cfg: SimulationConfig) -> None:
    """Строит все рисунки численного эксперимента."""
    outdir.mkdir(parents=True, exist_ok=True)

    a_res = simulate_effective_a(cfg)
    action_res = simulate_action_exact_cir(
        times=a_res.times,
        I0=a_res.I_from_a[0],
        seed=cfg.seed + 17,
    )

    selected = cfg.L // 2  # центральная мода k=0 при выбранной нумерации

    # 1) Фазовый портрет a_ζ в комплексной плоскости.
    plt.figure(figsize=(7, 6))
    plt.plot(a_res.a[:, selected].real, a_res.a[:, selected].imag, linewidth=0.8)
    plt.xlabel(r"$\operatorname{Re} a_\zeta$")
    plt.ylabel(r"$\operatorname{Im} a_\zeta$")
    plt.title(r"Траектория выбранной моды $a_\zeta$ в комплексной плоскости")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "a_phase_portrait.png", dpi=200)
    plt.close()

    # 2) Тепловая карта действий, восстановленных из a-переменных.
    plt.figure(figsize=(9, 5))
    extent = [a_res.times[0], a_res.times[-1], 0, cfg.L - 1]
    plt.imshow(a_res.I_from_a.T, aspect="auto", origin="lower", extent=extent)
    plt.colorbar(label=r"$I_\zeta(\tau)=|a_\zeta(\tau)|^2/2$")
    plt.xlabel(r"$\tau$")
    plt.ylabel("номер независимой моды")
    plt.title(r"Действия, вычисленные по решению в $a$-переменных")
    plt.tight_layout()
    plt.savefig(outdir / "actions_from_a_heatmap.png", dpi=200)
    plt.close()

    # 3) Одна траектория: действие из a-переменных и прямое решение action-SDE.
    plt.figure(figsize=(9, 5))
    plt.plot(a_res.times, a_res.I_from_a[:, selected], linewidth=1.0, label=r"$|a_\zeta|^2/2$")
    plt.plot(action_res.times, action_res.I[:, selected], linewidth=1.0, alpha=0.8, label="прямое CIR-решение")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$I_\zeta(\tau)$")
    plt.title("Сравнение траекторий действия")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "action_path_comparison.png", dpi=200)
    plt.close()

    # 4) Сравнение средних по ансамблю с аналитическим средним.
    t_mean, mean_a, mean_I, mean_theory = ensemble_mean_validation(
        T=min(6.0, cfg.T),
        dt=max(cfg.dt, 2.0e-3),
        n_paths=3000,
        seed=cfg.seed,
    )
    plt.figure(figsize=(9, 5))
    plt.plot(t_mean, mean_a, linewidth=1.0, label=r"среднее из $a$-переменных")
    plt.plot(t_mean, mean_I, linewidth=1.0, label="среднее прямого action-SDE")
    plt.plot(t_mean, mean_theory, linestyle="--", linewidth=1.0, label=r"$\frac{1}{2}(1-e^{-2\tau})$")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\mathbb{E} I_\zeta(\tau)$")
    plt.title("Проверка среднего действия")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ensemble_mean_validation.png", dpi=200)
    plt.close()

    # 5) Сходимость остатка формулы Ито при уменьшении шага.
    dts, rms_final, rms_path = residual_convergence_table(
        dts=[4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4],
        T=2.0,
        L=3,
        seed=cfg.seed,
    )
    plt.figure(figsize=(7, 5))
    plt.loglog(dts, rms_final, marker="o", label="RMS конечного накопленного остатка")
    plt.loglog(dts, rms_path, marker="s", label="RMS накопленного остатка по траектории")
    plt.gca().invert_xaxis()
    plt.xlabel(r"$\Delta\tau$")
    plt.ylabel("RMS")
    plt.title("Убывание остатка дискретной формулы Ито")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "ito_residual_convergence.png", dpi=200)
    plt.close()

    # Текстовый отчёт с численными метриками.
    final_mean_error_a = float(abs(mean_a[-1] - mean_theory[-1]))
    final_mean_error_I = float(abs(mean_I[-1] - mean_theory[-1]))
    report = [
        "Validation report",
        "=================",
        f"L = {cfg.L}",
        f"rho = {cfg.rho}",
        f"T = {cfg.T}",
        f"dt = {cfg.dt}",
        f"selected mode k = {a_res.k[selected]:.0f}, sigma = {a_res.sigma[selected]:.0f}",
        "",
        "Mean validation at final stored ensemble time:",
        f"  |mean_from_a - theory|      = {final_mean_error_a:.6e}",
        f"  |mean_direct_action - theory| = {final_mean_error_I:.6e}",
        "",
        "Ito residual convergence:",
        "  dt          RMS_final       RMS_path",
    ]
    for dt, rf, rp in zip(dts, rms_final, rms_path):
        report.append(f"  {dt:<10.4g} {rf:<15.6e} {rp:<15.6e}")
    (outdir / "validation_report.txt").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate action equation from effective a-SDE.")
    parser.add_argument("--outdir", type=Path, default=Path("numerical_action_validation"))
    parser.add_argument("--L", type=int, default=7)
    parser.add_argument("--rho", type=float, default=6.0)
    parser.add_argument("--T", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--alpha-phase", type=float, default=0.35)
    parser.add_argument("--store-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        L=args.L,
        rho=args.rho,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
        alpha_phase=args.alpha_phase,
        store_stride=args.store_stride,
    )
    save_plots(args.outdir, cfg)
    print(f"Готово. Рисунки и отчёт сохранены в: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
