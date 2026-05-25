"""
main.py — запуск симуляции и генерация визуализаций.

Ожидания:
  1) В стационарном режиме профиль локальных энергий ⟨E_j⟩ монотонно
     спадает от (близкого к) T_L к T_R по всей цепочке → закон Фурье.
  2) Средний поток энергии ⟨J_{j,j+1}⟩ не зависит от j (кроме концов)
     и пропорционален (T_L - T_R)/L.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

from langevin_chain import ChainConfig, Simulator


OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
#  Эксперимент
# ──────────────────────────────────────────────────────────────────────

cfg = ChainConfig(
    L           = 64,
    eps         = 0.2,           # нелинейность H_4
    nu          = 0.01,          # слабое bulk-трение на каждой частице
    T_bulk_mode = "linear",      # T_bulk_j — линейная интерполяция T_L→T_R
    gamma_L     = 1.0,  T_L = 2.0,
    gamma_R     = 1.0,  T_R = 0.2,
    dt          = 0.02,
    T_total     = 800.0,
    n_store     = 700,
    seed        = 1,
)
print("Параметры:", cfg)
print("Запуск интегрирования…")
sim = Simulator(cfg)
res = sim.run(progress=True)

t    = res["t"]
Eij  = res["E"]     # (N, L)
Jij  = res["J"]     # (N, L-1)
H    = res["H"]
qij  = res["q"]
pij  = res["p"]

# Стационарное среднее берём по второй половине траектории
n_stat = len(t) // 2
E_stat = Eij[n_stat:].mean(axis=0)
J_stat = Jij[n_stat:].mean(axis=0)

print(f"\nСтационарная энергия:  E[0]={E_stat[0]:.3f}  E[-1]={E_stat[-1]:.3f}")
print(f"Средний поток:         J≈{J_stat[1:-1].mean():.4e}  (по внутренним связям)")


# ──────────────────────────────────────────────────────────────────────
#  1) Стационарный профиль энергии (закон Фурье)
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
j = np.arange(cfg.L)
ax.plot(j, E_stat, "o-", ms=4, lw=1.2, label=r"$\langle E_j\rangle$ (стационар)")
ax.axhline(cfg.T_L, color="tab:red", ls="--", alpha=0.5, label=fr"$T_L={cfg.T_L}$")
ax.axhline(cfg.T_R, color="tab:blue", ls="--", alpha=0.5, label=fr"$T_R={cfg.T_R}$")
# линейный fit по внутренним частицам
jj = j[2:-2]
coef = np.polyfit(jj, E_stat[2:-2], 1)
ax.plot(j, np.polyval(coef, j), "k--", alpha=0.4, label=f"линейный фит: наклон={coef[0]:+.3e}")
ax.set_xlabel(r"$j$ — номер частицы")
ax.set_ylabel(r"$\langle E_j\rangle$")
ax.set_title("Стационарный профиль локальной энергии\n(ожидаем монотонный спад — закон Фурье)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
jb = np.arange(cfg.L - 1) + 0.5
ax.plot(jb, J_stat, "s-", ms=3, lw=1, color="tab:green",
        label=r"$\langle J_{j,j+1}\rangle$")
ax.axhline(J_stat[1:-1].mean(), color="k", ls=":", alpha=0.6,
           label=f"среднее по внутренним = {J_stat[1:-1].mean():+.3e}")
ax.set_xlabel(r"связь $(j, j+1)$")
ax.set_ylabel(r"$\langle J_{j,j+1}\rangle$")
ax.set_title("Стационарный поток энергии\n(в законе Фурье он постоянен по $j$)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/01_stationary_profile.png", dpi=130)
plt.close()
print("→ 01_stationary_profile.png")


# ──────────────────────────────────────────────────────────────────────
#  2) Space-time диаграмма E_j(t)
# ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(
    Eij.T, aspect="auto", origin="lower",
    extent=[t[0], t[-1], -0.5, cfg.L - 0.5],
    cmap="inferno",
)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$j$")
ax.set_title(r"Пространственно-временная диаграмма локальной энергии $E_j(t)$")
plt.colorbar(im, ax=ax, label=r"$E_j$")
plt.tight_layout()
plt.savefig(f"{OUT}/02_spacetime.png", dpi=130)
plt.close()
print("→ 02_spacetime.png")


# ──────────────────────────────────────────────────────────────────────
#  3) Полный гамильтониан H(t)
# ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, H, lw=0.9)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$H(t)$")
ax.set_title("Полная энергия системы — выходит на стационар после переходного режима")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/03_hamiltonian.png", dpi=130)
plt.close()
print("→ 03_hamiltonian.png")


# ──────────────────────────────────────────────────────────────────────
#  4) Анимация E_j(t)
# ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.2))
line, = ax.plot(j, Eij[0], "o-", ms=3.5, lw=1.0, color="tab:orange")
hline_stat, = ax.plot(j, E_stat, "--", lw=1, color="gray", alpha=0.7,
                      label="среднее по стационару")
ax.axhline(cfg.T_L, color="tab:red", ls=":", alpha=0.4, label=fr"$T_L={cfg.T_L}$")
ax.axhline(cfg.T_R, color="tab:blue", ls=":", alpha=0.4, label=fr"$T_R={cfg.T_R}$")
ax.set_xlabel(r"$j$")
ax.set_ylabel(r"$E_j$")
ymax = np.percentile(Eij, 99.5) * 1.15
ax.set_ylim(0, ymax)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
title = ax.set_title("")

# прореживаем кадры, чтобы гиф не был огромным
step_frame = max(1, len(t) // 150)
idx = np.arange(0, len(t), step_frame)

def _update(i):
    k = idx[i]
    line.set_ydata(Eij[k])
    title.set_text(fr"$E_j(t)$,  $t={t[k]:.1f}$,   $L={cfg.L}$, "
                   fr"$\varepsilon={cfg.eps}$, $\nu={cfg.nu}$")
    return line, title

anim = FuncAnimation(fig, _update, frames=len(idx), interval=60, blit=False)
anim.save(f"{OUT}/04_energy_animation.gif", writer=PillowWriter(fps=20))
plt.close()
print("→ 04_energy_animation.gif")


# ──────────────────────────────────────────────────────────────────────
#  5) Фазовые портреты нескольких частиц
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, i_part in zip(axes, [0, cfg.L // 2, cfg.L - 1]):
    ax.plot(qij[n_stat:, i_part], pij[n_stat:, i_part], ".", ms=0.8, alpha=0.5)
    ax.set_xlabel(fr"$q_{{{i_part}}}$")
    ax.set_ylabel(fr"$p_{{{i_part}}}$")
    ax.set_title(f"Частица {i_part}  (T_eff≈{sim.T[i_part]:.2f})")
    ax.axis("equal"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/05_phase_portraits.png", dpi=130)
plt.close()
print("→ 05_phase_portraits.png")


# ──────────────────────────────────────────────────────────────────────
#  6) Сходимость профиля к стационару (скользящее среднее)
# ──────────────────────────────────────────────────────────────────────

# берём окна и смотрим как профиль стабилизируется
windows = [(0.00, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.00)]
fig, ax = plt.subplots(figsize=(8, 4.2))
for (a, b) in windows:
    ia, ib = int(a * len(t)), int(b * len(t))
    profile = Eij[ia:ib].mean(axis=0)
    ax.plot(j, profile, "o-", ms=3, lw=1, label=fr"$t\in[{t[ia]:.0f},{t[max(ib-1,ia)]:.0f}]$")
ax.axhline(cfg.T_L, color="tab:red",  ls=":", alpha=0.4)
ax.axhline(cfg.T_R, color="tab:blue", ls=":", alpha=0.4)
ax.set_xlabel("$j$"); ax.set_ylabel(r"$\overline{E_j}$ по окну")
ax.set_title("Сходимость профиля локальной энергии к стационарному")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/06_profile_convergence.png", dpi=130)
plt.close()
print("→ 06_profile_convergence.png")

# ──────────────────────────────────────────────────────────────────────
#  7) GIF-анимация колебаний частиц q_j(t)
# ──────────────────────────────────────────────────────────────────────

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5.5),
                                     gridspec_kw={"height_ratios": [3, 1.2]})

# ── верхняя панель: частицы как шарики, пружинки, смещения ──
j_pos = np.arange(cfg.L)                       # равновесные позиции
q_max = np.percentile(np.abs(qij), 99.5)       # масштаб смещений

# рисуем частицы как кружки, сдвинутые по вертикали на q_j
scatter = ax_top.scatter(j_pos, qij[0], s=28, c=Eij[0], cmap="coolwarm",
                         vmin=0, vmax=np.percentile(Eij, 97),
                         edgecolors="k", linewidths=0.3, zorder=3)
# пружинки — линия, соединяющая соседние частицы
springs, = ax_top.plot(j_pos, qij[0], "-", lw=0.6, color="gray", alpha=0.6, zorder=1)
# нулевая линия — равновесные позиции
ax_top.axhline(0, color="k", lw=0.3, alpha=0.4)
ax_top.set_xlim(-1, cfg.L)
ax_top.set_ylim(-q_max * 1.3, q_max * 1.3)
ax_top.set_ylabel(r"смещение $q_j$")
ax_top.set_xlabel(r"$j$")
ttl7 = ax_top.set_title("")
cb = plt.colorbar(scatter, ax=ax_top, label=r"$E_j$", pad=0.02, aspect=25)

# ── нижняя панель: мгновенный профиль p_j (импульсы) ──
bar_p = ax_bot.bar(j_pos, pij[0], width=0.8, color="tab:cyan", alpha=0.7)
ax_bot.set_xlim(-1, cfg.L)
p_max = np.percentile(np.abs(pij), 99)
ax_bot.set_ylim(-p_max * 1.2, p_max * 1.2)
ax_bot.set_ylabel(r"$p_j$")
ax_bot.set_xlabel(r"$j$")

plt.tight_layout()

step7 = max(1, len(t) // 150)
idx7 = np.arange(0, len(t), step7)

def _update7(i):
    k = idx7[i]
    # частицы
    offsets = np.column_stack([j_pos, qij[k]])
    scatter.set_offsets(offsets)
    scatter.set_array(Eij[k])
    # пружинки
    springs.set_ydata(qij[k])
    ttl7.set_text(fr"Колебания частиц $q_j(t)$,  $t={t[k]:.1f}$")
    # импульсы
    for rect, h in zip(bar_p, pij[k]):
        rect.set_height(h)
    return scatter, springs, ttl7

anim7 = FuncAnimation(fig, _update7, frames=len(idx7), interval=60, blit=False)
anim7.save(f"{OUT}/07_particle_positions.gif", writer=PillowWriter(fps=20))
plt.close()
print("→ 07_particle_positions.gif")


# ──────────────────────────────────────────────────────────────────────
#  8) Временной спектр — действия I_k(t) в Фурье-представлении
# ──────────────────────────────────────────────────────────────────────

L = cfg.L
k_modes = np.arange(L)
omega_k = np.sqrt(1.0 + 4.0 * np.sin(np.pi * k_modes / L) ** 2)

N_frames = len(t)
I_kt = np.empty((N_frames, L))
for n in range(N_frames):
    q_hat = np.fft.fft(qij[n]) / np.sqrt(L)
    p_hat = np.fft.fft(pij[n]) / np.sqrt(L)
    I_kt[n] = (np.abs(p_hat) ** 2 + omega_k ** 2 * np.abs(q_hat) ** 2) / (2 * omega_k)

# ── 8a) Динамика действий для выбранных мод ──
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

# верхняя панель: несколько мод I_k(t)
modes_show = [1, 2, 3, 5, 10, L // 4, L // 2]
modes_show = [m for m in modes_show if m < L]
ax = axes[0]
for m in modes_show:
    ax.plot(t, I_kt[:, m], lw=0.7, alpha=0.85, label=fr"$I_{{{m}}}(t)$")
ax.set_ylabel(r"$I_k(t)$")
ax.set_title(r"Динамика действий $I_k(t)$ в Фурье-представлении"
             "\n(аналог переменных из статьи)")
ax.legend(ncol=4, fontsize=8)
ax.grid(alpha=0.3)

# нижняя панель: heat-map всех мод
ax = axes[1]
# ограничиваем до L//2 (положительные моды, отрицательные — зеркальное отражение)
n_show = L // 2
im8 = ax.imshow(
    I_kt[:, :n_show].T, aspect="auto", origin="lower",
    extent=[t[0], t[-1], -0.5, n_show - 0.5],
    cmap="magma",
    vmax=np.percentile(I_kt[:, 1:n_show], 97),
)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"мода $k$")
ax.set_title(r"Пространственно-временная диаграмма $I_k(t)$  (моды $k=0\ldots L/2$)")
plt.colorbar(im8, ax=ax, label=r"$I_k$", pad=0.02)

plt.tight_layout()
plt.savefig(f"{OUT}/08_mode_actions.png", dpi=130)
plt.close()
print("→ 08_mode_actions.png")

# ── 8b) Стационарный спектр ⟨I_k⟩ ──
fig, ax = plt.subplots(figsize=(8, 3.8))
I_stat = I_kt[n_stat:].mean(axis=0)
ax.semilogy(k_modes[:n_show], I_stat[:n_show], "o-", ms=4, lw=1)
ax.set_xlabel(r"мода $k$")
ax.set_ylabel(r"$\langle I_k \rangle$  (лог. шкала)")
ax.set_title("Стационарный спектр действий — распределение энергии по модам")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{OUT}/08b_stationary_spectrum.png", dpi=130)
plt.close()
print("→ 08b_stationary_spectrum.png")


# ──────────────────────────────────────────────────────────────────────
#  9) Heat-map потока J_{j,j+1}(t) — сходимость к стационару
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                         gridspec_kw={"height_ratios": [3, 1.2]})

ax = axes[0]
# clip для лучшей визуализации: медиана ± 3σ
J_med = np.median(Jij)
J_std = np.std(Jij[n_stat:])
vabs = min(3 * J_std + abs(J_med), np.percentile(np.abs(Jij), 98))
im9 = ax.imshow(
    Jij.T, aspect="auto", origin="lower",
    extent=[t[0], t[-1], 0, cfg.L - 1],
    cmap="RdBu_r",
    vmin=-vabs, vmax=vabs,
)
ax.set_ylabel(r"связь $(j, j\!+\!1)$")
ax.set_title(r"Поток энергии $J_{j,j+1}(t)$: сходимость к стационарному режиму"
             "\n(красный — поток влево, синий — вправо)")
plt.colorbar(im9, ax=ax, label=r"$J_{j,j+1}$", pad=0.02)

# нижняя панель: средний поток по всем связям как функция времени
ax = axes[1]
# скользящее среднее по связям
J_mean_t = Jij[:, 2:-2].mean(axis=1)   # средний внутренний поток
# кумулятивное среднее по времени
J_cumavg = np.cumsum(J_mean_t) / np.arange(1, len(t) + 1)
ax.plot(t, J_mean_t, lw=0.5, alpha=0.5, color="tab:green", label="мгновенный")
ax.plot(t, J_cumavg, lw=1.5, color="k", label="кумулятивное среднее")
ax.axhline(J_stat[2:-2].mean(), color="tab:red", ls="--", lw=1,
           label=f"стационар = {J_stat[2:-2].mean():.3e}")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\overline{J}(t)$")
ax.set_title(r"Средний внутренний поток $\overline{J}(t)$ → стационарное значение")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/09_current_heatmap.png", dpi=130)
plt.close()
print("→ 09_current_heatmap.png")
