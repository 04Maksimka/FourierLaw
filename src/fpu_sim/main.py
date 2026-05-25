"""
main.py — FPU-эксперимент: вся энергия стартует в моде k=1.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chain import OscillatorChain


L, gamma, varepsilon, T_slow = 31, 0, 0.1, 20.0


# ── Начальное условие ─────────────────────────────────────────────────────
j  = np.arange(L)
q0 = np.ones_like(j)   # чистая мода k=1
p0 = np.ones_like(j) + j

# ── Симуляция ─────────────────────────────────────────────────────────────
chain = OscillatorChain(L=L, gamma=gamma, varepsilon=varepsilon,
                        T_slow=T_slow, seed=42)
t0 = time.time()
chain.solve(q0=q0, p0=p0, n_store=2000)
print(f"  Завершено за {time.time() - t0:.2f} с")

# ── Сохранение графиков ───────────────────────────────────────────────────
out = "outputs"
os.makedirs(out, exist_ok=True)

plots = {
    "hamiltonian":     chain.plot_hamiltonian(),
    "mode_energies":   chain.plot_mode_energies(modes=[1, ]),
    "particle_energy": chain.plot_particle_energy(i=1),
    "phase_portrait":  chain.plot_phase_portrait(i=1),
    "summary":         chain.plot_summary(particle_i=100,
                                          modes=[1, 300]),
}

for name, fig in plots.items():
    path = os.path.join(out, f"fpu_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")
