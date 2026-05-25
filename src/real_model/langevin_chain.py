"""
langevin_chain.py — открытая цепочка + ланжевеновские термостаты

Используется FPU:
    H = H_2 + ε·H_4
    H_2 = (1/2)Σ_j(p_j^2+q_j^2) + (1/2)Σ_{bonds}(q_{j+1}-q_j)^2
    H_4 = (1/4)Σ_{bonds}(q_{j+1}-q_j)^4
т.е. каждая связь учитывается один раз. Уравнения движения:
    q̇_j = p_j
    ṗ_j = -q_j + (q_{j+1}-q_j) - (q_j-q_{j-1})
           + ε[(q_{j+1}-q_j)^3 - (q_j-q_{j-1})^3]
           - γ_j p_j + √(2γ_j T_j) β̇_j

Термостаты:
  * bulk:    γ_j = ν, T_j = T_bulk(j) (на всех частицах)
  * границы: дополнительно слева (γ_L, T_L), справа (γ_R, T_R)
Два независимых термостата на концах сведены к одному эквивалентному:
    γ_eff = γ1+γ2, T_eff = (γ1 T1 + γ2 T2)/γ_eff.

Локальная энергия:
    E_j = (p_j^2+q_j^2)/2 + (1/2) Σ_{b∼j} [d_b^2/2 + (ε/4)d_b^4]
где d_b = q_{b+1}-q_b и «b∼j» — связь, инцидентная частице j.
Это гарантирует Σ_j E_j ≡ H.

Поток энергии через связь (j, j+1) в смысле Дымова (1.14):
    J_{j+1→j} = (p_{j+1}+p_j)·[(q_{j+1}-q_j) + ε(q_{j+1}-q_j)^3]
«Физический» поток «слева направо» противоположного знака:
    J_phys_{j→j+1} = -J_{j+1→j} > 0 при T_L > T_R в стационаре.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def forces(q: np.ndarray, eps: float) -> np.ndarray:
    """Гамильтонова сила F_j = -∂H/∂q_j (FPU-соглашение)."""
    bond = np.diff(q)
    F = -q.copy()
    F[:-1] += bond
    F[1:]  -= bond
    if eps != 0.0:
        b3 = bond ** 3
        F[:-1] += eps * b3
        F[1:]  -= eps * b3
    return F


def hamiltonian(q: np.ndarray, p: np.ndarray, eps: float):
    """Полный H и его части (H_2, εH_4)."""
    bond = np.diff(q)
    H2 = 0.5 * float(p @ p + q @ q) + 0.5 * float(bond @ bond)
    H4 = 0.25 * float(np.sum(bond ** 4))
    return H2 + eps * H4, H2, eps * H4


def local_energies(q: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    """E_j каждая связь делится поровну между двумя концами."""
    E = 0.5 * (p ** 2 + q ** 2)
    bond = np.diff(q)
    half_V = 0.25 * bond ** 2 + 0.125 * eps * bond ** 4
    E[:-1] += half_V
    E[1:]  += half_V
    return E


def energy_currents(q: np.ndarray, p: np.ndarray, eps: float) -> np.ndarray:
    """J_{j+1→j} положителен при потоке справа-налево."""
    bond = np.diff(q)
    return (p[:-1] + p[1:]) * (bond + eps * bond ** 3)



@dataclass
class ChainConfig:
    L: int = 64
    eps: float = 0.2
    nu: float = 0.01                 # bulk-трение
    T_bulk_mode: str = "linear"      # 'zero'|'linear'|'equal_mean'|'equal'|'off'
    T_bulk_value: float = 0.0
    gamma_L: float = 1.0;  T_L: float = 2.0
    gamma_R: float = 1.0;  T_R: float = 0.2
    dt: float = 0.02
    T_total: float = 500.0
    n_store: int = 600
    seed: int = 42
    q0_amp: float = 0.0
    p0_amp: float = 0.0

    def gamma_and_T(self):
        L = self.L
        if self.T_bulk_mode == "off":
            gamma = np.zeros(L); T_eff = np.zeros(L)
            gamma[0],  T_eff[0]  = self.gamma_L, self.T_L
            gamma[-1], T_eff[-1] = self.gamma_R, self.T_R
            return gamma, T_eff

        if self.T_bulk_mode == "zero":
            T_bulk = np.zeros(L)
        elif self.T_bulk_mode == "linear":
            T_bulk = np.linspace(self.T_L, self.T_R, L)
        elif self.T_bulk_mode == "equal_mean":
            T_bulk = np.full(L, 0.5 * (self.T_L + self.T_R))
        elif self.T_bulk_mode == "equal":
            T_bulk = np.full(L, self.T_bulk_value)
        else:
            raise ValueError(self.T_bulk_mode)

        gamma = np.full(L, self.nu)
        T_eff = T_bulk.copy()
        for idx, (g_b, T_b) in [
            (0,     (self.gamma_L, self.T_L)),
            (L - 1, (self.gamma_R, self.T_R)),
        ]:
            g1, t1 = self.nu, T_bulk[idx]
            gamma[idx] = g1 + g_b
            T_eff[idx] = (g1 * t1 + g_b * T_b) / gamma[idx]
        return gamma, T_eff


class Simulator:
    def __init__(self, cfg: ChainConfig):
        self.cfg = cfg
        self.gamma, self.T = cfg.gamma_and_T()
        self.rng = np.random.default_rng(cfg.seed)

        L = cfg.L
        self.q = cfg.q0_amp * self.rng.standard_normal(L)
        self.p = cfg.p0_amp * self.rng.standard_normal(L)

        self._a = np.exp(-self.gamma * cfg.dt)
        self._b = np.sqrt(self.T * (1.0 - self._a ** 2))

    def step(self):
        dt, eps = self.cfg.dt, self.cfg.eps
        self.p += 0.5 * dt * forces(self.q, eps)          # B
        self.q += 0.5 * dt * self.p                        # A
        self.p = self._a * self.p + self._b * self.rng.standard_normal(self.cfg.L)  # O
        self.q += 0.5 * dt * self.p                        # A
        self.p += 0.5 * dt * forces(self.q, eps)          # B

    def run(self, progress: bool = False):
        cfg = self.cfg
        n_steps = int(round(cfg.T_total / cfg.dt))
        stride  = max(1, n_steps // cfg.n_store)
        n_frames = n_steps // stride + 1

        L = cfg.L
        t_arr = np.empty(n_frames)
        q_arr = np.empty((n_frames, L))
        p_arr = np.empty((n_frames, L))
        E_arr = np.empty((n_frames, L))
        H_arr = np.empty(n_frames)
        J_arr = np.empty((n_frames, L - 1))

        def snap(k, t):
            t_arr[k] = t
            q_arr[k] = self.q
            p_arr[k] = self.p
            E_arr[k] = local_energies(self.q, self.p, cfg.eps)
            H_arr[k] = hamiltonian(self.q, self.p, cfg.eps)[0]
            J_arr[k] = energy_currents(self.q, self.p, cfg.eps)

        k = 0
        snap(k, 0.0)
        for i in range(1, n_steps + 1):
            self.step()
            if i % stride == 0 and k + 1 < n_frames:
                k += 1
                snap(k, i * cfg.dt)
                if progress and k % max(1, n_frames // 15) == 0:
                    print(f"   t={t_arr[k]:7.1f}  H={H_arr[k]:8.3f}")
        k1 = k + 1
        return {
            "t": t_arr[:k1], "q": q_arr[:k1], "p": p_arr[:k1],
            "E": E_arr[:k1], "H": H_arr[:k1], "J": J_arr[:k1],
            "gamma": self.gamma, "T": self.T,
        }
