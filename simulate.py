"""TD on observations, simulated with numba along a trajectory of the POMDP:

  O_t ~ M(S_t, .),  A_t ~ phi(. | O_t),  R_t = r(S_t, A_t),  S_{t+1} ~ P_{A_t}(S_t, .),
  v_{t+1}(O_t) = v_t(O_t) + alpha_t (R_t + gamma v_t(O_{t+1}) - v_t(O_t)),

with v_0 = 0 and alpha_t = min(1, c / (t + t_shift)). Records ||v_t - V_M||_inf and ||v_t - V_TD||_inf at the
checkpoints ck."""
import time

import numpy as np
from numba import njit, prange


class Progress:
    """Prints the share of the TD updates done, the elapsed time and an estimate of the time left."""

    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()
        print(f"{total:.2e} TD updates in total", flush=True)

    def add(self, n, msg):
        self.done += n
        el = time.time() - self.t0
        left = el / self.done * (self.total - self.done)
        print(f"{msg}  [{100 * self.done / self.total:5.1f}% done, {el / 60:.1f} min, about {left / 60:.0f} min left]",
              flush=True)


@njit(cache=True)
def _draw(cum):
    u = np.random.random()
    k = 0
    while k < cum.shape[0] - 1 and u >= cum[k]:
        k += 1
    return k


@njit(cache=True)
def _run(Pcum, r, Mcum, PHIcum, s0, gamma, T, c, t_shift, ck, vm, vtd, seed, err, err_td):
    np.random.seed(seed)
    v = np.zeros(vm.shape[0])
    s = s0
    o = _draw(Mcum[s])
    a = _draw(PHIcum[o])
    j = 0
    while j < ck.shape[0] and ck[j] == 0:
        err[j] = np.max(np.abs(v - vm))
        err_td[j] = np.max(np.abs(v - vtd))
        j += 1
    for t in range(T):
        s2 = _draw(Pcum[a, s])
        o2 = _draw(Mcum[s2])
        a2 = _draw(PHIcum[o2])
        al = min(1.0, c / (t + t_shift))
        v[o] += al * (r[s, a] + gamma * v[o2] - v[o])
        s, o, a = s2, o2, a2
        if j < ck.shape[0] and t + 1 == ck[j]:
            err[j] = np.max(np.abs(v - vm))
            err_td[j] = np.max(np.abs(v - vtd))
            j += 1


@njit(parallel=True, cache=True)
def run_batch(Pcum, r, Mcums, PHIcum, s0s, gamma, T, c, t_shift, ck, vm, vtds, seeds):
    """Independent runs in parallel; run i uses the emission matrix Mcums[i], start state s0s[i], the value
    vtds[i] of the surrogate MDP for that matrix and seed seeds[i]. Returns both errors of every run."""
    err = np.full((s0s.shape[0], ck.shape[0]), np.nan)
    err_td = np.full((s0s.shape[0], ck.shape[0]), np.nan)
    for i in prange(s0s.shape[0]):
        _run(Pcum, r, Mcums[i], PHIcum, s0s[i], gamma, T, c, t_shift, ck, vm, vtds[i], seeds[i], err[i], err_td[i])
    return err, err_td
