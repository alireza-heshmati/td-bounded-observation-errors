"""Runs for Fig. 1: TD on observations from v_0 = 0 with alpha_t = 200 / (t + 200) for T = 1e8 updates, 500 runs
for every gamma, rho_min and eps (20 emission matrices x 5 start states x 5 seeds).

Saves results/conv_g09.npz, results/conv_g05.npz and results/conv_g01.npz with ||v_t - V_M||_inf (err) and
||v_t - V_TD||_inf (err_td) of every run at log-spaced checkpoints.
Usage: python run_convergence.py [--gammas 0.9 0.5 0.1]
"""
import argparse
import os

import numpy as np

import config as cfg
import model
from simulate import Progress, run_batch


def run_seed(q, key):
    """Seed of one run, derived from the base seed q and the setting of the run."""
    return np.int64(np.random.SeedSequence(entropy=int(q), spawn_key=tuple(int(x) for x in key))
                    .generate_state(1, dtype=np.uint32)[0])


def main(gammas):
    os.makedirs(cfg.RESULTS, exist_ok=True)
    ck = np.unique(np.logspace(0, np.log10(cfg.T_RUN), cfg.N_CKPT).astype(np.int64))
    runs = [(k, s0, q) for k in range(cfg.K) for s0 in range(cfg.N) for q in range(len(cfg.SEEDS))]
    run_k = np.array([k for k, _, _ in runs])
    run_s0 = np.array([s0 for _, s0, _ in runs], dtype=np.int64)
    prog = Progress(len(gammas) * len(cfg.RHO_MIN) * len(cfg.EPS_LIST) * len(runs) * cfg.T_RUN)
    for g in gammas:
        ex = model.setting(g, cfg.K)
        print(f"gamma={g:g}: t_mix between {ex['tmix'].min()} and {ex['tmix'].max()}")
        err = np.zeros((len(cfg.RHO_MIN), len(cfg.EPS_LIST), len(runs), len(ck)), dtype=np.float32)
        err_td = np.zeros_like(err)
        for c, rs in enumerate(cfg.RHO_MIN):
            Pcum = model.cumrows(ex["P"][c])
            for ie, e in enumerate(cfg.EPS_LIST):
                Mcums = model.cumrows(model.emission(e, ex["W"]))[run_k]
                seeds = np.array([run_seed(cfg.SEEDS[q], (round(1000 * g), c, ie, 0, s0, k, q)) for k, s0, q in runs])
                err[c, ie], err_td[c, ie] = run_batch(Pcum, ex["r"], Mcums, model.cumrows(ex["phi"]), run_s0, g,
                                                      cfg.T_RUN, cfg.C_STEP, cfg.T_SHIFT, ck, ex["VM"][c],
                                                      ex["VTD"][c, ie][run_k], seeds)
                prog.add(len(runs) * cfg.T_RUN, f"  rho_min={rs:g} eps={e:g}: mean ||v_T - V_M|| = "
                         f"{err[c, ie, :, -1].mean():.3e}, mean bias = {ex['bias'][c, ie].mean():.3e}")
        np.savez_compressed(os.path.join(cfg.RESULTS, f"conv_g{round(10 * g):02d}.npz"), gamma=g, ck=ck, err=err,
                            err_td=err_td)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gammas", type=float, nargs="+", default=cfg.GAMMAS)
    main(p.parse_args().gammas)
