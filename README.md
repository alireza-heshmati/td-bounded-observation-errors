# Temporal Difference Learning with Bounded Observation Errors

Code for the numerical example (Section V) of the paper

> A. Heshmati, M. Amiri and S. Magnússon, *Temporal Difference Learning with Bounded Observation Errors:
> Convergence, Bias and Sample Complexity*.

It runs tabular TD(0) on observations of a 5-state MDP whose states are sometimes reported wrongly, and draws
Fig. 1 (convergence) and Fig. 2 (bias) of the paper.

## Files

| File | Content |
|---|---|
| `config.py` | all settings of Section V |
| `model.py` | MDP, policy, emission matrices `M = (1 - eps/2) I + (eps/2) W`, exact `V_TD`, `V_M`, `rho`, `mu`, `t_mix` and the bias bound of Theorem 2 |
| `simulate.py` | TD on observations along a trajectory of the POMDP, simulated with numba |
| `run_convergence.py` | runs for Fig. 1: `alpha_t = 200/(t + 200)`, `v_0 = 0`, `T = 1e8`, 500 runs per setting; records the distances of `v_t` to `V_M` and to `V_TD` |
| `plot_figures.py` | Fig. 1 from the runs, the final distance to `V_TD` quoted in Section V, and Fig. 2, which is computed exactly |

## Usage

```
pip install -r requirements.txt
python run_convergence.py      # results/conv_g09.npz, conv_g05.npz, conv_g01.npz
python plot_figures.py         # figures/fig1_convergence.pdf, figures/fig2_bias.pdf
```

Fig. 1 needs about 3e12 TD updates (3 discount factors x 2 values of `rho_min` x 10 values of `eps` x 500 runs x
1e8 updates), spread over all CPU cores. While running, the script prints the share done, the elapsed time and
an estimate of the time left. The runs can be split by discount factor, for example
`python run_convergence.py --gammas 0.9`. Fig. 2 needs no runs. All random draws are seeded, so the figures and
the numbers in the paper are reproduced exactly.

## Setting

- 5 states, 2 actions, rewards drawn once from U[-1, 1], and the policy `phi(a_2 | o) = 0.8, 0.2, 0.8, 0.2` on four
  observations and `0.5` on the rare one.
- Two pairs of transition matrices, with stationary probability `rho_min = 0.15` and `0.05` of the rare state for
  every emission matrix, and mixing times between 11 and 15.
- Emission matrices `M = (1 - eps/2) I + (eps/2) W` with 20 random row-stochastic `W` with zero diagonal, so that
  `||M - I||_inf = eps`, for `eps = 0` and `0.005 * 2^k`, `k = 0, ..., 8`.
- Discount factors `gamma = 0.9, 0.5, 0.1`.
- Fig. 1: `||v_t - V_M||_inf` for 500 runs per setting (20 emission matrices x 5 start states x 5 seeds).
- Fig. 2: worst case and mean of the bias `||V_TD - V_M||_inf` over the 20 emission matrices, with the bound of
  Theorem 2.
