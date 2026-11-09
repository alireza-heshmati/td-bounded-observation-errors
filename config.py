"""Settings of the numerical example (Section V of the paper)."""

# model
N, A = 5, 2                        # states (= observations) and actions
RHO_MIN = [0.15, 0.05]             # stationary probability of the rare state, the same for every emission matrix
LAZY = 0.1                         # every action leaves the rare state with probability LAZY
POLICY_ORDINARY = [0.8, 0.2]       # phi(a_2 | o) on the other four observations (repeated)
POLICY_RARE = 0.5                  # phi(a_2 | o) on the rare observation
MDP_SEED = 2026                    # random base matrices and rewards

# observation errors: M = (1 - eps/2) I + (eps/2) W
EPS_LIST = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
PATTERN_SEED = 7                   # random matrices W
K = 20                             # number of emission matrices
GAMMAS = [0.9, 0.5, 0.1]

# TD runs (Fig. 1)
SEEDS = [1, 2, 3, 4, 5]
T_RUN = 10**8                      # updates per run
C_STEP, T_SHIFT = 200.0, 200.0     # alpha_t = 200 / (t + 200)
N_CKPT = 150                       # log-spaced checkpoints

RESULTS = "results"
FIGURES = "figures"
