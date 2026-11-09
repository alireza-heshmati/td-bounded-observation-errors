"""MDP, policy, emission matrices and exact quantities (notation of the paper).

P[a] (N x N) transition matrices, r (N x A) rewards, phi (N x A) policy phi(a | o),
M (N x N) emission matrix with M(s, o) = Pr(O = o | S = s)."""
import numpy as np

import config as cfg


def sinkhorn(X, tol=1e-14, max_iter=100000):
    X = np.array(X, dtype=float)
    for _ in range(max_iter):
        X /= X.sum(axis=1, keepdims=True)
        X /= X.sum(axis=0, keepdims=True)
        if np.abs(X.sum(axis=1) - 1.0).max() < tol:
            break
    return X / X.sum(axis=1, keepdims=True)


def stationary(Ps):
    """Stationary distributions of a stack of ergodic chains."""
    k, n, _ = Ps.shape
    lhs = np.transpose(Ps, (0, 2, 1)) - np.eye(n)[None]
    lhs[:, -1, :] = 1.0
    rhs = np.zeros((k, n, 1))
    rhs[:, -1, 0] = 1.0
    return np.linalg.solve(lhs, rhs)[:, :, 0]


def mixing_time(P):
    """t_mix = min{t : max_s ||P^t(s, .) - rho||_TV <= 1/4}."""
    rho = stationary(P[None])[0]
    Q, t = np.eye(len(P)), 0
    while 0.5 * np.abs(Q - rho[None]).sum(1).max() > 0.25:
        Q, t = Q @ P, t + 1
    return t


def cumrows(X):
    """Row-wise cumulative sums with last entry 1, for sampling."""
    c = np.cumsum(X, axis=-1)
    return np.ascontiguousarray(c / c[..., -1:])


def base_model():
    """Random base matrices Q_a, rewards r ~ U[-1, 1] and the rare state s* (state 5)."""
    rng = np.random.default_rng(cfg.MDP_SEED)
    Q = np.stack([sinkhorn(rng.exponential(size=(cfg.N, cfg.N))) for _ in range(cfg.A)])
    r = rng.uniform(-1.0, 1.0, size=(cfg.N, cfg.A))
    s_star = int(rng.integers(cfg.N))
    return Q, r, s_star


def transitions(Q, s_star, rho_min):
    """P_a with rho(s*) = rho_min for every policy and emission matrix: every action moves from every other state
    to s* with the same probability c and leaves s* with probability LAZY, so P^phi(s, s*) = c and
    P^phi(s*, s*) = 1 - LAZY for every M, and rho(s*) = c / (c + LAZY)."""
    N, lazy = cfg.N, cfg.LAZY
    others = [i for i in range(N) if i != s_star]
    c = lazy * rho_min / (1.0 - rho_min)
    P = np.zeros((cfg.A, N, N))
    for a in range(cfg.A):
        Qt = Q[a][np.ix_(others, others)]
        Qt = Qt / Qt.sum(1, keepdims=True)
        for i, s in enumerate(others):
            P[a, s, others] = (1.0 - c) * ((1.0 - lazy) * np.eye(N - 1)[i] + lazy * Qt[i])
            P[a, s, s_star] = c
        P[a, s_star, s_star] = 1.0 - lazy
        P[a, s_star, others] = lazy * Q[a][s_star, others] / Q[a][s_star, others].sum()
    return P


def policy(s_star):
    """phi(a_2 | o) = 0.8, 0.2, 0.8, 0.2 on the other observations and 0.5 on s*."""
    phi = np.zeros((cfg.N, cfg.A))
    j = 0
    for s in range(cfg.N):
        if s == s_star:
            p2 = cfg.POLICY_RARE
        else:
            p2 = cfg.POLICY_ORDINARY[j % len(cfg.POLICY_ORDINARY)]
            j += 1
        phi[s] = [1.0 - p2, p2]
    return phi


def emission_patterns(K):
    """K random row-stochastic matrices W with zero diagonal: row s puts Dirichlet weights on 1 to N - 1 other
    states. They are drawn one after the other, so the first 20 are the same for every K >= 20."""
    rng = np.random.default_rng(cfg.PATTERN_SEED)
    W = np.zeros((max(K, 20), cfg.N, cfg.N))
    for k in range(len(W)):
        for s in range(cfg.N):
            others = np.array([o for o in range(cfg.N) if o != s])
            m = int(rng.integers(1, cfg.N))
            support = rng.choice(others, size=m, replace=False)
            W[k, s, support] = rng.dirichlet(np.ones(m))
    return W[:K]


def emission(eps, W):
    """M = (1 - eps/2) I + (eps/2) W, so that ||M - I||_inf = eps."""
    return (1.0 - eps / 2.0) * np.eye(cfg.N) + (eps / 2.0) * W


def exact_values(P, r, Ms, phi, gamma):
    """V_TD (value of the surrogate MDP with q = rho), V_M (phi applied to the true state), rho, mu and t_mix
    for every M."""
    I = np.eye(cfg.N)
    VM = np.linalg.solve(I - gamma * np.einsum('sa,ast->st', phi, P), (phi * r).sum(1))
    chains = np.einsum('kso,oa,ast->kst', Ms, phi, P, optimize=True)            # latent chain P^phi
    rho = stationary(chains)
    mu = np.einsum('ks,kso->ko', rho, Ms)
    B = np.einsum('ks,kso->kos', rho, Ms) / mu[:, :, None]                      # posterior B_rho
    r_til = np.einsum('kos,sa,oa->ko', B, r, phi, optimize=True)
    P_til = np.einsum('kos,ast,ktq,oa->koq', B, P, Ms, phi, optimize=True)
    VTD = np.linalg.solve(I[None] - gamma * P_til, r_til[:, :, None])[:, :, 0]
    tmix = np.array([mixing_time(Pk) for Pk in chains])
    return VTD, VM, rho, mu, tmix


def setting(gamma, K):
    """Everything for one discount factor and the first K emission matrices; arrays indexed (rho_min, eps, matrix)."""
    Q, r, s_star = base_model()
    phi = policy(s_star)
    W = emission_patterns(K)
    C, E = len(cfg.RHO_MIN), len(cfg.EPS_LIST)
    out = dict(r=r, phi=phi, W=W, s_star=s_star, RMAX=float(np.abs(r).max()),
               P=np.stack([transitions(Q, s_star, rs) for rs in cfg.RHO_MIN]),
               VTD=np.zeros((C, E, K, cfg.N)), VM=np.zeros((C, cfg.N)), rho=np.zeros((C, E, K, cfg.N)),
               mu=np.zeros((C, E, K, cfg.N)), tmix=np.zeros((C, E, K), dtype=int))
    for c in range(C):
        for ie, e in enumerate(cfg.EPS_LIST):
            vals = exact_values(out["P"][c], r, emission(e, W), phi, gamma)
            out["VTD"][c, ie], out["VM"][c], out["rho"][c, ie], out["mu"][c, ie], out["tmix"][c, ie] = vals
    out["bias"] = np.abs(out["VTD"] - out["VM"][:, None, None]).max(-1)          # ||V_TD - V_M||_inf
    return out


def bias_bound(eps, rho_min, rmax, gamma):
    """Bound B_rho of Theorem 2."""
    return (eps * (1.0 - rho_min) / rho_min + gamma * eps) * rmax / (1.0 - gamma) ** 2


def eps_bar(rho_min, gamma):
    """B_rho < 2 V_max if and only if eps < eps_bar."""
    return 2.0 * (1.0 - gamma) * rho_min / (1.0 - (1.0 - gamma) * rho_min)
