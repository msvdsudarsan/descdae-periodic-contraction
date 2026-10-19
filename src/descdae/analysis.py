from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def second_derivative_bound(model, xstar, npts=8001):
    """M = max_t |x*''(t)|_P on a fine grid, using the exact identity
    x*'' = S(x*,t) x*' + d/dt f~(x*,t), with d/dt f~ = e1 u'(t)/(Rs C_1) for the ladder.
    Grid maximum of the (smooth) reference solution; not a certified bound."""
    ts = np.linspace(0.0, model.T, npts)
    X = xstar(ts)
    vals = []
    for j in range(npts):
        x, t = X[j], ts[j]
        xd = model.ftilde(x, t)
        if hasattr(model, "dsrc"):
            ft = model.dsrc(t) / model.C
        else:
            ft = np.zeros(model.n)
            ft[0] = model.V * model.omega * np.cos(model.omega * t) / (model.Rs * model.C[0])
        vals.append(model.norm_P(model.schur(x, t) @ xd + ft))
    return float(max(vals))


def observed_rate(model, seed=2026, tf=None):
    """Distance D(t) = |xa - xb|_P between two reduced-ODE solutions; returns t, D, fitted decay rate."""
    rng = np.random.RandomState(seed)
    c = model.contraction_rate()
    tf = tf if tf is not None else np.log(1e9) / c
    xa0 = rng.uniform(-3, 3, model.n)
    xb0 = rng.uniform(-3, 3, model.n)
    ts = np.linspace(0, tf, 2001)
    rhs = lambda t, x: model.ftilde(x, t)
    jac = lambda t, x: model.schur(x, t)
    A = solve_ivp(rhs, (0, tf), xa0, method="Radau", t_eval=ts, rtol=1e-10, atol=1e-12, jac=jac)
    B = solve_ivp(rhs, (0, tf), xb0, method="Radau", t_eval=ts, rtol=1e-10, atol=1e-12, jac=jac)
    D = np.array([model.norm_P(A.y[:, j] - B.y[:, j]) for j in range(len(ts))])
    rel = D / D[0]
    sel = (rel < 1e-2) & (rel > 1e-8)
    rate = -np.polyfit(ts[sel], np.log(D[sel]), 1)[0] if sel.sum() > 10 else float("nan")
    return ts, D, float(rate)


def sampled_contraction_margin(model, nsamples=2000, seed=0):
    """Largest eigenvalue of sym(P^{1/2} S P^{-1/2}) over sampled (x1,t); must be <= -c."""
    rng = np.random.RandomState(seed)
    Cs = np.sqrt(model.C)
    worst = -np.inf
    for _ in range(nsamples):
        x = rng.uniform(-12, 12, model.n) * rng.choice([0.01, 1.0, 5.0])
        t = rng.uniform(0, model.T)
        S = model.schur(x, t)
        Ss = (Cs[:, None] * S) / Cs[None, :]
        worst = max(worst, float(np.linalg.eigvalsh(0.5 * (Ss + Ss.T)).max()))
    return worst
