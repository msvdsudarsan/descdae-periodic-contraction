"""Checks of two corollaries: (V) implicit Euler with variable steps, (R) robustness of the periodic solution
to model perturbations (corollary of the a posteriori theorem). Examples 1-3."""
import sys, json, copy
sys.path.insert(0, "src")
import numpy as np
from scipy.optimize import root
from descdae import reference_periodic
from descdae.analysis import second_derivative_bound
from strengthen import models

out = {}
rs = np.random.RandomState(5)
for k in "123":
    m = models()[k]; c = m.contraction_rate(); xs = reference_periodic(m); M = second_derivative_bound(m, xs)
    rows = []
    for K in (8, 16, 32, 64):
        h = rs.uniform(0.4, 1.6, K); h *= m.T / h.sum()          # T-periodic grid pattern, uneven steps
        hmax = h.max(); ell = M * hmax / (2 * c)
        x = rs.uniform(-3, 3, m.n); t = 0.0; e0 = m.norm_P(x - xs(0.0)[0])
        prod = 1.0; worst = 0.0; n_per = int(np.ceil(np.log(1e12) / (c * m.T))) + 3
        for p in range(n_per):
            for hn in h:
                t1 = t + hn
                sol = root(lambda y: y - x - hn * m.ftilde(y, t1), x, jac=lambda y: np.eye(m.n) - hn * m.schur(y, t1), tol=1e-13)
                x = sol.x; t = t1; prod *= 1.0 / (1.0 + hn * c)
                e = m.norm_P(x - xs(t)[0]); bound = ell + prod * max(e0 - ell, 0.0)
                worst = max(worst, e / bound)
        rows.append(dict(K=K, hmax=float(hmax), worst_ratio=float(worst)))
    # robustness
    rob = []
    for par in ("V", "Rg"):
        for eps in (0.01, 0.05):
            mp = copy.deepcopy(m)
            if par == "V": mp.V *= 1 + eps
            else:
                mp.Rg *= 1 + eps; mp.m = 1 / mp.Ra + 1 / mp.Rg
                if hasattr(mp, "L22"): mp.L22 = mp.m + mp.a * mp.b + getattr(mp, "gc", 0) * 0
            xp = reference_periodic(mp)
            ts = np.linspace(0, m.T, 2001); X = xp(ts)
            D1 = D2 = 0.0
            for x, t in zip(X, ts):
                w = mp.h(x, t)
                D1 = max(D1, m.norm_P(mp.f1(x, w, t) - m.f1(x, w, t)))
                D2 = max(D2, float(np.linalg.norm(m.f2(x, w, t))))
            kappa = 1 / (m.Ra * np.sqrt(m.C.min())); D = D1 + kappa * D2 / m.m
            true = max(m.norm_P(xp(t)[0] - xs(t)[0]) for t in ts)
            rob.append(dict(par=par, eps=eps, true=float(true), bound=float(D / c), ratio=float(true / (D / c))))
    out[k] = dict(c=c, M=M, var=rows, rob=rob)
    print(k, [round(r["worst_ratio"], 3) for r in rows], [(r["par"], r["eps"], round(r["ratio"], 3)) for r in rob], flush=True)
json.dump(out, open(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "results", "robust.json"), "w"), indent=1)
