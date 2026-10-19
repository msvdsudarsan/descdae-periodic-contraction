"""Extra studies: (i) refinement + certified bound for M, (ii) inexact algebraic solves."""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import root
from descdae import RCLadder, reference_periodic
from descdae.analysis import second_derivative_bound

MODELS = {
    "1": RCLadder(N=1, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=5.0, omega=2.0, c0=4.0, cvar=0.0),
    "2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
}
summ = json.load(open("results/summary.json"))

def fd_M(model, xs, npts):
    ts = np.linspace(0, model.T, npts); X = xs(ts)
    xd = np.array([model.ftilde(X[j], ts[j]) for j in range(npts)])
    xdd = np.gradient(xd, ts, axis=0, edge_order=2)
    return max(model.norm_P(r) for r in xdd[5:-5])

def cert_M(model, c):
    """A-priori bound  M <= F (1 + L_S/c),  F = sup |d_t ftilde|_P,  L_S >= sup ||S||_P."""
    Cm = np.diag(model.C ** -0.5)
    LS = float(eigh(Cm @ model.Lam @ Cm, eigvals_only=True)[-1])
    F = model.V * model.omega / (model.Rs * np.sqrt(model.C[0]))
    return F * (1 + LS / c), F, LS

def inexact_euler(model, x10, K, nper, d, seed=11, mode='random'):
    rng = np.random.RandomState(seed); h = model.T / K
    x = x10.copy(); out = [x.copy()]
    for n in range(nper * K):
        t1 = (n + 1) * h
        delta = d * (rng.choice([-1.0, 1.0], model.N) if mode == 'random' else np.ones(model.N)) / np.sqrt(model.N)   # |delta|_2 = d
        F = lambda y: y - x - h * model.f1(y, model.h(y, t1) + delta, t1)
        y = root(F, x, tol=1e-13).x
        x = y; out.append(x.copy())
    return np.array(out)

rows_M, rows_in = [], []
res = {}
for key, mod in MODELS.items():
    s = summ["ex" + key]; c = s["c"]; xs = reference_periodic(mod)
    Mex = second_derivative_bound(mod, xs)
    for npts in (501, 1001, 2001, 4001, 8001, 16001):
        rows_M.append((key, npts, fd_M(mod, xs, npts), second_derivative_bound(mod, xs, npts)))
    Mc, F, LS = cert_M(mod, c)
    res["Mcert" + key] = Mc; res["Mex" + key] = Mex
    kappa = 1.0 / (mod.Ra * np.sqrt(mod.C.min()))      # sup |G12 v|_P / |v|

def sci(x): 
    return "0" if x == 0 else f"${x:.2e}$".replace("e-0", "\\times10^{-").replace("e+0", "\\times10^{").replace("e-", "\\times10^{-").replace("e+", "\\times10^{") .replace("$", "") if False else f"{x:.2e}"
with open("results/tables/Mrefine.tex", "w") as f:
    f.write("\\begin{tabular}{rrrr}\n\\toprule\nExample & grid points & $M$ (finite difference) & $M$ (identity)\\\\\n\\midrule\n")
    for k, n, a, b in rows_M: f.write(f"{k} & {n} & {a:.5f} & {b:.5f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
with open("results/tables/numbers_extra.tex", "w") as f:
    for k in "12":
        f.write(f"\\newcommand{{\\McertEx{'One' if k=='1' else 'Two'}}}{{{res['Mcert'+k]:.1f}}}\n")
json.dump(dict(res=res, M=rows_M), open("results/extra.json", "w"), indent=1)
print("EXTRA DONE")
