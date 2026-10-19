"""Example 3: RC ladder with coupled (non-diagonal) algebraic part."""
import sys, json, math
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
from descdae import CoupledRCLadder, implicit_euler, reference_periodic, sampled_contraction_margin, observed_rate
from descdae.analysis import second_derivative_bound
mod = CoupledRCLadder(gc=1.5, N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3)
c = mod.contraction_rate(); T = mod.T
xs = reference_periodic(mod)
M = second_derivative_bound(mod, xs)
Cm = np.diag(mod.C ** -0.5)
LS = float(eigh(Cm @ mod.Lam @ Cm, eigvals_only=True)[-1])
F = mod.V * mod.omega / (mod.Rs * np.sqrt(mod.C[0]))
Mcert = F * (1 + LS / c)
margin = sampled_contraction_margin(mod, 800)
ts, D, rate = observed_rate(mod)
x10 = np.random.RandomState(7).uniform(-3, 3, mod.n)
e0 = mod.norm_P(x10 - xs(0.0)[0])
nper = int(math.ceil(math.log(max(e0, 1.0) / 1e-12) / (c * T))) + 3
lh = mod.lipschitz_h_P()
rows = []
for K in (16, 32, 64, 128, 256):
    t, X1, X2, info = implicit_euler(mod, x10, K, nper)
    Xs = xs(t)
    E1 = np.array([mod.norm_P(r) for r in (X1 - Xs)])
    E2 = np.linalg.norm(X2 - np.array([mod.h(r, 0.0) for r in Xs]), axis=1)
    win = len(t) - 3 * K; h = T / K; b = h * M / (2 * c); sel = E1 > 1e-8
    rows.append(dict(K=K, h=h, sup=float(E1[win:].max()), bound=b, ratio=float(E1[win:].max() / b),
                     sup_all=float(E1.max()), gbound=max(e0, b), x2=float((E2[sel] / E1[sel]).max()), ell_h=lh,
                     newton=info["max_newton_iters"], res=info["max_alg_residual"]))
    print(rows[-1], flush=True)
out = dict(c=c, M=M, Mcert=Mcert, LS=LS, F=F, margin=margin, c_obs=rate, nper=nper, e0=e0, gc=mod.gc, rows=rows,
           m=mod.m, L22=mod.L22)
json.dump(out, open("results/coupled.json", "w"), indent=1)
print("COUPLED DONE", c, M, Mcert, margin, rate)
