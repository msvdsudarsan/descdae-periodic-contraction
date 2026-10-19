"""Comparison of implicit Euler with two other one-step methods (implicit midpoint, trapezoidal rule)
on Example 2. Only implicit Euler is covered by the theorem; the others are reported empirically."""
import sys, json, math, time
sys.path.insert(0, "src")
import numpy as np
from descdae import RCLadder, reference_periodic
from descdae.solvers import reduced_onestep
mod = RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3)
c = mod.contraction_rate(); T = mod.T
xs = reference_periodic(mod)
x10 = np.random.RandomState(7).uniform(-3, 3, mod.n)
e0 = mod.norm_P(x10 - xs(0.0)[0])
nper = int(math.ceil(math.log(max(e0, 1.0) / 1e-12) / (c * T))) + 3
rows = []
t0 = time.time()
for K in (4, 8, 16, 32, 64, 128, 256):
    for meth in ("euler", "midpoint", "trapezoid"):
        X = reduced_onestep(mod, x10, K, nper, meth)
        t = (T / K) * np.arange(len(X))
        E = np.array([mod.norm_P(r) for r in (X - xs(t))])
        win = len(t) - 3 * K
        rows.append(dict(K=K, h=T / K, method=meth, sup_last3=float(E[win:].max()), sup_all=float(E.max())))
        print(rows[-1], round(time.time() - t0), flush=True)
# observed orders
for meth in ("euler", "midpoint", "trapezoid"):
    r = [x for x in rows if x["method"] == meth]
    for a, b in zip(r[:-1], r[1:]):
        b["order"] = float(math.log2(a["sup_last3"] / b["sup_last3"]))
json.dump(dict(c=c, nper=nper, e0=e0, rows=rows), open("results/methods.json", "w"), indent=1)
print("METHODS DONE")
