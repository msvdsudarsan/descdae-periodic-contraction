"""Inexact algebraic solves: random (3 seeds), biased and adversarial perturbations; tolerance sweep."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from descdae import RCLadder, reference_periodic, inexact_euler
MODELS = {
    "1": RCLadder(N=1, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=5.0, omega=2.0, c0=4.0, cvar=0.0),
    "2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
}
summ = json.load(open("results/summary.json"))
K = 64
rows = []
for key, mod in MODELS.items():
    s = summ["ex" + key]; c = s["c"]; M = s["M"]; h = mod.T / K; nper = s["n_periods"]
    xs = reference_periodic(mod)
    kappa = 1.0 / (mod.Ra * np.sqrt(mod.C.min()))
    x10 = np.random.RandomState(7).uniform(-3, 3, mod.n)
    ts = np.arange(nper * K + 1) * h; Xs = xs(ts); win = len(ts) - 3 * K
    for d in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        b = h * M / (2 * c) + kappa * d / c
        for mode, seeds in (("random", (11, 12, 13)), ("bias", (0,)), ("adv", (0,))):
            sup = 0.0
            for sd in seeds:
                X = inexact_euler(mod, x10, K, nper, d, xs=xs, seed=sd, mode=mode)
                sup = max(sup, max(mod.norm_P(r) for r in (X - Xs)[win:]))
            rows.append(dict(ex=key, mode=mode, d=d, sup=sup, bound=b, ratio=sup / b, tol_rule=h * M / (2 * kappa)))
            print(key, mode, d, sup, b, flush=True)
json.dump(rows, open("results/residual.json", "w"), indent=1)
print("RESIDUAL DONE")
