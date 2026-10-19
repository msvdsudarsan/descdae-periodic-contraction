"""Which ingredient matters: Euler and midpoint grid data, each with the piecewise linear and the Hermite interpolant.
The Euler/linear and midpoint/Hermite pairs are in certified.py and certified_mid.py; this script adds the two mixed pairs."""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from descdae import RCLadder, RCMesh, RCGraph, reference_periodic
from strengthen import LS_cert
import certified as ce
import certified_mid as cm

def run(m, tol, c, L, F, L2, F2, xs, stepper, defect, K0=8):
    ce.STEPS = 0; ce.CNT[0] = 0; K = K0; xi = np.zeros(m.n); E = math.exp(-c * m.T); it = 0
    while True:
        X = stepper(m, xi, K); it += 1
        eps = m.norm_P(X[-1] - xi); D = defect(m, X, K); B = eps / (1 - E) + D / c
        if B <= tol:
            return dict(K=K, periods=it, steps=ce.STEPS, evals=ce.CNT[0], bound=B, true=ce.true_grid_error(m, X, K, xs))
        if D / c > tol / 2: K *= 2
        xi = X[-1]
        if it > 600: raise RuntimeError("no convergence")

if __name__ == "__main__":
    models = {"2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3), "4": RCMesh(), "5": RCGraph()}
    tols = [float(v) for v in os.environ.get("TOLS", "1e-1,1e-2").split(",")]
    keys = sys.argv[1:] or list(models); res = {}
    for key in keys:
        m = ce.instrument(models[key]); c = m.contraction_rate(); L = LS_cert(m); F = ce.F_cert(m); L2 = cm.L2_cert(m); F2 = cm.F2_cert(m)
        xs = reference_periodic(m); rows = []
        for tol in tols:
            a = run(m, tol, c, L, F, L2, F2, xs, ce.run_period, lambda m_, X, K: cm.defect_mid(m_, X, K, L, L2, F2))
            b = run(m, tol, c, L, F, L2, F2, xs, cm.run_period_mid, lambda m_, X, K: ce.defect(m_, X, K, L, F))
            rows.append(dict(tol=tol, euler_hermite=a, mid_linear=b))
            print(f"ex{key} tol={tol:.0e}: Euler+Hermite K={a['K']} steps={a['steps']} B={a['bound']:.2e} true={a['true']:.2e} | mid+linear K={b['K']} steps={b['steps']} B={b['bound']:.2e} true={b['true']:.2e}", flush=True)
            res[key] = dict(rows=rows)
            json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "certified_pairs_%s.json" % "_".join(keys)), "w"), indent=1)
