"""Certified periodic steady state by implicit Euler shooting (Theorem 'end-to-end certificate').

For a computed trajectory x_0=xi, ..., x_K on one period (piecewise linear interpolant xhat) let
   eps = |x_K - x_0|_P            (shooting mismatch, computable)
   D   = defect bound of Theorem 'post' evaluated from grid data with the rigorous sampling margin.
Then  |xhat(t) - x*(t)|_P <= exp(-ct) eps/(1-exp(-cT)) + D/c  for 0<=t<=T, in particular the grid error is <= B := eps/(1-e^{-cT}) + D/c.

Certified algorithm: refine K while D/c > tol/2, otherwise iterate the period map, until B <= tol.
Baselines: (1) fixed K, stop when the shooting mismatch eps <= tol;  (2) K doubling, stop when two successive
discrete periodic solutions differ by <= tol (each solved to eps<=1e-10).
Cost = number of implicit Euler steps.  True error is measured against a Radau reference."""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from descdae import RCLadder, CoupledRCLadder, RCMesh, RCGraph, reference_periodic
from strengthen import LS_cert, F_cert

STEPS = 0
CNT = [0]

def instrument(m):
    orig = m.h
    def h(x, t=0.0):
        CNT[0] += 1
        return orig(x, t)
    m.h = h
    return m

def euler_step(m, x, t1, h):
    """One implicit Euler step for the reduced ODE, Newton with the analytic Schur complement."""
    global STEPS
    STEPS += 1
    y = x.copy(); I = np.eye(m.n)
    for _ in range(50):
        R = y - x - h * m.ftilde(y, t1)
        if m.norm_P(R) < 1e-13:
            break
        y = y - np.linalg.solve(I - h * m.schur(y, t1), R)
    return y

def run_period(m, xi, K):
    h = m.T / K
    X = [xi.copy()]
    for k in range(K):
        X.append(euler_step(m, X[-1], (k + 1) * h, h))
    return np.array(X)

def defect(m, X, K, L, F, S=4):
    h = m.T / K; D = 0.0; Xd = 0.0
    for k in range(K):
        fk1 = m.ftilde(X[k + 1], (k + 1) * h)
        Xd = max(Xd, m.norm_P(fk1))
        for s in range(S + 1):
            th = s / S
            xh = (1 - th) * X[k] + th * X[k + 1]
            D = max(D, m.norm_P(fk1 - m.ftilde(xh, (k + th) * h)))
    return D + 0.5 * (h / S) * (L * Xd + F)

def true_grid_error(m, X, K, xs):
    ts = np.arange(K + 1) * (m.T / K)
    ref = xs(ts)
    return max(m.norm_P(X[i] - ref[i]) for i in range(K + 1))

def certified(m, tol, xi0, K0, c, L, F, xs):
    global STEPS
    STEPS = 0; CNT[0] = 0; K = K0; xi = xi0.copy(); E = math.exp(-c * m.T); it = 0; hist = []
    while True:
        X = run_period(m, xi, K); it += 1
        eps = m.norm_P(X[-1] - xi); D = defect(m, X, K, L, F)
        B = eps / (1 - E) + D / c
        hist.append((K, eps, D / c, B))
        if B <= tol:
            return dict(K=K, periods=it, steps=STEPS, evals=CNT[0], bound=B, eps=eps, disc=D / c, true=true_grid_error(m, X, K, xs), hist=hist)
        if D / c > tol / 2:
            K *= 2
        xi = X[-1]
        if it > 400: raise RuntimeError("no convergence")

def baseline_fixed(m, tol, xi0, K, xs):
    global STEPS
    STEPS = 0; CNT[0] = 0; xi = xi0.copy(); it = 0
    while True:
        X = run_period(m, xi, K); it += 1
        if m.norm_P(X[-1] - xi) <= tol or it > 400:
            return dict(K=K, periods=it, steps=STEPS, evals=CNT[0], true=true_grid_error(m, X, K, xs))
        xi = X[-1]

def baseline_doubling(m, tol, xi0, K0, xs):
    global STEPS
    STEPS = 0; CNT[0] = 0; K = K0; prev = None; xi = xi0.copy(); tot = 0
    while True:
        for it in range(1, 400):
            X = run_period(m, xi, K)
            if m.norm_P(X[-1] - xi) <= 1e-10: break
            xi = X[-1]
        if prev is not None and m.norm_P(X[0] - prev) <= tol:
            return dict(K=K, steps=STEPS, evals=CNT[0], true=true_grid_error(m, X, K, xs))
        prev = X[0].copy(); K *= 2

if __name__ == "__main__":
    t0 = time.time()
    models = {"2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
              "4": RCMesh(), "5": RCGraph()}
    tols = [float(v) for v in os.environ.get("TOLS", "1e-1,3e-2,1e-2").split(",")]
    res = {}
    keys = sys.argv[1:] or list(models)
    for key in keys:
        m = instrument(models[key])
        c = m.contraction_rate(); L = LS_cert(m); F = F_cert(m)
        xs = reference_periodic(m)
        xi0 = np.zeros(m.n)
        rows = []
        for tol in tols:
            r = certified(m, tol, xi0, 8, c, L, F, xs)
            b1 = baseline_fixed(m, tol, xi0, 16, xs)
            b2 = baseline_doubling(m, tol, xi0, 8, xs)
            rows.append(dict(tol=tol, cert=r, fixed=b1, doubling=b2))
            print(f"ex{key} tol={tol:.0e}: CERT K={r['K']} steps={r['steps']} B={r['bound']:.2e} true={r['true']:.2e} | "
                  f"fixed K=16 steps={b1['steps']} true={b1['true']:.2e} | doubling K={b2['K']} steps={b2['steps']} true={b2['true']:.2e}", flush=True)
            res[key] = dict(c=c, L=L, F=F, rows=rows)
            json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "certified_%s.json" % "_".join(keys)), "w"), indent=1)
            assert r["true"] <= r["bound"] * (1 + 1e-9) and r["bound"] <= tol
        res[key] = dict(c=c, L=L, F=F, rows=rows)
        json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "certified_%s.json" % "_".join(keys)), "w"), indent=1)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "certified_%s.json" % "_".join(keys)), "w"), indent=1)
    print("time", time.time() - t0)
