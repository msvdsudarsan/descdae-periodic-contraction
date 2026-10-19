"""Certified shooting with the implicit midpoint rule (second-order certificate).

x-hat is the cubic Hermite interpolant of the midpoint solution with derivative data f~(x_k,t_k); its residual
r(t) = x-hat'(t) - f~(x-hat(t),t) vanishes at the nodes and is O(h^2).  Between samples of spacing delta, |r| is bounded by Taylor
expansion from the samples, using values r and derivatives r' = x-hat'' - S x-hat' - f~_t computed at the samples and the
bound  |r''|_P <= |x-hat'''|_P + L|x-hat''|_P + L2|x-hat'|_P^2 + F2   (f~_xt = 0, additive source), see Lemma in the paper.
Constants for the RC networks: L2 = (4/(3 sqrt 3)) a b^2 / (Ra^3 m^3 cmin^{3/2}),  F2 = omega^2 sqrt(sum amp_j^2/C_j)."""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from descdae import RCLadder, RCMesh, RCGraph, reference_periodic
from strengthen import LS_cert, F_cert
import certified as ce

def L2_cert(m):
    dprime = 4.0 / (3.0 * math.sqrt(3.0)) * m.a * m.b ** 2
    return dprime / (m.Ra ** 3 * m.m ** 3 * m.C.min() ** 1.5)

def F2_cert(m):
    if hasattr(m, "src_amp"):
        return m.omega ** 2 * math.sqrt(np.sum(m.src_amp ** 2 / m.C[m.src_nodes]))
    return m.V * m.omega ** 2 / (m.Rs * math.sqrt(m.C[0]))

def f_t(m, t):
    """partial_t f~ (additive source)."""
    if hasattr(m, "dsrc"):
        return m.dsrc(t) / m.C
    v = np.zeros(m.n); v[0] = m.V * m.omega * math.cos(m.omega * t) / (m.Rs * m.C[0]); return v

def mid_step(m, x, t0, h):
    ce.STEPS += 1
    y = x.copy(); I = np.eye(m.n); tm = t0 + 0.5 * h
    for _ in range(50):
        ym = 0.5 * (x + y)
        R = y - x - h * m.ftilde(ym, tm)
        if m.norm_P(R) < 1e-13: break
        y = y - np.linalg.solve(I - 0.5 * h * m.schur(ym, tm), R)
    return y

def run_period_mid(m, xi, K):
    h = m.T / K; X = [xi.copy()]
    for k in range(K):
        X.append(mid_step(m, X[-1], k * h, h))
    return np.array(X)

def hermite(xa, xb, da, db, h, th):
    h00, h10, h01, h11 = 2*th**3 - 3*th**2 + 1, th**3 - 2*th**2 + th, -2*th**3 + 3*th**2, th**3 - th**2
    d00, d10, d01, d11 = 6*th**2 - 6*th, 3*th**2 - 4*th + 1, -6*th**2 + 6*th, 3*th**2 - 2*th
    s00, s10, s01, s11 = 12*th - 6, 6*th - 4, -12*th + 6, 6*th - 2
    x = h00*xa + h10*h*da + h01*xb + h11*h*db
    x1 = (d00*xa + d01*xb)/h + d10*da + d11*db
    x2 = (s00*xa + s01*xb)/h**2 + (s10*da + s11*db)/h
    return x, x1, x2

def defect_mid(m, X, K, L, L2, F2, S=4):
    h = m.T / K; nP = m.norm_P; D = 0.0
    dn = [m.ftilde(X[k], k * h) for k in range(K + 1)]
    for k in range(K):
        xa, xb, da, db = X[k], X[k + 1], dn[k], dn[k + 1]
        x3 = nP((12*(xa - xb))/h**3 + (6*(da + db))/h**2)
        pts = []
        for j in range(S + 1):
            th = j / S; t = (k + th) * h
            x, x1, x2 = hermite(xa, xb, da, db, h, th)
            r = x1 - m.ftilde(x, t)
            rp = x2 - m.schur(x, t) @ x1 - f_t(m, t)
            pts.append((nP(r), nP(rp), nP(x1)))
        x2max = max(nP(hermite(xa, xb, da, db, h, 0.0)[2]), nP(hermite(xa, xb, da, db, h, 1.0)[2]))
        delta = h / S
        x1max = max(p[2] for p in pts) + 0.5 * delta * x2max
        Lam2 = x3 + L * x2max + L2 * x1max ** 2 + F2
        for j in range(S):
            va, wa = pts[j][0], pts[j][1]; vb, wb = pts[j + 1][0], pts[j + 1][1]
            D = max(D, max(va + 0.5 * delta * wa, vb + 0.5 * delta * wb) + Lam2 * delta ** 2 / 8)
    return D

def certified_mid(m, tol, xi0, K0, c, L, L2, F2, xs):
    ce.STEPS = 0; ce.CNT[0] = 0; K = K0; xi = xi0.copy(); E = math.exp(-c * m.T); it = 0
    while True:
        X = run_period_mid(m, xi, K); it += 1
        eps = m.norm_P(X[-1] - xi); D = defect_mid(m, X, K, L, L2, F2)
        B = eps / (1 - E) + D / c
        if B <= tol:
            return dict(K=K, periods=it, steps=ce.STEPS, evals=ce.CNT[0], bound=B, eps=eps, disc=D / c, true=ce.true_grid_error(m, X, K, xs))
        if D / c > tol / 2: K *= 2
        xi = X[-1]
        if it > 400: raise RuntimeError("no convergence")

def baseline_doubling_mid(m, tol, xi0, K0, xs):
    """Step refinement with the midpoint rule: K = K0, 2K0, ...; each periodic solution to 1e-10; stop when two successive
    periodic solutions differ by less than tol (a heuristic without guarantee, same rule as certified.baseline_doubling)."""
    ce.STEPS = 0; ce.CNT[0] = 0; K = K0; prev = None; xi = xi0.copy()
    while True:
        for it in range(1, 400):
            X = run_period_mid(m, xi, K)
            if m.norm_P(X[-1] - xi) <= 1e-10: break
            xi = X[-1]
        if prev is not None and m.norm_P(X[0] - prev) <= tol:
            return dict(K=K, steps=ce.STEPS, evals=ce.CNT[0], true=ce.true_grid_error(m, X, K, xs))
        prev = X[0].copy(); K *= 2

if __name__ == "__main__":
    t0 = time.time()
    models = {"2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
              "4": RCMesh(), "5": RCGraph()}
    tols = [float(v) for v in os.environ.get("TOLS", "1e-1,3e-2,1e-2").split(",")]
    keys = sys.argv[1:] or list(models); res = {}
    for key in keys:
        m = ce.instrument(models[key]); c = m.contraction_rate(); L = LS_cert(m); L2 = L2_cert(m); F2 = F2_cert(m)
        xs = reference_periodic(m); rows = []
        print(f"ex{key}: c={c:.4f} L={L:.3f} L2={L2:.3f} F2={F2:.3f}", flush=True)
        for tol in tols:
            r = certified_mid(m, tol, np.zeros(m.n), 8, c, L, L2, F2, xs)
            rows.append(dict(tol=tol, mid=r, mid_doubling=baseline_doubling_mid(m, tol, np.zeros(m.n), 8, xs)))
            print(f"ex{key} tol={tol:.0e}: MID K={r['K']} steps={r['steps']} evals={r['evals']} B={r['bound']:.2e} true={r['true']:.2e}", flush=True)
            b = rows[-1]["mid_doubling"]
            print(f"ex{key} tol={tol:.0e}: midpoint refinement K={b['K']} steps={b['steps']} evals={b['evals']} true={b['true']:.2e}", flush=True)
            assert r["true"] <= r["bound"] * (1 + 1e-9) and r["bound"] <= tol
            res[key] = dict(c=c, L=L, L2=L2, F2=F2, rows=rows)
            json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "certified_mid_%s.json" % "_".join(keys)), "w"), indent=1)
    print("time", time.time() - t0)
