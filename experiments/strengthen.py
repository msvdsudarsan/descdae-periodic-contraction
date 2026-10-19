"""New results: (A) contraction and error bound for implicit midpoint, (B) residual-based a posteriori bound for
the periodic solution and a shooting stopping criterion, on Examples 1-4 (Example 4 = 8x8 RC mesh)."""
import sys, json, math, time
sys.path.insert(0, "src")
import numpy as np
from descdae import RCLadder, CoupledRCLadder, RCMesh, reference_periodic
from descdae.solvers import reduced_onestep
from descdae.analysis import second_derivative_bound
from scipy.linalg import eigh

def models():
    return {
     "1": RCLadder(N=1, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=5.0, omega=2.0, c0=4.0, cvar=0.0),
     "2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
     "3": CoupledRCLadder(gc=1.5, N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
     "4": RCMesh(),
    }

def LS_cert(m):
    Cm = np.diag(m.C ** -0.5)
    return float(eigh(Cm @ m.Lam @ Cm, eigvals_only=True)[-1])

def F_cert(m):
    if hasattr(m, "F_cert"):
        return m.F_cert()
    return m.V * m.omega / (m.Rs * np.sqrt(m.C[0]))

def M2_M3(m, xs, npts):
    """Grid maxima of |x''|_P and |x'''|_P of the periodic solution (x'' from the exact identity, x''' by
    periodic central differences of x''); empirical, not certified."""
    ts = np.linspace(0.0, m.T, npts, endpoint=False)
    X = xs(ts)
    acc = []
    for x, t in zip(X, ts):
        xd = m.ftilde(x, t)
        ft = m.dsrc(t) / m.C if hasattr(m, "dsrc") else np.concatenate([[m.V * m.omega * np.cos(m.omega * t) / (m.Rs * m.C[0])], np.zeros(m.n - 1)])
        acc.append(m.schur(x, t) @ xd + ft)
    A = np.array(acc)
    dt = m.T / npts
    A3 = (np.roll(A, -1, axis=0) - np.roll(A, 1, axis=0)) / (2 * dt)
    nP = lambda v: np.sqrt(np.sum(m.C * v * v, axis=1))
    return float(nP(A).max()), float(nP(A3).max())

def run_example(key, m, Ks, log, parts="AB"):
    out = {"key": key}
    c, T, n = m.contraction_rate(), m.T, m.n
    L = LS_cert(m); F = F_cert(m)
    xs = reference_periodic(m)
    M2, M3 = M2_M3(m, xs, 1000 if n > 10 else 2000)
    out.update(c=c, L=L, F=F, T=T, M2=M2, M3=M3)
    log(f"ex{key}: c={c:.4f} L={L:.3f} F={F:.3f} M2={M2:.4f} M3={M3:.4f}")
    rng = np.random.RandomState(5)
    x0 = xs(0.0)[0]
    # ---------- (A) midpoint: per-step contraction and error
    rowsA = []
    for K in (Ks if "A" in parts else ()):
        h = T / K
        gam = h * c / (1 + (h * L) ** 2 / 4)
        rho_mid = math.sqrt((1 - gam) / (1 + gam))
        rho_eul = 1 / (1 + h * c)
        a0 = x0 + rng.uniform(-1, 1, n); b0 = x0 + rng.uniform(-1, 1, n)
        A = reduced_onestep(m, a0, K, 2, "midpoint"); B = reduced_onestep(m, b0, K, 2, "midpoint")
        D = np.array([m.norm_P(u - v) for u, v in zip(A, B)])
        ratio_mid = float((D[1:] / D[:-1]).max())
        A = reduced_onestep(m, a0, K, 2, "euler"); B = reduced_onestep(m, b0, K, 2, "euler")
        D = np.array([m.norm_P(u - v) for u, v in zip(A, B)])
        ratio_eul = float((D[1:] / D[:-1]).max())
        # error against the exact periodic solution
        xi = x0 + 0.5 * rng.uniform(-1, 1, n)
        e0 = m.norm_P(xi - x0)
        nper = int(math.ceil(math.log(max(e0, 1e-3) / 1e-11) / (c * T))) + 2
        res = {}
        for meth in ("euler", "midpoint"):
            X = reduced_onestep(m, xi, K, nper, meth)
            t = h * np.arange(len(X))
            E = np.array([m.norm_P(r) for r in (X - xs(t))])
            res[meth] = float(E[-2 * K:].max())
        r = h ** 3 * (M3 / 24 + L * M2 / 8)
        level_mid = r / ((1 - rho_mid) * (1 + h * c / 2))
        level_eul = h * M2 / (2 * c)
        rowsA.append(dict(K=K, h=h, rho_mid_bound=rho_mid, ratio_mid_obs=ratio_mid, rho_eul_bound=rho_eul, ratio_eul_obs=ratio_eul,
                          err_mid=res["midpoint"], level_mid=level_mid, err_eul=res["euler"], level_eul=level_eul))
        log(f"  A K={K}: mid ratio obs {ratio_mid:.4f} <= {rho_mid:.4f}; euler {ratio_eul:.4f} <= {rho_eul:.4f}; "
            f"err mid {res['midpoint']:.3e} (level {level_mid:.3e}), euler {res['euler']:.3e} (level {level_eul:.3e})")
    out["A"] = rowsA
    # ---------- (B) a posteriori bound for the piecewise-linear implicit-Euler periodic solution
    rowsB = []
    S = 32
    for K in (Ks if "B" in parts else ()):
        h = T / K
        xi = x0.copy()
        nper = int(math.ceil(math.log(1e3 / 1e-11) / (c * T))) + 2
        # periodic numerical solution: iterate the period map from the reference value
        X = reduced_onestep(m, xi, K, 12, "euler")
        for _ in range(60):
            Xn = reduced_onestep(m, X[-1], K, 1, "euler")
            if m.norm_P(Xn[-1] - X[-1]) < 1e-12: X = Xn; break
            X = Xn
        # now X has K+1 points on one period (t_0 = 0): periodic numerical solution
        D = 0.0; Eerr = 0.0; Xdmax = 0.0
        for k in range(K):
            xk, xk1 = X[k], X[k + 1]
            tk1 = (k + 1) * h
            fk1 = m.ftilde(xk1, tk1)                         # = (x_{k+1} - x_k)/h
            Xdmax = max(Xdmax, m.norm_P(fk1))
            for s in range(0, S + 1):
                th = s / S
                tau = k * h + th * h
                xh = (1 - th) * xk + th * xk1
                d = fk1 - m.ftilde(xh, tau)
                D = max(D, m.norm_P(d))
                Eerr = max(Eerr, m.norm_P(xh - xs(tau)[0]))
        margin = 0.5 * (h / S) * (L * Xdmax + F)            # rigorous: |d'| <= L |xhat'| + F between samples
        D_cert = D + margin
        out_B = dict(K=K, h=h, true_err=Eerr, post_bound=D_cert / c, post_bound_sample=D / c,
                     prior_level=h * M2 / (2 * c), effectivity=D_cert / c / Eerr)
        # shooting stopping criterion
        rowsB.append(out_B)
        log(f"  B K={K}: true {Eerr:.3e}  a posteriori {D_cert / c:.3e} (eff {out_B['effectivity']:.2f})  a priori(M_emp) {h * M2 / (2 * c):.3e}")
    out["B"] = rowsB
    # shooting bound for the numerical period map (K = 32)
    if "B" not in parts:
        return out
    K = 32 if 32 in Ks else Ks[len(Ks) // 2]
    h = T / K; rho = 1 / (1 + h * c)
    Xs = reduced_onestep(m, x0 + 0.5, K, 1, "euler")
    xi = x0 + rng.uniform(-1, 1, n)
    Xf = reduced_onestep(m, x0, K, 1, "euler")
    for _ in range(80):
        Xf = reduced_onestep(m, Xf[-1], K, 1, "euler")
    xih = Xf[-1]
    rows_sh = []
    for j in range(1, 9):
        Xn = reduced_onestep(m, xi, K, 1, "euler")
        res = m.norm_P(Xn[-1] - xi)
        rows_sh.append(dict(j=j, true=m.norm_P(xi - xih), bound=res / (1 - rho ** K), resid=res))
        xi = Xn[-1]
    out["shoot"] = dict(K=K, rows=rows_sh)
    log("  shooting: " + ", ".join(f"j{r['j']}: {r['true']:.2e}<={r['bound']:.2e}" for r in rows_sh))
    return out

if __name__ == "__main__":
    t0 = time.time()
    lf = open("results/strengthen_log_%s_%s.txt" % (sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "AB"), "w")
    def log(s):
        s = f"[{time.time() - t0:6.0f}s] {s}"; print(s, flush=True); lf.write(s + "\n"); lf.flush()
    M = models()
    parts = sys.argv[2] if len(sys.argv) > 2 else "AB"
    k = sys.argv[1]
    Ks = (4, 8, 16, 32, 64) if k != "4" else (8, 16, 32, 64)
    res = run_example(k, M[k], Ks, log, parts)
    json.dump(res, open("results/strengthen_ex%s_%s.json" % (k, parts), "w"), indent=1)
    log("STRENGTHEN DONE")
