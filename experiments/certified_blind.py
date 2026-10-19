"""Example 6: a certificate with no reference solution in the loop.

A 200-node ladder with parameters not used in Examples 1-5.  The stopping decision uses only the certified bound B
(Corollary 12 with the Hermite defect of Proposition 13).  Afterwards, as a plausibility check that is NOT part of the
certificate, the periodic solution is recomputed with 4K midpoint steps and compared with the K-step grid values."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from descdae import RCLadder
from strengthen import LS_cert
import certified as ce
import certified_mid as cm

def run(N=200, tol=1e-2):
    m = ce.instrument(RCLadder(N=N, Rc=0.7, Rs=1.0, Ra=1.0, Rg=3.0, a=1.0, b=1.5, V=3.0, omega=3.0, c0=1.0, cvar=0.4))
    c = m.contraction_rate(); L = LS_cert(m); L2 = cm.L2_cert(m); F2 = cm.F2_cert(m)
    t0 = time.time()
    r = certified(m, tol, c, L, L2, F2)
    r.update(N=N, tol=tol, c=c, L=L, L2=L2, F2=F2, seconds=time.time() - t0)
    return r

def certified(m, tol, c, L, L2, F2):
    import math
    ce.STEPS = 0; ce.CNT[0] = 0; K = 8; xi = np.zeros(m.n); E = math.exp(-c * m.T); it = 0
    while True:
        X = cm.run_period_mid(m, xi, K); it += 1
        eps = m.norm_P(X[-1] - xi); D = cm.defect_mid(m, X, K, L, L2, F2); B = eps / (1 - E) + D / c
        if B <= tol:
            steps, evals = ce.STEPS, ce.CNT[0]
            # plausibility check, outside the certificate: 4K-step periodic solution
            xi4 = X[0].copy()
            for _ in range(400):
                X4 = cm.run_period_mid(m, xi4, 4 * K)
                if m.norm_P(X4[-1] - xi4) <= 1e-10: break
                xi4 = X4[-1]
            diff = max(m.norm_P(X[k] - X4[4 * k]) for k in range(K + 1))
            return dict(K=K, periods=it, steps=steps, evals=evals, bound=B, eps=eps, disc=D / c, check_4K=diff)
        if D / c > tol / 2: K *= 2
        xi = X[-1]
        if it > 400: raise RuntimeError("no convergence")

if __name__ == "__main__":
    out = [run(200, t) for t in (1e-1, 1e-2)]
    for r in out: print(r, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "blind_ex6.json"), "w"), indent=1)
