"""Sensitivity of the midpoint certificate to the number J of defect samples per cell (Example 2).
The O(h^2) term of the defect bound is Lambda*delta^2/8 with delta = h/J, so J is a design choice."""
import sys, os, json, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from descdae import RCLadder, reference_periodic
from strengthen import LS_cert
import certified as ce
import certified_mid as cm

if __name__ == "__main__":
    m = ce.instrument(RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3))
    c = m.contraction_rate(); L = LS_cert(m); L2 = cm.L2_cert(m); F2 = cm.F2_cert(m); xs = reference_periodic(m)
    orig = cm.defect_mid; rows = []
    for J in (2, 4, 8):
        cm.defect_mid = functools.partial(orig, S=J)
        for tol in (1e-1, 1e-2):
            r = cm.certified_mid(m, tol, np.zeros(m.n), 8, c, L, L2, F2, xs)
            rows.append(dict(J=J, tol=tol, K=r["K"], steps=r["steps"], evals=r["evals"], bound=float(r["bound"]), true=float(r["true"])))
            print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "sampling_J_ex2.json"), "w"), indent=1)
