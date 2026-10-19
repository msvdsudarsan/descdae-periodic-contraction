"""Index-1 semilinear descriptor systems used in the paper.

The system has the semi-explicit form
    x1' = f1(x1, x2, t),      0 = f2(x1, x2, t),
with x1 in R^n (differential part), x2 in R^q (algebraic part), f T-periodic in t.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


class RCLadder:
    """RC ladder with N capacitor nodes v (= x1) and N algebraic nodes w (= x2).

        C v' = -Lam v + w/Ra + e1 u(t)/Rs
        0    = v/Ra - m w - a tanh(b w),            u(t) = V sin(omega t)

    where Lam = K + I/Ra, K = Laplacian(path)/Rc + e1 e1^T / Rs and m = 1/Ra + 1/Rg.
    N = 1 is the single-stage circuit (Example 1).
    """

    def __init__(self, N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0,
                 V=4.0, omega=2.0, c0=1.0, cvar=0.3):
        self.N, self.n, self.q = N, N, N
        self.Rc, self.Rs, self.Ra, self.Rg, self.a, self.b = Rc, Rs, Ra, Rg, a, b
        self.V, self.omega = V, omega
        self.T = 2.0 * np.pi / omega
        k = np.arange(1, N + 1)
        self.C = c0 + cvar * np.sin(k)
        lap = np.zeros((N, N))
        for i in range(N - 1):
            lap[i, i] += 1.0
            lap[i + 1, i + 1] += 1.0
            lap[i, i + 1] -= 1.0
            lap[i + 1, i] -= 1.0
        K = lap / Rc
        K[0, 0] += 1.0 / Rs
        self.K = K
        self.Lam = K + np.eye(N) / Ra
        self.m = 1.0 / Ra + 1.0 / Rg            # strong monotonicity constant of -f2 in x2
        self.L22 = self.m + a * b               # Lipschitz constant of f2 in x2
        self.L21 = 1.0 / Ra
        self.e1 = np.zeros(N)
        self.e1[0] = 1.0

    # ----- data ---------------------------------------------------------
    def u(self, t):
        return self.V * np.sin(self.omega * t)

    def f1(self, x1, x2, t):
        return (-self.Lam @ x1 + x2 / self.Ra + self.e1 * self.u(t) / self.Rs) / self.C

    def f2(self, x1, x2, t):
        return x1 / self.Ra - self.m * x2 - self.a * np.tanh(self.b * x2)

    def jac(self, x1, x2, t):
        N = self.N
        G11 = -self.Lam / self.C[:, None]
        G12 = np.diag(1.0 / (self.C * self.Ra))
        G21 = np.eye(N) / self.Ra
        G22 = -np.diag(self.m + self.a * self.b / np.cosh(self.b * x2) ** 2)
        return G11, G12, G21, G22

    # ----- algebraic solve (exact up to rounding): componentwise monotone equation
    def h(self, x1, t=0.0):
        rhs = np.asarray(x1, dtype=float) / self.Ra
        B = np.abs(rhs) / self.m + 1e-12
        lo, hi = -B, B
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            F = self.m * mid + self.a * np.tanh(self.b * mid) - rhs
            pos = F > 0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
        w = 0.5 * (lo + hi)
        for _ in range(4):
            F = self.m * w + self.a * np.tanh(self.b * w) - rhs
            Fp = self.m + self.a * self.b / np.cosh(self.b * w) ** 2
            w = w - F / Fp
        return w

    def ftilde(self, x1, t):
        return self.f1(x1, self.h(x1, t), t)

    def schur(self, x1, t):
        w = self.h(x1, t)
        G11, G12, G21, G22 = self.jac(x1, w, t)
        return G11 - G12 @ np.linalg.solve(G22, G21)

    # ----- theory constants ------------------------------------------------
    def norm_P(self, e):
        e = np.asarray(e, dtype=float)
        return float(np.sqrt(np.sum(self.C * e * e)))

    def contraction_rate(self):
        """c = lambda_min of (K + I/(Ra+Rg)) relative to diag(C): exact worst-case rate."""
        Kmat = self.K + np.eye(self.N) / (self.Ra + self.Rg)
        return float(eigh(Kmat, np.diag(self.C), eigvals_only=True)[0])

    def lipschitz_h_P(self):
        """Lipschitz constant of x1 -> h(x1) from the P-norm on x1 to the 2-norm on x2."""
        return 1.0 / (self.Ra * self.m * np.sqrt(self.C.min()))

    def block_bound_rate(self):
        """Cheap sufficient rate from Corollary: -mu_P(G11) - ||G12 G22^-1 G21||_P (Euclid. weighted)."""
        G11 = -self.Lam / self.C[:, None]
        Cs = np.sqrt(self.C)
        Gs = (Cs[:, None] * G11) / Cs[None, :]
        mu = float(np.linalg.eigvalsh(0.5 * (Gs + Gs.T)).max())
        coup = 1.0 / (self.Ra ** 2 * self.m) / self.C.min()
        return -mu - coup


class CoupledRCLadder(RCLadder):
    """RC ladder whose algebraic nodes are also coupled by resistors (non-diagonal G22).

        0 = v/Ra - (m I + Gc) w - a tanh(b w),     Gc = gc * Laplacian(path) >= 0.

    Then G22 = -(m I + Gc + diag(a b sech^2(b w))) is symmetric with G22 <= -m I, so the
    monotonicity assumption holds with the same m; the rate bound uses (m I + Gc)^{-1}.
    """

    def __init__(self, gc=1.5, **kw):
        super().__init__(**kw)
        N = self.N
        lap = np.zeros((N, N))
        for i in range(N - 1):
            lap[i, i] += 1.0; lap[i + 1, i + 1] += 1.0
            lap[i, i + 1] -= 1.0; lap[i + 1, i] -= 1.0
        self.Gc = gc * lap
        self.gc = gc
        self.L22 = self.m + self.a * self.b + float(np.linalg.eigvalsh(self.Gc).max())

    def f2(self, x1, x2, t):
        return x1 / self.Ra - self.m * x2 - self.Gc @ x2 - self.a * np.tanh(self.b * x2)

    def jac(self, x1, x2, t):
        G11, G12, G21, _ = super().jac(x1, x2, t)
        G22 = -(self.m * np.eye(self.N) + self.Gc + np.diag(self.a * self.b / np.cosh(self.b * x2) ** 2))
        return G11, G12, G21, G22

    def h(self, x1, t=0.0):
        """Damped Newton for the strongly monotone equation (m I + Gc) w + a tanh(b w) = v/Ra."""
        rhs = np.asarray(x1, dtype=float) / self.Ra
        A = self.m * np.eye(self.N) + self.Gc
        w = np.linalg.solve(A, rhs)
        F = lambda z: A @ z + self.a * np.tanh(self.b * z) - rhs
        for _ in range(100):
            Fw = F(w)
            nF = np.linalg.norm(Fw, np.inf)
            if nF < 1e-14 * (1.0 + np.linalg.norm(rhs, np.inf)):
                break
            J = A + np.diag(self.a * self.b / np.cosh(self.b * w) ** 2)
            step = np.linalg.solve(J, Fw)
            s = 1.0
            while np.linalg.norm(F(w - s * step), np.inf) > (1 - 0.25 * s) * nF and s > 1e-8:
                s *= 0.5
            w = w - s * step
        return w

    def contraction_rate(self):
        A = self.m * np.eye(self.N) + self.Gc
        M = self.Lam - np.linalg.inv(A) / self.Ra ** 2
        return float(eigh(0.5 * (M + M.T), np.diag(self.C), eigvals_only=True)[0])


class RCMesh(CoupledRCLadder):
    """Nonlinear RC mesh on an nx-by-ny grid (a model of an on-chip power-distribution network).

        C v' = -Lam v + w/Ra + s(t),    0 = v/Ra - (m I + Gc) w - a tanh(b w)

    with Lam = K + I/Ra, K = Laplacian(grid)/Rc + sum_p e_p e_p^T/Rs (supply pads p), Gc = gc * Laplacian(grid),
    and a T-periodic source vector s(t) = sum_j amp_j sin(omega t + phase_j) e_{node_j}  (supply and load currents).
    Only K >= 0 and Gc >= 0 are used by the theory, so the rate formula of the ladder carries over.
    """

    def __init__(self, nx=8, ny=8, Rc=1.0, Rs=0.5, Ra=1.0, Rg=2.0, a=1.0, b=2.0, gc=0.5, omega=2.0,
                 c0=1.0, cvar=0.3, n_loads=6, V=2.0, load_amp=1.5, seed=3):
        N = nx * ny
        self.N, self.n, self.q = N, N, N
        self.nx, self.ny = nx, ny
        self.Rc, self.Rs, self.Ra, self.Rg, self.a, self.b, self.gc = Rc, Rs, Ra, Rg, a, b, gc
        self.omega = omega
        self.T = 2.0 * np.pi / omega
        self.V = V
        idx = lambda i, j: i * ny + j
        lap = np.zeros((N, N))
        for i in range(nx):
            for j in range(ny):
                for di, dj in ((1, 0), (0, 1)):
                    ii, jj = i + di, j + dj
                    if ii < nx and jj < ny:
                        p, q_ = idx(i, j), idx(ii, jj)
                        lap[p, p] += 1.0; lap[q_, q_] += 1.0
                        lap[p, q_] -= 1.0; lap[q_, p] -= 1.0
        self.lap = lap
        k = np.arange(1, N + 1)
        self.C = c0 + cvar * np.sin(k)
        pads = [idx(0, 0), idx(0, ny - 1), idx(nx - 1, 0), idx(nx - 1, ny - 1)]
        K = lap / Rc
        for p in pads:
            K[p, p] += 1.0 / Rs
        self.K = K
        self.Lam = K + np.eye(N) / Ra
        self.m = 1.0 / Ra + 1.0 / Rg
        self.Gc = gc * lap
        self.L22 = self.m + a * b + float(np.linalg.eigvalsh(self.Gc).max())
        self.L21 = 1.0 / Ra
        rng = np.random.RandomState(seed)
        interior = [idx(i, j) for i in range(1, nx - 1) for j in range(1, ny - 1)]
        loads = list(rng.choice(interior, n_loads, replace=False))
        nodes = pads + loads
        self.src_nodes = np.array(nodes)
        self.src_amp = np.array([V / Rs] * len(pads) + [-load_amp] * n_loads)   # pads: supply through Rs; loads: sink currents
        self.src_phase = np.concatenate([np.zeros(len(pads)), rng.uniform(0, 2 * np.pi, n_loads)])

    def src(self, t):
        s = np.zeros(self.N)
        s[self.src_nodes] = self.src_amp * np.sin(self.omega * t + self.src_phase)
        return s

    def dsrc(self, t):
        s = np.zeros(self.N)
        s[self.src_nodes] = self.src_amp * self.omega * np.cos(self.omega * t + self.src_phase)
        return s

    def F_cert(self):
        """sup_t |C^{-1} ds/dt|_P = sup_t |C^{-1/2} s'(t)|, in closed form: with alpha_j = |amp_j| omega / sqrt(C_j) the square
        of the norm equals A + B cos(2 omega t + psi), A = sum alpha_j^2 / 2, B = |sum alpha_j^2 exp(2 i phi_j)| / 2."""
        al2 = (self.src_amp * self.omega) ** 2 / self.C[self.src_nodes]
        A = 0.5 * al2.sum()
        B = 0.5 * abs(np.sum(al2 * np.exp(2j * self.src_phase)))
        return float(np.sqrt(A + B))

    def f1(self, x1, x2, t):
        return (-self.Lam @ x1 + x2 / self.Ra + self.src(t)) / self.C

    def LS_cert(self):
        Cm = np.diag(self.C ** -0.5)
        return float(eigh(Cm @ self.Lam @ Cm, eigvals_only=True)[-1])


class RCGraph(RCMesh):
    """RC network on a random connected graph (Erdos-Renyi plus a spanning path), a structurally different test for the
    theory: only K >= 0, Gc >= 0 and diagonal C are used, so the rate formula of the ladder applies unchanged."""

    def __init__(self, N=40, p=0.08, Rc=1.0, Rs=0.5, Ra=1.0, Rg=2.0, a=1.0, b=2.0, gc=0.5, omega=2.0,
                 c0=1.0, cvar=0.3, n_pads=3, n_loads=6, V=2.0, load_amp=1.5, seed=11):
        rng = np.random.RandomState(seed)
        A = np.triu((rng.uniform(size=(N, N)) < p).astype(float), 1)
        perm = rng.permutation(N)
        for i in range(N - 1):
            a_, b_ = sorted((perm[i], perm[i + 1])); A[a_, b_] = 1.0
        A = A + A.T
        lap = np.diag(A.sum(1)) - A
        self.N, self.n, self.q = N, N, N
        self.Rc, self.Rs, self.Ra, self.Rg, self.a, self.b, self.gc = Rc, Rs, Ra, Rg, a, b, gc
        self.omega = omega; self.T = 2.0 * np.pi / omega; self.V = V
        self.lap = lap; self.adj = A
        self.C = c0 + cvar * np.sin(np.arange(1, N + 1))
        nodes = rng.permutation(N)
        pads = list(nodes[:n_pads]); loads = list(nodes[n_pads:n_pads + n_loads])
        K = lap / Rc
        for q_ in pads:
            K[q_, q_] += 1.0 / Rs
        self.K = K
        self.Lam = K + np.eye(N) / Ra
        self.m = 1.0 / Ra + 1.0 / Rg
        self.Gc = gc * lap
        self.L22 = self.m + a * b + float(np.linalg.eigvalsh(self.Gc).max())
        self.L21 = 1.0 / Ra
        self.src_nodes = np.array(pads + loads)
        self.src_amp = np.array([V / Rs] * len(pads) + [-load_amp] * n_loads)
        self.src_phase = np.concatenate([np.zeros(len(pads)), rng.uniform(0, 2 * np.pi, n_loads)])
