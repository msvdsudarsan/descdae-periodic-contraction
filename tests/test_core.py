import numpy as np
from descdae import RCLadder, implicit_euler, reference_periodic, sampled_contraction_margin


def test_algebraic_solve_residual():
    m = RCLadder(N=6)
    x = np.random.RandomState(1).uniform(-5, 5, 6)
    assert np.max(np.abs(m.f2(x, m.h(x), 0.0))) < 1e-13


def test_schur_matches_finite_difference():
    m = RCLadder(N=4)
    x = np.random.RandomState(2).uniform(-3, 3, 4)
    S = m.schur(x, 0.3)
    eps = 1e-6
    Sfd = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4); e[j] = eps
        Sfd[:, j] = (m.ftilde(x + e, 0.3) - m.ftilde(x - e, 0.3)) / (2 * eps)
    assert np.max(np.abs(S - Sfd)) < 1e-6


def test_contraction_margin_below_minus_c():
    m = RCLadder(N=6)
    assert sampled_contraction_margin(m, 300) <= -m.contraction_rate() + 1e-10


def test_implicit_euler_converges_to_reference():
    m = RCLadder(N=1, c0=4.0, cvar=0.0, V=5.0)
    xs = reference_periodic(m)
    t, X1, X2, _ = implicit_euler(m, np.array([3.0]), 400, 12)
    assert abs(X1[-1, 0] - xs(t[-1])[0, 0]) < 5e-3


def test_coupled_model_schur_and_residual():
    import numpy as np
    from descdae import CoupledRCLadder
    m = CoupledRCLadder(gc=1.5, N=4, c0=1.0, cvar=0.3)
    x = np.random.RandomState(3).uniform(-9, 9, 4)
    w = m.h(x, 0.0)
    assert np.abs(m.f2(x, w, 0.0)).max() < 1e-12
    S = m.schur(x, 0.0); e = 1e-6
    Sfd = np.array([(m.ftilde(x + e * v, 0) - m.ftilde(x - e * v, 0)) / (2 * e) for v in np.eye(4)]).T
    assert np.abs(S - Sfd).max() < 1e-6
    # worst-case margin from the closed form is an upper bound for random states
    Cs = np.diag(m.C ** 0.5); Ci = np.diag(m.C ** -0.5)
    mu = np.linalg.eigvalsh(0.5 * (Cs @ S @ Ci + (Cs @ S @ Ci).T)).max()
    assert mu <= -m.contraction_rate() + 1e-9


def test_midpoint_contraction_and_mesh():
    import numpy as np
    from descdae import RCMesh
    from descdae.solvers import reduced_onestep
    m = RCMesh(nx=4, ny=4, n_loads=2)
    assert m.F_cert() > 0 and m.LS_cert() > 0
    # two midpoint solutions contract per step at least by rho_h of the theorem (P = C here)
    rs = np.random.RandomState(4)
    a, b = rs.uniform(-2, 2, 16), rs.uniform(-2, 2, 16)
    c, L, K = m.contraction_rate(), m.LS_cert(), 16
    h = m.T / K
    g = h * c / (1 + h * h * L * L / 4); rho = np.sqrt((1 - g) / (1 + g))
    A = reduced_onestep(m, a, K, 1, "midpoint"); B = reduced_onestep(m, b, K, 1, "midpoint")
    d0 = np.sqrt(np.sum(m.C * (a - b) ** 2)); d1 = np.sqrt(np.sum(m.C * (A[1] - B[1]) ** 2))
    assert d1 <= rho * d0 * (1 + 1e-9)


def _step(m, x, t, h, method):
    from scipy.optimize import fsolve
    if method == "euler":
        return fsolve(lambda y: y - x - h * m.ftilde(y, t + h), x, xtol=1e-12)
    return fsolve(lambda y: y - x - h * m.ftilde(0.5 * (x + y), t + h / 2), x, xtol=1e-12)


def test_midpoint_rho_h_random_instances():
    """Per-step contraction of two midpoint solutions, random ladders, states and step sizes (incl. large h)."""
    from descdae import RCLadder
    rs = np.random.RandomState(7)
    for trial in range(12):
        m = RCLadder(N=int(rs.randint(2, 7)), c0=rs.uniform(0.5, 3), cvar=0.3, V=rs.uniform(1, 6))
        c = m.contraction_rate()
        Ci = np.diag(m.C ** -0.5)
        L = np.linalg.eigvalsh(Ci @ (m.K + np.eye(m.N) / m.Ra) @ Ci).max()
        h = 10 ** rs.uniform(-2, 1)
        g = h * c / (1 + h * h * L * L / 4); rho = np.sqrt((1 - g) / (1 + g))
        a, b = rs.uniform(-4, 4, m.N), rs.uniform(-4, 4, m.N)
        t = rs.uniform(0, m.T)
        A, B = _step(m, a, t, h, "midpoint"), _step(m, b, t, h, "midpoint")
        n = lambda v: np.sqrt(np.sum(m.C * v ** 2))
        assert rho < 1
        assert n(A - B) <= rho * n(a - b) * (1 + 1e-8)


def test_variable_step_euler_contraction_and_level():
    """Corollary (variable steps): per-step factor 1/(1+h_n c) for two solutions with random steps."""
    from descdae import RCLadder
    rs = np.random.RandomState(8)
    m = RCLadder(N=3, c0=1.0, cvar=0.3, V=4.0)
    c = m.contraction_rate()
    a, b = rs.uniform(-3, 3, 3), rs.uniform(-3, 3, 3)
    t = 0.0
    n = lambda v: np.sqrt(np.sum(m.C * v ** 2))
    for _ in range(40):
        h = rs.uniform(0.02, 1.5)
        d0 = n(a - b)
        a, b = _step(m, a, t, h, "euler"), _step(m, b, t, h, "euler")
        t += h
        assert n(a - b) <= d0 / (1 + h * c) * (1 + 1e-8)


def test_end_to_end_certificate_holds():
    """Corollary (end-to-end certificate): B = eps/(1-e^{-cT}) + D/c bounds the grid error of a non-periodic iterate."""
    import os, sys, math
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import certified as ce
    from strengthen import LS_cert, F_cert
    from descdae import RCLadder, reference_periodic
    m = RCLadder(N=3, c0=1.0, cvar=0.3, V=4.0)
    c, L, F = m.contraction_rate(), LS_cert(m), F_cert(m)
    xs = reference_periodic(m)
    xi = np.array([2.0, -1.5, 1.0])
    for K in (8, 16):
        X = ce.run_period(m, xi, K)
        eps = m.norm_P(X[-1] - xi); D = ce.defect(m, X, K, L, F)
        B = eps / (1 - math.exp(-c * m.T)) + D / c
        assert ce.true_grid_error(m, X, K, xs) <= B


def test_midpoint_defect_bound_is_rigorous_and_L2_valid():
    """Second-order certificate: the sampled defect bound dominates the true sup of the Hermite residual, and L2 dominates f_xx."""
    import os, sys, math
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import certified_mid as cm
    from strengthen import LS_cert
    from descdae import RCLadder, RCMesh
    rs = np.random.RandomState(3)
    for m in (RCLadder(N=3, V=4.0), RCMesh(nx=3, ny=3, n_loads=1)):
        L2 = cm.L2_cert(m)
        for _ in range(10):
            x = rs.uniform(-8, 8, m.n); u = rs.randn(m.n); v = rs.randn(m.n); e = 1e-3
            f = lambda z: m.ftilde(z, 0.3)
            d2 = (f(x + e*u + e*v) - f(x + e*u - e*v) - f(x - e*u + e*v) + f(x - e*u - e*v)) / (4*e*e)
            assert m.norm_P(d2) <= L2 * m.norm_P(u) * m.norm_P(v)
        K = 16; h = m.T / K
        X = cm.run_period_mid(m, np.zeros(m.n), K)
        D = cm.defect_mid(m, X, K, LS_cert(m), L2, cm.F2_cert(m))
        dn = [m.ftilde(X[k], k*h) for k in range(K + 1)]
        sup = 0.0
        for k in range(K):
            for th in np.linspace(0, 1, 41):
                x, x1, _ = cm.hermite(X[k], X[k+1], dn[k], dn[k+1], h, th)
                sup = max(sup, m.norm_P(x1 - m.ftilde(x, (k + th) * h)))
        assert sup <= D <= 3 * sup
