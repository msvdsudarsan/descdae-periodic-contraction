from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def implicit_euler(model, x10, K, n_periods, t0=0.0, tol=1e-13, maxit=40):
    """Implicit Euler for the full DAE with Newton on (x1, x2), step h = T/K.

    Initial algebraic variable is the consistent value h(x10). Returns times, x1, x2 and diagnostics.
    """
    n, q, h = model.n, model.q, model.T / K
    nsteps = K * n_periods
    t = t0 + h * np.arange(nsteps + 1)
    X1 = np.zeros((nsteps + 1, n))
    X2 = np.zeros((nsteps + 1, q))
    X1[0] = x10
    X2[0] = model.h(x10, t0)
    max_it, max_res = 0, 0.0
    I = np.eye(n)
    for i in range(nsteps):
        z1, z2 = X1[i].copy(), X2[i].copy()
        for it in range(maxit):
            R = np.concatenate([z1 - X1[i] - h * model.f1(z1, z2, t[i + 1]), model.f2(z1, z2, t[i + 1])])
            if np.max(np.abs(R)) < tol:
                break
            G11, G12, G21, G22 = model.jac(z1, z2, t[i + 1])
            J = np.block([[I - h * G11, -h * G12], [G21, G22]])
            dz = np.linalg.solve(J, -R)
            z1 += dz[:n]
            z2 += dz[n:]
        else:
            raise RuntimeError("Newton did not converge")
        X1[i + 1], X2[i + 1] = z1, z2
        max_it = max(max_it, it)
        max_res = max(max_res, float(np.max(np.abs(model.f2(z1, z2, t[i + 1])))))
    return t, X1, X2, {"max_newton_iters": max_it, "max_alg_residual": max_res}


def numeric_poincare(model, xi, K):
    """One period (K implicit-Euler steps) map x1(0) -> x1(T)."""
    _, X1, _, _ = implicit_euler(model, xi, K, 1)
    return X1[-1]


def reference_periodic(model, tol=1e-9):
    """Accurate periodic steady state of the reduced ODE (Radau, rtol 1e-12)."""
    T, n = model.T, model.n
    rhs = lambda t, x: model.ftilde(x, t)
    jac = lambda t, x: model.schur(x, t)
    xi = np.zeros(n)
    for _ in range(400):
        sol = solve_ivp(rhs, (0, T), xi, method="Radau", rtol=1e-10, atol=1e-12, jac=jac)
        xn = sol.y[:, -1]
        done = model.norm_P(xn - xi) < tol
        xi = xn
        if done:
            break
    sol = solve_ivp(rhs, (0, T), xi, method="Radau", rtol=1e-10, atol=1e-12, dense_output=True, jac=jac)

    def xstar(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return sol.sol(np.mod(t, T)).T      # shape (len(t), n)

    xstar.xi = xi
    return xstar


def inexact_euler(model, x10, K, n_periods, d, xs=None, seed=11, mode="random"):
    """Implicit Euler on the reduced ODE with the algebraic variable perturbed by a vector of
    Euclidean norm d in every step: z = h(y) + delta.  mode: 'random' (random signs),
    'bias' (fixed direction), 'adv' (aligned with the current error, needs xs)."""
    from scipy.optimize import root
    rng = np.random.RandomState(seed)
    h = model.T / K
    x = np.array(x10, dtype=float)
    out = [x.copy()]
    N = model.N
    for n in range(n_periods * K):
        t1 = (n + 1) * h
        if d == 0.0:
            delta = np.zeros(N)
        elif mode == "random":
            delta = d * rng.choice([-1.0, 1.0], N) / np.sqrt(N)
        elif mode == "bias":
            delta = d * np.ones(N) / np.sqrt(N)
        else:
            e = x - xs(n * h)[0]
            nn = np.linalg.norm(e)
            delta = d * (e / nn if nn > 0 else np.ones(N) / np.sqrt(N))
        F = lambda y: y - x - h * model.f1(y, model.h(y, t1) + delta, t1)
        x = root(F, x, tol=1e-13).x
        out.append(x.copy())
    return np.array(out)


def reduced_onestep(model, x10, K, n_periods, method="euler"):
    """One-step methods applied to the reduced ODE x' = ftilde(x, t) (equivalent to the DAE for index 1).

    method: 'euler' (implicit Euler), 'midpoint' (implicit midpoint), 'trapezoid' (trapezoidal rule).
    Returns the array of x1 values at t_n = n h.
    """
    from scipy.optimize import root
    h = model.T / K
    f = model.ftilde
    x = np.array(x10, dtype=float)
    out = [x.copy()]
    for n in range(n_periods * K):
        t0, t1 = n * h, (n + 1) * h
        if method == "euler":
            F = lambda y: y - x - h * f(y, t1)
        elif method == "midpoint":
            F = lambda y: y - x - h * f(0.5 * (x + y), t0 + 0.5 * h)
        elif method == "trapezoid":
            f0 = f(x, t0)
            F = lambda y: y - x - 0.5 * h * (f0 + f(y, t1))
        else:
            raise ValueError(method)
        x = root(F, x, tol=1e-13).x
        out.append(x.copy())
    return np.array(out)
