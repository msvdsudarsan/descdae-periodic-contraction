"""Reproduces every number, table and figure of the paper. Run: python experiments/run_all.py"""
import csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from descdae import (RCLadder, implicit_euler, reference_periodic, numeric_poincare,
                     second_derivative_bound, observed_rate, sampled_contraction_margin)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, TAB, FIG = (os.path.join(ROOT, d) for d in ("results", "results/tables", "figures"))
for d in (RES, TAB, FIG):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "lines.linewidth": 1.3,
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "font.family": "serif"})

def sci(x, d=2):
    if x == 0 or not np.isfinite(x):
        return "0" if x == 0 else "--"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.{d}f}\\times10^{{{e}}}$"

MODELS = {
    "ex1": RCLadder(N=1, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=5.0, omega=2.0, c0=4.0, cvar=0.0),
    "ex2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
}
KS = [16, 32, 64, 128, 256, 512]
summary, conv_rows, poin_rows = {}, [], []
refs = {}
t0 = time.time()
for key, mod in MODELS.items():
    xs = reference_periodic(mod); refs[key] = xs
    c = mod.contraction_rate(); T = mod.T; lh = mod.lipschitz_h_P()
    M = second_derivative_bound(mod, xs)
    margin = sampled_contraction_margin(mod, 1500)
    x10 = np.random.RandomState(7).uniform(-3, 3, mod.n)
    e0 = mod.norm_P(x10 - xs(0.0)[0])
    nper = int(np.ceil(np.log(max(e0, 1.0) / 1e-12) / (c * T))) + 3
    summary[key] = dict(N=mod.N, T=T, c=c, c_block=mod.block_bound_rate(), M=M, margin=margin, e0=e0,
                        n_periods=nper, ell_h=lh, m=mod.m, L22=mod.L22)
    for K in KS:
        t, X1, X2, info = implicit_euler(mod, x10, K, nper)
        Xs = xs(t)
        E1 = np.array([mod.norm_P(r) for r in (X1 - Xs)])
        Xs2 = np.array([mod.h(r, 0.0) for r in Xs])
        E2 = np.linalg.norm(X2 - Xs2, axis=1)
        win = len(t) - 3 * K
        h = T / K
        bound = h * M / (2 * c)
        sel = E1 > 1e-8
        conv_rows.append(dict(example=key, K=K, h=h, sup_e1=float(E1[win:].max()), bound=bound,
                              ratio=float(E1[win:].max() / bound), sup_e1_all=float(E1.max()),
                              global_bound=max(e0, bound), x2_over_x1=float((E2[sel] / E1[sel]).max()),
                              ell_h=lh, sup_e2=float(E2[win:].max()), max_newton=info["max_newton_iters"],
                              max_alg_res=info["max_alg_residual"]))
        print(key, K, conv_rows[-1]["sup_e1"], bound, f"{time.time()-t0:.0f}s", flush=True)
    for K in (16, 64, 256):
        h = T / K; rho = 1.0 / (1.0 + h * c); rhoK = rho ** K
        xi = np.zeros(mod.n); incs = []
        for j in range(200):
            xn = numeric_poincare(mod, xi, K); inc = mod.norm_P(xn - xi); incs.append(inc); xi = xn
            if inc < 1e-11:
                break
        pred = int(np.ceil(np.log(incs[0] / 1e-11) / (-K * np.log(rho))))
        poin_rows.append(dict(example=key, K=K, rho_K=rhoK, iters_obs=len(incs), iters_pred=pred,
                              dist_to_exact=mod.norm_P(xi - xs(0.0)[0]), bound=h * M / (2 * c),
                              ratio_first=float(incs[1] / incs[0]) if len(incs) > 1 else float("nan")))
        print("poincare", key, K, poin_rows[-1], flush=True)

# --- observed contraction rates (reduced ODE, high accuracy) and scaling in N
rate_rows = []
for N in (1, 3, 6, 10, 20, 40):
    m = RCLadder(N=N, c0=1.0, cvar=0.3)
    ts, D, rate = observed_rate(m)
    c = m.contraction_rate()
    ok = bool(np.all(D <= np.exp(-c * ts) * D[0] * (1 + 1e-6) + 1e-9))
    rate_rows.append(dict(N=N, c_theory=c, c_block=m.block_bound_rate(), c_obs=rate, bound_holds=ok,
                          margin=sampled_contraction_margin(m, 400)))
    print("rate", rate_rows[-1], flush=True)
dist = {}
for key, mod in MODELS.items():
    dist[key] = observed_rate(mod)

# --- save raw results
def dump(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
dump("convergence.csv", conv_rows); dump("poincare.csv", poin_rows); dump("rates.csv", rate_rows)
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2)

# --- LaTeX tables and macros
def conv_table(key):
    rows = [r for r in conv_rows if r["example"] == key]
    out = ["\\begin{tabular}{rrrrrr}", "\\toprule",
           "$K$ & $h$ & $\\sup|e^1_n|_P$ & bound $hM/(2c)$ & ratio & order\\\\", "\\midrule"]
    for i, r in enumerate(rows):
        order = "--" if i == 0 else f"{np.log2(rows[i-1]['sup_e1']/r['sup_e1']):.2f}"
        out.append(f"{r['K']} & {sci(r['h'],2)} & {sci(r['sup_e1'],2)} & {sci(r['bound'],2)} & {r['ratio']:.2f} & {order}\\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)
open(os.path.join(TAB, "conv_ex1.tex"), "w").write(conv_table("ex1"))
open(os.path.join(TAB, "conv_ex2.tex"), "w").write(conv_table("ex2"))
out = ["\\begin{tabular}{rrrrr}", "\\toprule", "$N$ & $c$ (Thm.~\\ref{thm:contraction}) & $c_{\\rm blk}$ (Cor.~\\ref{cor:blocks}) & $c_{\\rm obs}$ & bound holds\\\\", "\\midrule"]
for r in rate_rows:
    out.append(f"{r['N']} & {r['c_theory']:.4f} & {r['c_block']:.4f} & {r['c_obs']:.4f} & {'yes' if r['bound_holds'] else 'no'}\\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "rates.tex"), "w").write("\n".join(out))
out = ["\\begin{tabular}{llrrrr}", "\\toprule", "Example & $K$ & $\\rho_K$ & iterations (obs.) & iterations (pred.) & $|\\xi_h-x^\\ast(0)|_P$\\\\", "\\midrule"]
for r in poin_rows:
    out.append(f"{'1' if r['example']=='ex1' else '2'} & {r['K']} & {sci(r['rho_K'],2)} & {r['iters_obs']} & {r['iters_pred']} & {sci(r['dist_to_exact'],2)}\\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "poincare.tex"), "w").write("\n".join(out))
s1, s2 = summary["ex1"], summary["ex2"]
mac = {"cOne": f"{s1['c']:.4f}", "cTwo": f"{s2['c']:.4f}", "MOne": f"{s1['M']:.3f}", "MTwo": f"{s2['M']:.3f}",
       "marginOne": f"{s1['margin']:.4f}", "marginTwo": f"{s2['margin']:.4f}",
       "ellOne": f"{s1['ell_h']:.3f}", "ellTwo": f"{s2['ell_h']:.3f}",
       "TperiodOne": f"{s1['T']:.4f}", "nperOne": str(s1['n_periods']), "nperTwo": str(s2['n_periods']),
       "maxRatioOne": f"{max(r['ratio'] for r in conv_rows if r['example']=='ex1'):.2f}",
       "maxRatioTwo": f"{max(r['ratio'] for r in conv_rows if r['example']=='ex2'):.2f}",
       "maxResidual": sci(max(r['max_alg_res'] for r in conv_rows), 1),
       "maxNewton": str(max(r['max_newton'] for r in conv_rows)),
       "xtwoRatioMax": f"{max(r['x2_over_x1']/r['ell_h'] for r in conv_rows):.2f}",
       "globalOK": "all" if all(r['sup_e1_all'] <= r['global_bound'] * (1 + 1e-9) for r in conv_rows) else "not all",
       "cMinOverN": f"{min(r['c_theory'] for r in rate_rows):.4f}"}
open(os.path.join(TAB, "numbers.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")

# --- figures
mod1, mod2 = MODELS["ex1"], MODELS["ex2"]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
xs1 = refs["ex1"]; tt = np.linspace(0, 6 * mod1.T, 600)
for x0, col in zip((-3.0, 0.0, 3.0), ("C0", "C1", "C2")):
    t, X1, X2, _ = implicit_euler(mod1, np.array([x0]), 200, 6)
    ax[0].plot(t, X1[:, 0], color=col, label=f"$x^1(0)={x0:g}$")
    ax[2].plot(X1[:, 0], X2[:, 0], color=col, lw=0.9)
ax[0].plot(tt, xs1(tt)[:, 0], "k--", label="periodic solution")
ax[0].set_xlabel("time $t$ (normalised units)"); ax[0].set_ylabel("$x^1$ (node voltage, V)"); ax[0].legend(loc="lower right", fontsize=6.5)
for key, col in (("ex1", "C3"), ("ex2", "C4")):
    ts, D, _ = dist[key]; c = summary[key]["c"]
    ax[1].semilogy(ts, D / D[0], color=col, label=f"Example {key[-1]}")
    ax[1].semilogy(ts, np.exp(-c * ts), "--", color=col, lw=1.0)
ax[1].set_ylim(1e-9, 2); ax[1].set_xlim(0, 30)
ax[1].set_xlabel("time $t$ (normalised units)"); ax[1].set_ylabel("$|x_a-x_b|_P/|x_a-x_b|_P(0)$"); ax[1].legend(loc="upper right")
xg = np.linspace(-8, 8, 200); ax[2].plot(xg, [mod1.h(np.array([v]))[0] for v in xg], "k--", lw=1.0, label="$x^2=h(x^1)$")
ax[2].plot(xs1(tt)[:, 0], [mod1.h(r)[0] for r in xs1(tt)], "k", lw=2, alpha=0.5, label="periodic orbit")
ax[2].set_xlabel("$x^1$ (V)"); ax[2].set_ylabel("$x^2$ (V)"); ax[2].legend(loc="upper left", fontsize=6.5)
for a, l in zip(ax, "abc"):
    a.text(0.02, 0.92, f"({l})", transform=a.transAxes, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_contraction.pdf")); fig.savefig(os.path.join(FIG, "fig1_contraction.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for a, key in zip(ax, ("ex1", "ex2")):
    rows = [r for r in conv_rows if r["example"] == key]
    h = np.array([r["h"] for r in rows])
    a.loglog(h, [r["sup_e1"] for r in rows], "o-", label=r"$\sup|e^1_n|_P$")
    a.loglog(h, [r["sup_e2"] for r in rows], "s-", label=r"$\sup\|e^2_n\|_2$")
    a.loglog(h, [r["bound"] for r in rows], "k--", label=r"bound $hM/(2c)$")
    a.loglog(h, rows[0]["sup_e1"] * h / h[0], ":", color="gray", label="slope 1")
    a.set_xlabel("step size $h$ (normalised units)"); a.set_ylabel("error (V)")
    a.set_title(f"Example {key[-1]}", fontsize=9); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_error_vs_h.pdf")); fig.savefig(os.path.join(FIG, "fig2_error_vs_h.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
N = [r["N"] for r in rate_rows]
ax[0].semilogx(N, [r["c_theory"] for r in rate_rows], "o-", label="$c$ (Theorem)")
ax[0].semilogx(N, [r["c_block"] for r in rate_rows], "^--", label="$c_{\\rm blk}$ (Corollary)")
ax[0].semilogx(N, [r["c_obs"] for r in rate_rows], "s-", label="observed")
ax[0].set_xlabel("number of stages $N$"); ax[0].set_ylabel("contraction rate (1/time)"); ax[0].legend(fontsize=7)
for key, mk in (("ex1", "o"), ("ex2", "s")):
    rows = [r for r in poin_rows if r["example"] == key]
    ax[1].semilogy([r["K"] for r in rows], [r["rho_K"] for r in rows], mk + "-", label=f"Example {key[-1]}")
ax[1].set_xlabel("steps per period $K$"); ax[1].set_ylabel("period-map factor $\\rho^K$"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_scaling.pdf")); fig.savefig(os.path.join(FIG, "fig3_scaling.png")); plt.close(fig)
print("ALL DONE", f"{time.time()-t0:.0f}s")
