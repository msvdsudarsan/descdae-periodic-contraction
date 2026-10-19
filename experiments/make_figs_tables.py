"""Build the constants table, the certified-vs-empirical table, Example 3 table and the two new figures."""
import sys, json, csv, os
sys.path.insert(0, "src")
import numpy as np
from scipy.linalg import eigh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from descdae import RCLadder, CoupledRCLadder
R = "results/"
S = json.load(open(R + "summary.json")); E = json.load(open(R + "extra.json")); CP = json.load(open(R + "coupled.json"))
RS = json.load(open(R + "residual.json"))
conv = list(csv.DictReader(open(R + "convergence.csv")))
models = {
    "1": RCLadder(N=1, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=5.0, omega=2.0, c0=4.0, cvar=0.0),
    "2": RCLadder(N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
    "3": CoupledRCLadder(gc=1.5, N=6, Rc=1.0, Rs=1.0, Ra=1.0, Rg=2.0, a=1.0, b=2.0, V=4.0, omega=2.0, c0=1.0, cvar=0.3),
}
info = {}
for k, m in models.items():
    if k in ("1", "2"):
        c, M, Mc = S["ex" + k]["c"], S["ex" + k]["M"], E["res"]["Mcert" + k]
    else:
        c, M, Mc = CP["c"], CP["M"], CP["Mcert"]
    Cm = np.diag(m.C ** -0.5)
    LS = float(eigh(Cm @ m.Lam @ Cm, eigvals_only=True)[-1])
    F = m.V * m.omega / (m.Rs * np.sqrt(m.C[0]))
    kappa = 1.0 / (m.Ra * np.sqrt(m.C.min()))
    info[k] = dict(c=c, M=M, Mcert=Mc, LS=LS, F=F, kappa=kappa, m=m.m, L22=m.L22, L21=m.L21, L12=1.0 / (m.Ra * m.C.min()), T=m.T)
    assert abs(F * (1 + LS / c) - Mc) < 1e-8 * Mc
json.dump(info, open(R + "constants.json", "w"), indent=1)
os.makedirs(R + "tables", exist_ok=True)
def tx(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

with open(R + "tables/constants.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nEx. & $m$ & $L_{22}$ & $\\kappa$ & $c$ & $F$ & $L_S$ & $M_{\\rm emp}$ & $M_{\\rm cert}$ & $M_{\\rm cert}/M_{\\rm emp}$\\\\\n\\midrule\n")
    for k, d in info.items():
        f.write(f"{k} & {d['m']:.2f} & {d['L22']:.2f} & {d['kappa']:.3f} & {d['c']:.4f} & {d['F']:.3f} & {d['LS']:.3f} & {d['M']:.3f} & {d['Mcert']:.1f} & {d['Mcert']/d['M']:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
# error table with both bounds (K = 64)
rows = {}
for k in "12":
    r = [x for x in conv if x["example"] == "ex" + k and x["K"] == "64"][0]
    rows[k] = (float(r["h"]), float(r["sup_e1"]))
r3 = [x for x in CP["rows"] if x["K"] == 64][0]; rows["3"] = (r3["h"], r3["sup"])
with open(R + "tables/bounds_both.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\nEx. & observed error & bound with $M_{\\rm emp}$ & ratio & bound with $M_{\\rm cert}$ & ratio\\\\\n\\midrule\n")
    for k, (h, sup) in rows.items():
        d = info[k]; b1 = h * d["M"] / (2 * d["c"]); b2 = h * d["Mcert"] / (2 * d["c"])
        f.write(f"{k} & {sup:.3e} & {b1:.3e} & {sup/b1:.2f} & {b2:.3e} & {sup/b2:.3f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
with open(R + "tables/conv_ex3.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$K$ & $h$ & $\\sup|e^1_n|_P$ & bound $hM/(2c)$ & ratio & order\\\\\n\\midrule\n")
    prev = None
    for r in CP["rows"]:
        o = "--" if prev is None else f"{np.log(prev['sup']/r['sup'])/np.log(prev['h']/r['h']):.2f}"
        f.write(f"{r['K']} & {tx(r['h'])} & {tx(r['sup'])} & {tx(r['bound'])} & {r['ratio']:.2f} & {o}\\\\\n")
        prev = r
    f.write("\\bottomrule\n\\end{tabular}")
# residual table (worst over seeds, adversarial) for d in {1e-4,1e-2,1e-1}
with open(R + "tables/residual.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\nEx. & $\\bar\\delta$ & random (3 seeds) & bias & adversarial & bound~\\eqref{eq:inexact}\\\\\n\\midrule\n")
    for k in "12":
        for d in (1e-5, 1e-3, 1e-1):
            g = {m: [x for x in RS if x["ex"] == k and x["mode"] == m and abs(x["d"] - d) < 1e-12][0] for m in ("random", "bias", "adv")}
            f.write(f"{k} & $10^{{{int(round(np.log10(d)))}}}$ & {tx(g['random']['sup'])} & {tx(g['bias']['sup'])} & {tx(g['adv']['sup'])} & {tx(g['adv']['bound'])}\\\\\n".replace("e-0","\\cdot10^{-").replace("$\\cdot","$1\\cdot") if False else f"{k} & $10^{{{int(round(np.log10(d)))}}}$ & {tx(g['random']['sup'])} & {tx(g['bias']['sup'])} & {tx(g['adv']['sup'])} & {tx(g['adv']['bound'])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
# ---- figures
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
for a, k in zip(ax, "123"):
    if k in "12":
        rr = [x for x in conv if x["example"] == "ex" + k]
        h = np.array([float(x["h"]) for x in rr]); e = np.array([float(x["sup_e1"]) for x in rr])
    else:
        h = np.array([x["h"] for x in CP["rows"]]); e = np.array([x["sup"] for x in CP["rows"]])
    d = info[k]
    a.loglog(h, e, "o-", label="observed error")
    a.loglog(h, h * d["M"] / (2 * d["c"]), "k--", label=r"$hM_{\mathrm{emp}}/(2c)$")
    a.loglog(h, h * d["Mcert"] / (2 * d["c"]), "k:", label=r"$hM_{\mathrm{cert}}/(2c)$")
    a.set_title(f"Example {k}"); a.set_xlabel("step size $h$"); a.minorticks_off()
ax[0].set_ylabel(r"$\sup|e_n|_P$"); ax[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("figures/figA_error_both.pdf"); fig.savefig("figures/figA_error_both.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
for a, k in zip(ax, "12"):
    ds = sorted(set(x["d"] for x in RS if x["ex"] == k))
    for mode, mk in (("random", "o-"), ("bias", "s-"), ("adv", "^-")):
        a.loglog(ds, [max(x["sup"] for x in RS if x["ex"] == k and x["mode"] == mode and x["d"] == d) for d in ds], mk,
                 label={"random": "random signs", "bias": "fixed direction", "adv": "error-aligned"}[mode])
    a.loglog(ds, [[x for x in RS if x["ex"] == k and x["d"] == d][0]["bound"] for d in ds], "k--", label="bound (Prop.)")
    tol = [x for x in RS if x["ex"] == k][0]["tol_rule"]
    a.axvline(tol, color="gray", ls=":", lw=1); a.text(tol, a.get_ylim()[0] * 1.5, r" $hM/(2\kappa)$", fontsize=7)
    a.set_title(f"Example {k}"); a.set_xlabel(r"perturbation norm $\bar\delta$")
ax[0].set_ylabel(r"$\sup|e_n|_P$ (last 3 periods)"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/figB_residual.pdf"); fig.savefig("figures/figB_residual.png", dpi=150); plt.close(fig)
print("TABLES/FIGS DONE")
# ---- macros used in the text
mx = lambda mode: max(x["ratio"] for x in RS if x["mode"] == mode)
with open(R + "tables/numbers_main.tex", "w") as f:
    W = lambda n, v: f.write("\\newcommand{\\%s}{%s}\n" % (n, v))
    W("cThree", f"{CP['c']:.4f}"); W("MThree", f"{CP['M']:.3f}"); W("McertThree", f"{CP['Mcert']:.1f}")
    W("cobsThree", f"{CP['c_obs']:.2f}"); W("marginThree", f"{CP['margin']:.4f}"); W("nperThree", str(CP["nper"]))
    W("maxRatioThree", f"{max(r['ratio'] for r in CP['rows']):.2f}")
    W("ratioRandMax", f"{mx('random'):.2f}"); W("ratioBiasMax", f"{mx('bias'):.2f}"); W("ratioAdvMax", f"{mx('adv'):.2f}")
    W("MratioOne", f"{info['1']['Mcert']/info['1']['M']:.1f}"); W("MratioTwo", f"{info['2']['Mcert']/info['2']['M']:.1f}")
    W("MratioThree", f"{info['3']['Mcert']/info['3']['M']:.1f}")
    W("McertOne", f"{info['1']['Mcert']:.1f}"); W("McertTwo", f"{info['2']['Mcert']:.1f}")
    W("kappaTwo", f"{info['2']['kappa']:.3f}")
    ks = [x for x in RS if x["ex"] == "2"][0]["tol_rule"]; W("tolRuleTwo", f"{ks:.3f}")
    ks = [x for x in RS if x["ex"] == "1"][0]["tol_rule"]; W("tolRuleOne", f"{ks:.3f}")
print("MACROS DONE")
