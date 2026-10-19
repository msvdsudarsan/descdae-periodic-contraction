"""Tables, figures and text macros for the midpoint, a posteriori and mesh results."""
import json, os, sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "results/"
def ld(k, parts):
    return json.load(open(R + f"strengthen_ex{k}_{parts}.json"))
D = {"1": ld(1, "AB"), "2": ld(2, "AB"), "3": ld(3, "AB")}
for _k in "123":   # fine-sampled (S=32) a posteriori part overrides the earlier run
    _b = ld(_k, "B"); D[_k]["B"] = _b["B"]; D[_k]["shoot"] = _b["shoot"]
A4, B4 = ld(4, "A"), ld(4, "B")
D["4"] = dict(A4); D["4"]["B"] = B4["B"]; D["4"]["shoot"] = B4["shoot"]
CONST = json.load(open(R + "constants.json"))
Mcert = {k: (CONST[k]["Mcert"] if k in CONST else None) for k in "123"}
Mcert["4"] = D["4"]["F"] * (1 + D["4"]["L"] / D["4"]["c"])
def tx(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
os.makedirs(R + "tables", exist_ok=True)
# ---- constants of the four examples
with open(R + "tables/mesh_constants.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nEx. & $n$ & $c$ & $L$ & $F$ & $M_{2,\\rm emp}$ & $M_{3,\\rm emp}$ & $M_{\\rm cert}$\\\\\n\\midrule\n")
    nn = {"1": 1, "2": 6, "3": 6, "4": 64}
    for k in "1234":
        d = D[k]
        f.write(f"{k} & {nn[k]} & {d['c']:.4f} & {d['L']:.3f} & {d['F']:.3f} & {d['M2']:.3f} & {d['M3']:.3f} & {Mcert[k]:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
# ---- midpoint table (Examples 2 and 4)
with open(R + "tables/midpoint.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrrrrr}\n\\toprule\nEx. & $K$ & $\\rho_h$ & max. step ratio & midpoint error & level (b) & Euler error & level $hM/(2c)$\\\\\n\\midrule\n")
    for k in "24":
        for r in D[k]["A"]:
            f.write(f"{k} & {r['K']} & {r['rho_mid_bound']:.3f} & {r['ratio_mid_obs']:.3f} & {tx(r['err_mid'])} & {tx(r['level_mid'])} & {tx(r['err_eul'])} & {tx(r['level_eul'])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
# ---- a posteriori table
with open(R + "tables/posteriori.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrrrr}\n\\toprule\nEx. & $K$ & true error & a posteriori bound & ratio & $hM_{\\rm emp}/(2c)$ & $hM_{\\rm cert}/(2c)$\\\\\n\\midrule\n")
    for k in "1234":
        for r in D[k]["B"]:
            if r["K"] == 64:
                c = D[k]["c"]
                f.write(f"{k} & {r['K']} & {tx(r['true_err'])} & {tx(r['post_bound'])} & {r['effectivity']:.1f} & {tx(r['prior_level'])} & {tx(r['h'] * Mcert[k] / (2 * c))}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
# ---- shooting
with open(R + "tables/shooting.tex", "w") as f:
    f.write("\\begin{tabular}{rrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Example 2} & \\multicolumn{2}{c}{Example 4}\\\\\n$j$ & error & bound & error & bound\\\\\n\\midrule\n")
    for j in (1, 2, 3, 4, 6, 8):
        a = D["2"]["shoot"]["rows"][j - 1]; b = D["4"]["shoot"]["rows"][j - 1]
        f.write(f"{j} & {tx(a['true'])} & {tx(a['bound'])} & {tx(b['true'])} & {tx(b['bound'])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}")
# ---- figures
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
for a, k in zip(ax, "24"):
    r = D[k]["A"]; h = np.array([x["h"] for x in r])
    a.loglog(h, [x["err_eul"] for x in r], "o-", label="implicit Euler")
    a.loglog(h, [x["err_mid"] for x in r], "s-", label="implicit midpoint")
    a.loglog(h, [x["level_eul"] for x in r], "k--", label=r"Euler level $hM/(2c)$")
    a.loglog(h, [x["level_mid"] for x in r], "k:", label="midpoint level (Thm.)")
    a.set_title(f"Example {k}"); a.set_xlabel("step size $h$"); a.minorticks_off()
ax[0].set_ylabel(r"$\sup|e_n|_P$"); ax[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("figures/figC_midpoint.pdf"); fig.savefig("figures/figC_midpoint.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
for a, k in zip(ax, "24"):
    r = D[k]["B"]; h = np.array([x["h"] for x in r]); c = D[k]["c"]
    a.loglog(h, [x["true_err"] for x in r], "o-", label="true error")
    a.loglog(h, [x["post_bound"] for x in r], "s-", label="a posteriori bound")
    a.loglog(h, [x["prior_level"] for x in r], "k--", label=r"$hM_{\rm emp}/(2c)$")
    a.loglog(h, h * Mcert[k] / (2 * c), "k:", label=r"$hM_{\rm cert}/(2c)$")
    a.set_title(f"Example {k}"); a.set_xlabel("step size $h$"); a.minorticks_off()
ax[0].set_ylabel("sup-norm distance (P-norm)"); ax[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("figures/figD_posteriori.pdf"); fig.savefig("figures/figD_posteriori.png", dpi=150); plt.close(fig)
# ---- macros
allB = [(k, r) for k in "1234" for r in D[k]["B"]]
eff = [r["effectivity"] for _, r in allB]
effK = {k: [r["effectivity"] for r in D[k]["B"]] for k in "1234"}
rel_mid = [r["ratio_mid_obs"] / r["rho_mid_bound"] for k in "1234" for r in D[k]["A"]]
okA = all(r["ratio_mid_obs"] <= r["rho_mid_bound"] + 1e-12 and r["ratio_eul_obs"] <= r["rho_eul_bound"] + 1e-12 for k in "1234" for r in D[k]["A"])
okLev = all(r["err_mid"] <= r["level_mid"] for k in "1234" for r in D[k]["A"])
okPost = all(r["true_err"] <= r["post_bound"] for k in "1234" for r in D[k]["B"])
okShoot = all(x["true"] <= x["bound"] * (1 + 1e-9) for k in "1234" for x in D[k]["shoot"]["rows"])
import math
def order(rows, key):
    return [math.log2(rows[i][key] / rows[i + 1][key]) for i in range(len(rows) - 1)]
with open(R + "tables/numbers_new.tex", "w") as f:
    W = lambda n, v: f.write("\\newcommand{\\%s}{%s}\n" % (n, v))
    W("cFour", f"{D['4']['c']:.4f}"); W("LFour", f"{D['4']['L']:.2f}"); W("FFour", f"{D['4']['F']:.2f}")
    W("MFour", f"{D['4']['M2']:.2f}"); W("MthreeFour", f"{D['4']['M3']:.2f}"); W("McertFour", f"{Mcert['4']:.0f}"); W("MratioFour", f"{Mcert['4']/D['4']['M2']:.0f}")
    W("effMin", f"{min(eff):.0f}"); W("effMax", f"{max(eff):.0f}")
    W("effTwoMin", f"{min(effK['2']):.0f}"); W("effFourMin", f"{min(effK['4']):.0f}"); W("effFourMax", f"{max(effK['4']):.0f}")
    W("ordEulFour", f"{np.mean(order(D['4']['A'], 'err_eul')[-2:]):.2f}")
    W("ordMidFour", f"{np.mean(order(D['4']['A'], 'err_mid')[-2:]):.2f}")
    W("ordMidTwo", f"{np.mean(order(D['2']['A'], 'err_mid')[-2:]):.2f}")
    W("ratioMidMax", f"{max(r['ratio_mid_obs']/r['rho_mid_bound'] for k in '1234' for r in D[k]['A']):.2f}")
    W("levelMidRatioMin", f"{min(r['level_mid']/r['err_mid'] for k in '1234' for r in D[k]['A']):.0f}")
    W("levelMidRatioMax", f"{max(r['level_mid']/r['err_mid'] for k in '1234' for r in D[k]['A']):.0f}")
    # post vs cert a priori at K=64
    q = [D[k]["B"][-1]["h"] * Mcert[k] / (2 * D[k]["c"]) / D[k]["B"][-1]["post_bound"] for k in "1234"]
    W("postVsCertMin", f"{min(q):.1f}"); W("postVsCertMax", f"{max(q):.0f}")
    q2 = [D[k]["B"][-1]["post_bound"] / D[k]["B"][-1]["prior_level"] for k in "1234"]
    W("postVsEmpMin", f"{min(q2):.1f}"); W("postVsEmpMax", f"{max(q2):.1f}")
    sh = [x["bound"] / x["true"] for k in "1234" for x in D[k]["shoot"]["rows"][1:]]
    W("shootRatioMax", f"{max(sh):.2f}"); W("shootRatioMin", f"{min(sh):.2f}")
json.dump(dict(okA=okA, okLev=okLev, okPost=okPost, okShoot=okShoot), open(R + "strengthen_checks.json", "w"))
print("checks", okA, okLev, okPost, okShoot, "eff", min(eff), max(eff))
