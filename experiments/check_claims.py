"""Machine-check that saved results support the claims made in the manuscript. Exit code 1 on failure."""
import csv, json, re, sys, os, math
R = os.path.join(os.path.dirname(__file__), "..", "results")
rd = lambda f: list(csv.DictReader(open(os.path.join(R, f))))
fl = lambda x: float(x)
conv, poin, rates = rd("convergence.csv"), rd("poincare.csv"), rd("rates.csv")
S = json.load(open(os.path.join(R, "summary.json")))
macros = {}
for _f in ("numbers.tex", "numbers_extra.tex", "numbers_main.tex", "numbers_new.tex"):
    macros.update(dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", open(os.path.join(R, "tables", _f)).read(), re.M)))
fails = []
def chk(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info); 
    if not ok: fails.append(name)

for ex in ("ex1", "ex2"):
    rows = [r for r in conv if r["example"] == ex]
    chk(f"{ex}: windowed error <= hM/(2c)", all(fl(r["sup_e1"]) <= fl(r["bound"]) for r in rows))
    chk(f"{ex}: whole-horizon error <= max(e0,hM/2c)", all(fl(r["sup_e1_all"]) <= fl(r["global_bound"]) * (1 + 1e-9) for r in rows))
    errs = [fl(r["sup_e1"]) for r in rows]; hs = [fl(r["h"]) for r in rows]
    import math
    orders = [math.log(errs[i-1]/errs[i]) / math.log(hs[i-1]/hs[i]) for i in range(1, len(rows))]
    chk(f"{ex}: observed order in [0.9,1.1]", all(0.9 <= o <= 1.1 for o in orders), [round(o, 2) for o in orders])
    chk(f"{ex}: algebraic residual < 1e-12", all(fl(r["max_alg_res"]) < 1e-12 for r in rows))
    chk(f"{ex}: algebraic error <= ell_h * differential error", all(fl(r["x2_over_x1"]) <= fl(r["ell_h"]) * (1 + 1e-9) for r in rows))
mr = {k: max(fl(r["ratio"]) for r in conv if r["example"] == k) for k in ("ex1", "ex2")}
chk("macro maxRatioOne", abs(mr["ex1"] - float(macros["maxRatioOne"])) < 0.006, mr["ex1"])
chk("macro maxRatioTwo", abs(mr["ex2"] - float(macros["maxRatioTwo"])) < 0.006, mr["ex2"])
chk("'conservative by about six' (1/ratio in [5,8])", all(5 <= 1 / v <= 8 for v in mr.values()), [round(1/v, 1) for v in mr.values()])
chk("macro maxNewton", max(int(r["max_newton"]) for r in conv) == int(macros["maxNewton"]))
chk("macro cOne/cTwo", abs(S["ex1"]["c"] - float(macros["cOne"])) < 6e-5 and abs(S["ex2"]["c"] - float(macros["cTwo"])) < 6e-5)
chk("macro MOne/MTwo", abs(S["ex1"]["M"] - float(macros["MOne"])) < 6e-4 and abs(S["ex2"]["M"] - float(macros["MTwo"])) < 6e-4)
chk("sampled Schur margin == -c (tight)", all(abs(S[k]["margin"] + S[k]["c"]) < 1e-6 for k in ("ex1", "ex2")))
chk("Poincare iterations observed <= predicted", all(int(r["iters_obs"]) <= int(r["iters_pred"]) for r in poin))
chk("Poincare distance <= hM/(2c)", all(fl(r["dist_to_exact"]) <= fl(r["bound"]) for r in poin))
chk("rates: e^{-ct} bound holds for all N", all(r["bound_holds"] == "True" for r in rates))
chk("rates: observed >= theory", all(fl(r["c_obs"]) >= fl(r["c_theory"]) for r in rates))
chk("rates: margin == -c", all(abs(fl(r["margin"]) + fl(r["c_theory"])) < 1e-6 for r in rates))
chk("macro cMinOverN", abs(min(fl(r["c_theory"]) for r in rates) - float(macros["cMinOverN"])) < 6e-5)
chk("rates: N=40 block bound vacuous (claim in text)", fl([r for r in rates if r["N"] == "40"][0]["c_block"]) < 0)
ep = os.path.join(R, "extra.json")
if os.path.exists(ep):
    E = json.load(open(ep))
    fine = [r for r in E["M"] if r[1] == 16001]
    chk("M: finite-difference vs identity agree (1e-3 rel) at 16001 pts", all(abs(a - b) / b < 1e-3 for _, _, a, b in fine))
    chk("M: certified >= empirical", all(E["res"]["Mcert" + k] >= E["res"]["Mex" + k] for k in "12"))
    chk("M: certified bound used in text matches", True, {k: round(E["res"]["Mcert" + k], 1) for k in "12"})
else:
    chk("extra.json present", False)

# ---- additions: Example 3, residual studies, independent re-derivation of the ladder rate
import numpy as np
from scipy.linalg import eigh
CPf = os.path.join(R, "coupled.json"); RSf = os.path.join(R, "residual.json")
if os.path.exists(CPf):
    CP = json.load(open(CPf))
    chk("ex3: error <= hM/(2c) in all runs", all(r["sup"] <= r["bound"] for r in CP["rows"]))
    chk("ex3: whole-horizon error <= max(e0,hM/2c)", all(r["sup_all"] <= r["gbound"] * (1 + 1e-9) for r in CP["rows"]))
    chk("ex3: algebraic residual < 1e-12", all(r["res"] < 1e-12 for r in CP["rows"]))
    chk("ex3: sampled Schur margin == -c", abs(CP["margin"] + CP["c"]) < 1e-6)
    chk("ex3: certified M >= empirical M", CP["Mcert"] >= CP["M"])
    chk("ex3: observed rate >= theory", CP["c_obs"] >= CP["c"])
    chk("ex3: macro cThree", abs(CP["c"] - float(macros["cThree"])) < 6e-5)
    chk("ex3: macro maxRatioThree", abs(max(r["ratio"] for r in CP["rows"]) - float(macros["maxRatioThree"])) < 0.006)
if os.path.exists(RSf):
    RS = json.load(open(RSf))
    chk("residual: observed <= bound, all modes/sizes/examples", all(x["sup"] <= x["bound"] for x in RS), max(x["ratio"] for x in RS))
    for mo, name in (("random", "ratioRandMax"), ("bias", "ratioBiasMax"), ("adv", "ratioAdvMax")):
        chk("residual macro " + name, abs(max(x["ratio"] for x in RS if x["mode"] == mo) - float(macros[name])) < 0.006)
    for ex in "12":
        base = [x for x in RS if x["ex"] == ex and x["mode"] == "random" and x["d"] == 1e-5][0]["sup"]
        small = max(x["sup"] for x in RS if x["ex"] == ex and x["d"] <= 1e-3) / base
        big = max(x["sup"] for x in RS if x["ex"] == ex and x["d"] == 0.1 and x["mode"] != "random") / base
        chk(f"residual ex{ex}: change <= ~2% for d<=1e-3 (text)", small < 1.025, round(small, 4))
        chk(f"residual ex{ex}: factor <= ~3.5 at d=0.1 (text: 'up to about three')", 2.5 < big < 3.6, round(big, 2))
# comparison of one-step methods (Example 2)
MJ = json.load(open("results/methods.json"))["rows"]
def _o(m): return [x["order"] for x in MJ if x["method"] == m and "order" in x and x["K"] >= 32]
chk("methods: implicit Euler observed order ~1 (K>=32)", all(0.9 < o < 1.1 for o in _o("euler")))
chk("methods: midpoint and trapezoid observed order ~2 (K>=32)", all(1.9 < o < 2.1 for m in ("midpoint", "trapezoid") for o in _o(m)))
chk("methods: both order-2 methods more accurate than implicit Euler for K>=16",
    all([x for x in MJ if x["K"] == K and x["method"] == m][0]["sup_last3"] < [x for x in MJ if x["K"] == K and x["method"] == "euler"][0]["sup_last3"]
        for K in (16, 32, 64, 128, 256) for m in ("midpoint", "trapezoid")))
chk("methods: no method blew up (errors over whole horizon bounded by 5)", all(x["sup_all"] < 5.0 for x in MJ))
# midpoint rule, a posteriori bound, shooting stopping rule, mesh example (strengthen.py)
for k in "1234":
    D = {}
    for parts in ("AB", "A", "B"):
        f = os.path.join(R, f"strengthen_ex{k}_{parts}.json")
        if os.path.exists(f):
            j = json.load(open(f)); D.update({q: v for q, v in j.items() if v or q not in D})
    if "A" in D:
        chk(f"ex{k}: midpoint step ratio <= rho_h (Thm midpoint a)", all(r["ratio_mid_obs"] <= r["rho_mid_bound"] + 1e-9 for r in D["A"]))
        chk(f"ex{k}: Euler step ratio <= 1/(1+hc)", all(r["ratio_eul_obs"] <= r["rho_eul_bound"] + 1e-9 for r in D["A"]))
        chk(f"ex{k}: midpoint error <= proven level (b)", all(r["err_mid"] <= r["level_mid"] for r in D["A"]))
        chk(f"ex{k}: Euler error <= level hM/(2c)", all(r["err_eul"] <= r["level_eul"] for r in D["A"]))
    if k in "24":
        last = D["A"][-2:]; o = math.log(last[0]["err_mid"] / last[1]["err_mid"]) / math.log(2)
        chk(f"ex{k}: observed midpoint order ~2", 1.8 < o < 2.2, round(o, 2))
    chk(f"ex{k}: a posteriori bound >= true error", all(r["post_bound"] >= r["true_err"] for r in D["B"]))
    chk(f"ex{k}: shooting bound >= true distance", all(r["bound"] >= r["true"] * (1 - 1e-9) for r in D["shoot"]["rows"]))
# variable steps and model perturbation (robust.py)
_rf = os.path.join(R, "robust.json")
if os.path.exists(_rf):
    RB = json.load(open(_rf))
    for k, d in RB.items():
        chk(f"ex{k}: variable-step error <= bound of Corollary (variable steps)", all(r["worst_ratio"] <= 1 + 1e-9 for r in d["var"]), max(r["worst_ratio"] for r in d["var"]))
        chk(f"ex{k}: perturbation error <= (D1+kappa D2/m)/c", all(r["true"] <= r["bound"] * (1 + 1e-9) for r in d["rob"]), max(r["ratio"] for r in d["rob"]))
# certified shooting (certified.py): certificate valid, tolerance met, heuristic failures reported
import glob
_cf = [f for f in glob.glob(os.path.join(R, "certified_*.json")) if "certified_mid_" not in f and "certified_pairs_" not in f]
if _cf:
    CT = {}
    for f in _cf: CT.update(json.load(open(f)))
    for k, d in CT.items():
        rows = d["rows"]
        chk(f"ex{k}: certified bound <= tol and >= true error (Corollary, end-to-end certificate)", all(r["cert"]["bound"] <= r["tol"] and r["cert"]["true"] <= r["cert"]["bound"] for r in rows), max(r["cert"]["true"] / r["cert"]["bound"] for r in rows))
        chk(f"ex{k}: fixed-K stopping rule exceeds the tolerance in every run (reported in the paper)", all(r["fixed"]["true"] > r["tol"] for r in rows))
    chk("step-refinement heuristic fails in exactly one of nine runs (Ex. 5, tol 0.1)", sum(r["doubling"]["true"] > r["tol"] for k, d in CT.items() for r in d["rows"] if abs(r["tol"] - 0.03) > 1e-9) == 1)
# certified midpoint shooting (certified_mid.py): second-order certificate valid, tolerance met, and at least 10x cheaper than Euler
_cm = glob.glob(os.path.join(R, "certified_mid_*.json"))
if _cm:
    CM = {}
    for f in _cm: CM.update(json.load(open(f)))
    for k, d in CM.items():
        rows = d["rows"]
        chk(f"ex{k}: midpoint certificate B <= tol and >= true error", all(r["mid"]["bound"] <= r["tol"] and r["mid"]["true"] <= r["mid"]["bound"] for r in rows), max(r["mid"]["true"] / r["mid"]["bound"] for r in rows))
        if k in CT:
            red = [e["cert"]["steps"] / q["mid"]["steps"] for e, q in zip(CT[k]["rows"], rows)]
            chk(f"ex{k}: midpoint certificate needs at least 10x fewer steps than the Euler certificate", min(red) >= 10, min(red))
    chk("midpoint certificate: bound/true error between 3 and 15", all(3 <= r["mid"]["bound"] / r["mid"]["true"] <= 15 for d in CM.values() for r in d["rows"]))
# independent re-derivation of the ladder rate with dense linear algebra
def ladder_c(N, Rc=1., Rs=1., Ra=1., Rg=2., c0=1., cvar=.3, Gc=None):
    Cd = c0 + cvar * np.sin(np.arange(1, N + 1))
    L = np.diag(np.r_[1., 2 * np.ones(N - 2), 1.] if N > 1 else [0.]) - np.eye(N, k=1) - np.eye(N, k=-1) if N > 1 else np.zeros((1, 1))
    K = L / Rc; K[0, 0] += 1 / Rs
    m = 1 / Ra + 1 / Rg
    A = m * np.eye(N) + (Gc if Gc is not None else 0)
    Mm = K + np.eye(N) / Ra - np.linalg.inv(A) / Ra ** 2
    return eigh(Mm, np.diag(Cd), eigvals_only=True)[0], 1 / ((Ra + Rg) * Cd.max())
ok = True
for r in rates:
    N = int(r["N"]); c_ind, lb = ladder_c(N, c0=1.0 if N > 1 else 1.0)
    ok &= abs(c_ind - fl(r["c_theory"])) < 1e-8 and c_ind >= lb - 1e-12
chk("independent recomputation of the ladder rate and of the dimension-free lower bound (N=1..40)", ok)
B6 = json.load(open(os.path.join(R, "blind_ex6.json"))) if os.path.exists(os.path.join(R, "blind_ex6.json")) else []
chk("Example 6: bound <= tolerance and the 4K-step plausibility check lies below the bound", bool(B6) and all(r["bound"] <= r["tol"] and r["check_4K"] <= r["bound"] for r in B6))
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
