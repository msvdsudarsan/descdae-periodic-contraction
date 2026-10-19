"""Builds results/tables/cert_table.tex from results/certified_*.json (Euler certificate, baselines) and
results/certified_mid_*.json (midpoint certificate): certified shooting against step refinement."""
import json, glob, os
here = os.path.dirname(os.path.abspath(__file__))
eul, mid = {}, {}
for f in sorted(glob.glob(os.path.join(here, "..", "results", "certified_*.json"))):
    b = os.path.basename(f)
    if b.startswith("certified_pairs_"): continue
    (mid if b.startswith("certified_mid_") else eul).update(json.load(open(f)))
names = {"2": "Ex.~2", "4": "Ex.~4", "5": "Ex.~5"}
def e(x):
    m, p = ("%.1e" % x).split("e"); return "$%s\\!\\cdot\\!10^{%d}$" % (m, int(p))
rows = []
for k in ("2", "4", "5"):
    for r, q in zip(eul[k]["rows"], mid[k]["rows"]):
        if abs(r["tol"] - 0.03) < 1e-9: continue
        assert abs(r["tol"] - q["tol"]) < 1e-12
        c, mm, b = r["cert"], q["mid"], q["mid_doubling"]
        flag = "" if b["true"] <= r["tol"] else "$^\\ast$"
        rows.append("%s & %s & %d & %d & %d & %d & %s & %s & %d & %d & %s%s \\\\" % (
            names[k], e(r["tol"]), c["steps"], mm["K"], mm["steps"], mm["evals"], e(mm["bound"]), e(mm["true"]),
            b["steps"], b["evals"], e(b["true"]), flag))
tab = ("\\setlength{\\tabcolsep}{3.2pt}\\footnotesize\\begin{tabular}{llr|rrrcc|rrc}\n\\hline\n"
       " & & Euler cert. & \\multicolumn{5}{c}{midpoint certificate} & \\multicolumn{3}{c}{midpoint refinement} \\\\\n"
       " & tol & steps & $K$ & steps & solves & bound $B$ & error & steps & solves & error \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")
os.makedirs(os.path.join(here, "..", "results", "tables"), exist_ok=True)
open(os.path.join(here, "..", "results", "tables", "cert_table.tex"), "w").write(tab)
print(tab)

# second table: Euler certificate and the heuristics (certified.py output)
rows2 = []
for k in ("2", "4", "5"):
    for r in eul[k]["rows"]:
        if abs(r["tol"] - 0.03) < 1e-9: continue
        c, f, d = r["cert"], r["fixed"], r["doubling"]
        flag = lambda t: "" if t <= r["tol"] else "$^\\ast$"
        rows2.append("%s & %s & %d & %d & %d & %s & %s & %s%s & %d & %s%s \\\\" % (
            names[k], e(r["tol"]), c["K"], c["steps"], c["evals"], e(c["bound"]), e(c["true"]),
            e(f["true"]), flag(f["true"]), d["steps"], e(d["true"]), flag(d["true"])))
tab2 = ("\\setlength{\\tabcolsep}{3.2pt}\\footnotesize\\begin{tabular}{ll|rrrcc|c|rc}\n\\hline\n"
        " & & \\multicolumn{5}{c}{Euler certificate} & fixed $K=16$ & \\multicolumn{2}{c}{Euler refinement} \\\\\n"
        " & tol & $K$ & steps & solves & bound $B$ & error & error & steps & error \\\\\n\\hline\n" + "\n".join(rows2) + "\n\\hline\n\\end{tabular}\n")
open(os.path.join(here, "..", "results", "tables", "cert_table2.tex"), "w").write(tab2)
print(tab2)
