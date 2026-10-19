"""LaTeX table for the comparison of one-step methods (results/methods.json)."""
import json, math
d = json.load(open("results/methods.json"))
def tx(v):
    m, e = f"{v:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
rows = []
for K in sorted({r["K"] for r in d["rows"]}):
    cells = [str(K)]
    for m in ("euler", "midpoint", "trapezoid"):
        r = [x for x in d["rows"] if x["K"] == K and x["method"] == m][0]
        cells.append(tx(r["sup_last3"]) + (f" ({r['order']:.2f})" if "order" in r else ""))
    rows.append(" & ".join(cells) + "\\\\")
tab = ("\\begin{tabular}{rccc}\n\\toprule\n$K$ & implicit Euler & implicit midpoint & trapezoidal rule\\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open("results/tables/methods.tex", "w").write(tab)
print(tab)
