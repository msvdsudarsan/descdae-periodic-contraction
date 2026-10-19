# descdae: contraction and implicit Euler for monotone index-1 DAEs

Code, data and results for the study of periodically forced RC ladder descriptor systems
(Examples 1 and 2: diagonal algebraic part; Example 3: coupled algebraic part; Example 4: 64-node RC mesh, class `RCMesh`).

```
pip install -r requirements.txt        # tested versions: requirements-lock.txt
bash run_all.sh                        # tests, all experiments (about 30 min), tables/figures, checksums, claim checks
```
Steps: `experiments/run_all.py` (main study), `extra.py` (grid refinement of M, certified M), `coupled.py` (Example 3), `methods.py` and `make_methods_table.py` (comparison with implicit midpoint and trapezoidal rule),
`residual.py` (inexact algebraic solves: random, fixed-direction and error-aligned perturbations),
`strengthen.py <example> <A|B|AB>` (implicit midpoint rule: contraction factor and error; a posteriori bound and shooting stopping rule) with `make_strengthen_tables.py` (tables, figures, macros),
`robust.py` (variable step sizes and model-perturbation bound), `certified.py` (implicit Euler certificate and two heuristic stopping rules), `certified_mid.py` (second-order certificate for the implicit midpoint rule with cubic Hermite interpolant and the constants `L2`, `F2`) and `certified_J.py` (sampling sensitivity of the midpoint certificate, Example 2), `certified_blind.py` (Example 6: 200-node ladder, certificate without a reference solution) and `certified_pairs.py` (Euler data with Hermite interpolant and midpoint data with piecewise linear interpolant) and `make_cert_table.py`, on Examples 2, 4 and 5 (Example 5 is the random 40-node network `RCGraph`), `make_figs_tables.py` (tables, figures and the LaTeX macros), `manifest.py` (environment + SHA-256 of results),
`radau_reference_check.py` (Example 2: effect of the Radau reference tolerance on the reported error, about 2e-11 in the weighted norm),
`check_claims.py` (machine-check of the numerical statements made in the paper; exit code 1 on failure).

Notes. The reference solution and the empirical constant M are numerical; a proven a priori bound M_cert is reported next to it.
The "error-aligned" perturbation is a heuristic, not a proven worst case.
The manuscript is not part of this repository. This repository contains code, data, results, tests, README and LICENSE only.
License: MIT.
