#!/usr/bin/env bash
set -e
export PYTHONPATH=src
python -m pytest -q tests
python experiments/run_all.py          # main experiments (Examples 1, 2), ~10 min
python experiments/extra.py            # refinement study of M, certified M, ~3 min
python experiments/coupled.py          # Example 3 (coupled algebraic part)
python experiments/residual.py         # inexact algebraic solves, ~15 min
python experiments/methods.py           # comparison with midpoint and trapezoidal rule, ~5 min
python experiments/make_methods_table.py
python experiments/strengthen.py 1 AB  # midpoint rule, a posteriori bound, shooting (Ex. 1-3: ~1 min each)
python experiments/strengthen.py 2 AB
python experiments/strengthen.py 3 AB
python experiments/strengthen.py 4 A   # Example 4 (64-node RC mesh), ~7 min
python experiments/strengthen.py 4 B
for k in 1 2 3 4; do python experiments/strengthen.py $k B; done   # a posteriori part with 32 samples per step
python experiments/make_strengthen_tables.py
(cd experiments && PYTHONPATH=../src python robust.py)   # variable steps, model perturbation
(cd experiments && TOLS=1e-1,3e-2,1e-2 python certified.py 2 5)   # certified shooting vs heuristics, ~6 min
(cd experiments && TOLS=1e-1,3e-2,1e-2 python certified.py 4)     # same for the 64-node mesh, ~5 min
(cd experiments && TOLS=1e-1,3e-2,1e-2 python certified_mid.py 2)   # certified midpoint shooting, ~1 min
(cd experiments && TOLS=1e-1,3e-2,1e-2 python certified_mid.py 4 5)
(cd experiments && python certified_J.py)   # sampling sensitivity, Example 2
(cd experiments && python certified_blind.py)   # Example 6: certificate without a reference solution, ~1.5 min
(cd experiments && TOLS=1e-1,1e-2 python certified_pairs.py 2 5)   # mixed interpolant/data pairs
(cd experiments && TOLS=1e-1,1e-2 python certified_pairs.py 4)
python experiments/make_cert_table.py
python experiments/make_figs_tables.py # constants, tables, figures, macros
python experiments/manifest.py         # environment and checksums
python experiments/check_claims.py     # machine-check of the numerical claims in the paper
