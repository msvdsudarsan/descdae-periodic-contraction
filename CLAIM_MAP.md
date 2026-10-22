# Claim-to-code map

Status key: **Proof** = argument in the paper (full proofs in Online Resource 1); **Test** = automated check in `tests/test_core.py`; **Data** = saved raw output in `results/` checked by `experiments/check_claims.py`; **Balls** = enclosure in ball arithmetic (python-flint, `experiments/verified_cert.py`). Tests and data checks sample instances; they do not replace the proofs.

| Result in the paper | Status | Check |
|---|---|---|
| Lem. 1 (solvability), Lem. 2 (global reduction) | Proof | `test_algebraic_solve_residual`, `test_schur_matches_finite_difference` |
| Lem. 3, Thm. 4 (contraction) | Proof | `test_contraction_margin_below_minus_c` |
| Thm. 5 (periodic steady state) | Proof | `check_claims.py` (Poincare data) |
| Thm. 6 (implicit Euler), Cor. 7 (variable steps) | Proof | `test_implicit_euler_converges_to_reference`, `test_variable_step_euler_contraction_and_level`, `check_claims.py` (convergence data) |
| Prop. 8 (bound on M), Prop. 9 (inexact solves) | Proof | `check_claims.py`, `experiments/fp_inexact.py` |
| Thm. 10 (implicit midpoint) | Proof | `test_midpoint_contraction_and_mesh`, `test_midpoint_rho_h_random_instances` |
| Prop. 11 (scalar sharpness) | Proof | `test_contraction_factors_are_attained_in_linear_scalar_example`, `experiments/sharpness_check.py` |
| Thm. 12 (a posteriori bound), Cor. 13-14 | Proof | `test_end_to_end_certificate_holds` |
| Prop. 15 (second-order defect bound), Prop. 16 (algebraic residual) | Proof | `test_midpoint_defect_bound_is_rigorous_and_L2_valid`, `experiments/inexact_defect.py` |
| Prop. 17 (ball arithmetic) | Balls | `experiments/verified_cert.py`; encloses the constants and defect under rounding, it does not verify the program or the analytic inequalities |
| Prop. 18, Cor. 19 (ladder and network rates) | Proof | `check_claims.py`, `experiments/larger_ladders.py` |
| Abstract ranges (17-132 times, 4.3-11.4) | Data | recomputed from raw results by `check_claims.py` |

The Radau reference solution used in some comparisons has its own error; those comparisons are empirical. Theorem numbers above follow the order of the statements in `main.tex`; check them against the compiled paper.

Full check: `bash quick_verify.sh` (needs `pip install -r requirements-lock.txt`, which includes python-flint 0.9.0).
