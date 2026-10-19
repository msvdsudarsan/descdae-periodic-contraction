"""descdae: contraction analysis and uniform-in-time error bounds for
periodically forced index-1 semilinear descriptor systems."""
from .models import RCLadder, CoupledRCLadder, RCMesh, RCGraph
from .solvers import implicit_euler, reference_periodic, numeric_poincare, inexact_euler
from .analysis import second_derivative_bound, observed_rate, sampled_contraction_margin

__all__ = ["RCLadder", "CoupledRCLadder", "RCMesh", "RCGraph", "implicit_euler", "reference_periodic", "numeric_poincare",
           "second_derivative_bound", "observed_rate", "sampled_contraction_margin"]
