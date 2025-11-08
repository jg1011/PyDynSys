"""Type definitions for PyDynSys dynamical systems."""

from typing import Union, List, Callable, Dict, Tuple, Any, Literal, Optional, Set
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import sympy as syp


### Symbolic Types ###


SymbolicODE = Union[List[syp.Expr], syp.Expr, str, List[str]]
"""
Union type for symbolic ODE representations.
Accepts: single expression, list of expressions, or string forms.
Expected format: "d(x_i)/dt - F_i(x, t)"
"""

SystemParameters = Union[Dict[str, float], Dict[syp.Symbol, float]]
"""
Parameter substitution dictionary for symbolic systems.
Keys: parameter names (str) or SymPy symbols
Values: numerical parameter values
"""


### IVP Solution Types ###


class SciPyIvpSolution:
    """
    Wrapper for the result object from scipy.integrate.solve_ivp.
    
    Fields:
        raw_solution (Any): The scipy OdeResult object
    """
    raw_solution: Any  # scipy.integrate.OdeResult not exposed in type stubs
    
    def __init__(self, raw_solution: Any):
        self.raw_solution = raw_solution
    
    @property
    def t(self) -> NDArray[np.float64]:
        """Time points where solution is evaluated - shape (n_points,)"""
        return self.raw_solution.t
    
    @property
    def y(self) -> NDArray[np.float64]:
        """State vectors at each time point - shape (n_dim, n_points)"""
        return self.raw_solution.y
    
    @property
    def sol(self) -> Any:
        """
        Continuous solution interpolant: callable f(t) -> x(t)
        
        NOTE: Most accurate within [min(t), max(t)], extrapolates beyond.
        Only available if dense_output=True was passed to solve_ivp.
        """
        return self.raw_solution.sol
    
    @property
    def success(self) -> bool:
        """Whether integration succeeded"""
        return self.raw_solution.success
    
    @property
    def message(self) -> str:
        """Human-readable description of termination reason"""
        return self.raw_solution.message




