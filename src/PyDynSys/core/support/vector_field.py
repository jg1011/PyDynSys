"""
Vector field representation with dual symbolic and callable forms.

Separate classes for autonomous and non-autonomous vector fields, matching
the mathematical structure: F: X → R^n (autonomous) and F: X × T → R^n (non-autonomous).
"""

from typing import Optional, List, Callable, Union
import numpy as np
from numpy.typing import NDArray
import sympy as syp
import warnings
import inspect

from .phase_space import PhaseSpace
from .time_horizon import TimeHorizon


class AutVectorField:
    """
    Autonomous vector field: F: X → R^n where X subset R^n is the phase space.
    
    Dual representation of vector field: symbolic + callable.
    Supports three usage patterns:
    1. Symbolic only: Provides symbolic expression; symbolic functionality but slow auto-compiled callable
    2. Callable only: Provides function directly; fast evaluation without symbolic functionality
    3. Both (recommended): Provides both for optimal performance with symbolic functionality
    
    Symbolic representation enables:
    - Rigorous property detection (linear, hamiltonian, etc.)
    - Symbolic differentiation (e.g. Jacobian computation)
    - Pretty printing
    
    Callable representation provides:
    - Fast numerical evaluation
    
    Fields:
        dimension: int - Phase space dimension n
        symbolic_expr: Optional[List[syp.Expr]] - Symbolic expressions [F_1, ..., F_n]
        callable_field: Callable[[NDArray[np.float64]], NDArray[np.float64]] - Numerical function F(x)
        phase_space: Optional[PhaseSpace] - Domain X \n 
            -> Not necessary if being fed to an _AutonomousDynSys inheritor, but useful for standalone anaysis.
            -> NOTE: If not provided, a warning is issued. 
    
    Example:
        >>> vf = AutonomousVectorField(
        ...     dimension=2,
        ...     callable_field=lambda x: np.array([x[1], -x[0]]), #SHO
        ...     symbolic_expr=[x2, -x1],
        ...     phase_space=PhaseSpace.full(2)  # Optional domain
        ... )
    """
    
    
    ### --- Constructor --- ###
    
    
    def __init__(
        self,
        dimension: int,
        callable_field: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        symbolic_expr: Optional[List[syp.Expr]] = None,
        phase_space: Optional[PhaseSpace] = None
    ):
        """
        Initialize autonomous vector field.
        
        Args:
            dimension: Phase space dimension n
            callable_field: Numerical function F(x) -> dx/dt mapping R^n -> R^n
            symbolic_expr: Optional symbolic expressions [F_1, ..., F_n]
            phase_space: Optional phase space X (domain of F). If None, warning is issued.
            
        Raises:
            ValueError: If callable_field is None or dimension <= 0
        """
        if callable_field is None:
            raise ValueError("callable_field must be provided")
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        
        self.dimension = dimension
        self.callable_field = callable_field
        self.symbolic_expr = symbolic_expr
        self.phase_space = phase_space
        
        # Validate dimension consistency
        if symbolic_expr is not None:
            if len(symbolic_expr) != dimension:
                raise ValueError(
                    f"symbolic_expr length ({len(symbolic_expr)}) must match dimension ({dimension})"
                )
        
        if phase_space is not None:
            if phase_space.dimension != dimension:
                raise ValueError(
                    f"phase_space dimension ({phase_space.dimension}) must match "
                    f"vector field dimension ({dimension})"
                )
        else:
            warnings.warn(
                f"AutonomousVectorField created without phase_space. "
                f"Mathematically, F: X → R^n requires domain X. "
                f"Consider providing phase_space for domain validation and mathematical completeness.",
                UserWarning,
                stacklevel=2
            )
        
        # Compile symbolic if needed
        if symbolic_expr is not None and callable_field is None:
            self.callable_field = self._compile_callable()
    
    
    ### --- Dunder Methods --- ###
    
    
    def __repr__(self) -> str:
        """String representation of vector field."""
        domain_str = f", phase_space={self.phase_space}" if self.phase_space else ", phase_space=None"
        return (
            f"AutonomousVectorField(dimension={self.dimension}, "
            f"symbolic_expr={self.symbolic_expr is not None}, "
            f"callable_field=<function>, {domain_str})"
        )
    
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate vector field numerically: F(x).
        
        Args:
            x: State vector in R^n
            
        Returns:
            Vector field evaluation F(x)
            
        Raises:
            ValueError: If x not in domain X (when phase_space provided)
            RuntimeError: If callable_field is None
        """
        if self.callable_field is None:
            raise RuntimeError("No callable representation available")
        
        # Optional domain validation
        if self.phase_space is not None:
            if not self.phase_space.contains_point(x):
                raise ValueError(
                    f"State {x} is not in domain X. "
                    f"Phase space constraints violated."
                )
        
        return self.callable_field(x)
    
    
    ### --- Private Methods --- ###
    
    
    def _compile_callable(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """
        Compile symbolic expression to numerical function.
        
        Returns:
            Compiled callable vector field function
            
        Raises:
            RuntimeError: If symbolic_expr is None
        """
        if self.symbolic_expr is None:
            raise RuntimeError("Cannot compile callable without symbolic_expr")
        
        # Build state symbols
        state_names = [f'x{i}' for i in range(self.dimension)]
        state_symbols = syp.symbols(' '.join(state_names))
        if self.dimension == 1:
            state_symbols = (state_symbols,)
        
        # Build autonomous field
        vector_field_funcs = [
            syp.lambdify(state_symbols, expr, 'numpy') 
            for expr in self.symbolic_expr
        ]
        
        def compiled_field(state: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compiled autonomous vector field."""
            args = tuple(state)
            derivatives = np.array([
                func(*args) for func in vector_field_funcs
            ], dtype=np.float64)
            return derivatives
        
        return compiled_field


class NonAutVectorField:
    """
    Non-autonomous vector field: F: X × T → R^n where X ⊆ R^n is phase space, T ⊆ R is time horizon.
    
    Dual representation of vector field: symbolic + callable.
    Supports three usage patterns:
    1. Symbolic only: Provides symbolic expression, callable auto-compiled
    2. Callable only: Provides function directly (fast, no symbolic ops)
    3. Both (recommended): Provides both for optimal performance
    
    Symbolic representation enables:
    - Rigorous property detection (linear, hamiltonian, etc.)
    - Symbolic differentiation (Jacobian computation)
    - Pretty printing
    
    Callable representation provides:
    - Fast numerical evaluation
    
    Fields:
        dimension: int - Phase space dimension n
        symbolic_expr: Optional[List[syp.Expr]] - Symbolic expressions [F_1, ..., F_n]
        callable_field: Callable[[NDArray[np.float64], float], NDArray[np.float64]] - Numerical function F(x, t)
        phase_space: Optional[PhaseSpace] - Domain X (optional, but recommended)
        time_horizon: Optional[TimeHorizon] - Time domain T (optional, but recommended)
    
    Example:
        >>> vf = NonAutonomousVectorField(
        ...     dimension=2,
        ...     callable_field=lambda x, t: np.array([x[1], -x[0] + np.sin(t)]),
        ...     symbolic_expr=[x2, -x1 + sin(t)],
        ...     phase_space=PhaseSpace.full(2),  # Optional domain
        ...     time_horizon=TimeHorizon.real_line()  # Optional time domain
        ... )
    """
    
    def __init__(
        self,
        dimension: int,
        callable_field: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
        symbolic_expr: Optional[List[syp.Expr]] = None,
        phase_space: Optional[PhaseSpace] = None,
        time_horizon: Optional[TimeHorizon] = None
    ):
        """
        Initialize non-autonomous vector field.
        
        Args:
            dimension: Phase space dimension n
            callable_field: Numerical function F(x, t) -> dx/dt mapping R^n × R -> R^n
            symbolic_expr: Optional symbolic expressions [F_1, ..., F_n]
            phase_space: Optional phase space X (domain of F). If None, warning is issued.
            time_horizon: Optional time horizon T (time domain). If None, warning is issued.
            
        Raises:
            ValueError: If callable_field is None or dimension <= 0
        """
        if callable_field is None:
            raise ValueError("callable_field must be provided")
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        
        self.dimension = dimension
        self.callable_field = callable_field
        self.symbolic_expr = symbolic_expr
        self.phase_space = phase_space
        self.time_horizon = time_horizon
        
        # Validate dimension consistency
        if symbolic_expr is not None:
            if len(symbolic_expr) != dimension:
                raise ValueError(
                    f"symbolic_expr length ({len(symbolic_expr)}) must match dimension ({dimension})"
                )
        
        if phase_space is not None:
            if phase_space.dimension != dimension:
                raise ValueError(
                    f"phase_space dimension ({phase_space.dimension}) must match "
                    f"vector field dimension ({dimension})"
                )
        else:
            warnings.warn(
                f"NonAutonomousVectorField created without phase_space. "
                f"Mathematically, F: X × T → R^n requires domain X. "
                f"Consider providing phase_space for domain validation and mathematical completeness.",
                UserWarning,
                stacklevel=2
            )
        
        if time_horizon is None:
            warnings.warn(
                f"NonAutonomousVectorField created without time_horizon. "
                f"Mathematically, F: X × T → R^n requires time domain T. "
                f"Consider providing time_horizon for domain validation and mathematical completeness.",
                UserWarning,
                stacklevel=2
            )
        
        # Compile symbolic if needed
        if symbolic_expr is not None and callable_field is None:
            self.callable_field = self._compile_callable()
            
            
    ### --- Dunder Methods --- ###
    
    
    def __repr__(self) -> str:
        """String representation of vector field."""
        domain_str = f", phase_space={self.phase_space is not None}, time_horizon={self.time_horizon is not None}"
        return (
            f"NonAutonomousVectorField(dimension={self.dimension}, "
            f"symbolic_expr={self.symbolic_expr is not None}, "
            f"callable_field=<function>, {domain_str})"
        )
    
    def __call__(self, x: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """
        Evaluate vector field numerically: F(x, t).
        
        Args:
            x: State vector in R^n
            t: Time in R
            
        Returns:
            Vector field evaluation F(x, t)
            
        Raises:
            ValueError: If x not in domain X or t not in T (when domains provided)
            RuntimeError: If callable_field is None
        """
        if self.callable_field is None:
            raise RuntimeError("No callable representation available")
        
        # Optional domain validation
        if self.phase_space is not None:
            if not self.phase_space.contains_point(x):
                raise ValueError(
                    f"State {x} is not in domain X. "
                    f"Phase space constraints violated."
                )
        
        if self.time_horizon is not None:
            if not self.time_horizon.contains_time(t):
                raise ValueError(
                    f"Time {t} is not in time domain T. "
                    f"Time horizon constraints violated."
                )
        
        return self.callable_field(x, t)
    
    
    ### --- Private Methods --- ###
    
    
    def _compile_callable(self) -> Callable[[NDArray[np.float64], float], NDArray[np.float64]]:
        """
        Compile symbolic expression to numerical function.
        
        Returns:
            Compiled callable vector field function
            
        Raises:
            RuntimeError: If symbolic_expr is None
        """
        if self.symbolic_expr is None:
            raise RuntimeError("Cannot compile callable without symbolic_expr")
        
        # Build state symbols
        state_names = [f'x{i}' for i in range(self.dimension)]
        state_symbols = syp.symbols(' '.join(state_names))
        if self.dimension == 1:
            state_symbols = (state_symbols,)
        
        # Build non-autonomous field
        t_sym = syp.symbols('t')
        state_vars = state_symbols + (t_sym,)
        
        vector_field_funcs = [
            syp.lambdify(state_vars, expr, 'numpy')
            for expr in self.symbolic_expr
        ]
        
        def compiled_field(
            state: NDArray[np.float64], 
            time: float
        ) -> NDArray[np.float64]:
            """Compiled non-autonomous vector field."""
            args = tuple(state) + (time,)
            derivatives = np.array([
                func(*args) for func in vector_field_funcs
            ], dtype=np.float64)
            return derivatives
        
        return compiled_field


### --- Factory Function --- ###


def vector_field_factory(
    dimension: int,
    callable_field: Optional[Callable] = None,
    symbolic_expr: Optional[List[syp.Expr]] = None,
    phase_space: Optional[PhaseSpace] = None,
    time_horizon: Optional[TimeHorizon] = None
) -> Union[AutVectorField, NonAutVectorField]:
    """
    Factory: Auto-detect and construct appropriate vector field class.
    
    Detection logic:
    1. If callable provided: inspect signature to determine autonomous/non-autonomous
    2. If symbolic only: detect from expressions (check for time symbol)
    3. If both: validate consistency, use callable signature for detection
    
    Args:
        dimension: Phase space dimension n
        callable_field: Optional numerical function
        symbolic_expr: Optional symbolic expressions [F_1, ..., F_n]
        phase_space: Optional phase space X (domain)
        time_horizon: Optional time horizon T (only used for non-autonomous)
        
    Returns:
        AutonomousVectorField or NonAutonomousVectorField based on detection
        
    Raises:
        ValueError: If both callable_field and symbolic_expr are None
        ValueError: If detection is ambiguous or inconsistent
    """
    if callable_field is None and symbolic_expr is None:
        raise ValueError(
            "from_input requires at least one representation: "
            "callable_field or symbolic_expr must be provided"
        )
    
    # Detection strategy: prefer callable signature, fall back to symbolic analysis
    is_autonomous = None
    
    # Try callable signature detection
    if callable_field is not None:
        try:
            sig = inspect.signature(callable_field)
            params = list(sig.parameters.values())
            
            # Count parameters (excluding self if bound method)
            # Autonomous: F(x) -> 1 parameter
            # Non-autonomous: F(x, t) -> 2 parameters
            if len(params) == 1:
                is_autonomous = True
            elif len(params) == 2:
                is_autonomous = False
            else:
                raise ValueError(
                    f"Callable must take 1 parameter (autonomous) or 2 parameters (non-autonomous), "
                    f"got {len(params)}"
                )
        except (ValueError, TypeError):
            # If signature inspection fails, try calling with test input
            # This is a fallback for callables without proper signatures
            pass
    
    # Fall back to symbolic detection
    if is_autonomous is None and symbolic_expr is not None:
        # Check if time symbol appears in expressions
        t_sym = syp.symbols('t')
        has_time = any(
            t_sym in expr.free_symbols for expr in symbolic_expr
        )
        is_autonomous = not has_time
    
    # If still unknown, default to autonomous (simpler case)
    if is_autonomous is None:
        warnings.warn(
            "Could not determine if vector field is autonomous or non-autonomous. "
            "Defaulting to autonomous. Provide explicit callable_field with clear signature.",
            UserWarning,
            stacklevel=2
        )
        is_autonomous = True
    
    # Construct appropriate class
    if is_autonomous:
        if time_horizon is not None:
            warnings.warn(
                "time_horizon provided but vector field is autonomous. Ignoring time_horizon.",
                UserWarning,
                stacklevel=2
            )
        return AutVectorField(
            dimension=dimension,
            callable_field=callable_field,
            symbolic_expr=symbolic_expr,
            phase_space=phase_space
        )
    else:
        return NonAutVectorField(
            dimension=dimension,
            callable_field=callable_field,
            symbolic_expr=symbolic_expr,
            phase_space=phase_space,
            time_horizon=time_horizon
        )
