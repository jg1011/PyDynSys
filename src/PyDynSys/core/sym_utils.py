"""Symbolic to numerical system conversion utilities."""

from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import sympy as syp

from .types import (
    SymbolicODE, 
    SystemParameters, 
    AutonomousVectorField, 
    NonAutonomousVectorField,
    VectorField
)

@dataclass(frozen=True)
class SymbolicToVectorFieldResult:
    """
    Result of converting symbolic ODE to numerical vector field.
    
    Encapsulates all information about the conversion process. Frozen
    to ensure immutability and hashability (if needed for caching).
    
    Fields:
        vector_field: Callable vector field with appropriate signature on is_autonomous bool.
        dimension: dimension of the phase space (not as linear subspace), i.e. #state_components
        is_autonomous: True <==> partial(vector_field, t) == 0
        
    Future Extensions:
        - conversion_method: Details on nth → 1st order reduction (when supported)
        - symbolic_derivatives: Cached symbolic forms for Jacobian
        - parameter_values: Substituted parameters for reference
    """
    vector_field: VectorField
    dimension: int
    is_autonomous: bool


class SymbolicSystemBuilder:
    """
    Converts symbolic ODE representations to numerical vector fields.
    
    Handles:
    - Parsing various symbolic input formats (str, syp.Expr, lists)
    - Parameter substitution
    - First-order system validation
    - Autonomous vs non-autonomous detection
    - Compilation to efficient numerical functions via syp.lambdify
    """
    
    @staticmethod
    def build_vector_field(
        equations: SymbolicODE,
        variables: List[syp.Function],
        parameters: SystemParameters = None
    ) -> SymbolicToVectorFieldResult:
        """
        Convert symbolic ODE system to numerical vector field.
        
        Auto-detects whether system is autonomous by checking if time `t` 
        appears in any derivative after solving for dx/dt.
        
        
        Args:
            equations (SymbolicODE): Symbolic system expressed as d(x_i)/dt - F_i(x, t), i=1,...,n
                -> SymbolicODE = Union[List[syp.Expr], syp.Expr, str, List[str]]
                -> str use is highly experimental and not recommended. 
            variables (List[syp.Function]): List of dependent variables as SymPy Function objects
                -> support for strings may be supported later, but not a pressing matter.
            parameters (SystemParameters): Optional parameter substitution dict 
                -> SystemParameters = Union[Dict[str, float], Dict[syp.Symbol, float]]
            
            
        Returns:
            SymbolicToVectorFieldResult wrapper, which itself contains:
                - vector_field: Callable with appropriate signature
                - dimension: Phase space dimension n
                - is_autonomous: True <==>  partial(vector_field, t) = 0
                
        Raises:
            ValueError: If system is not first-order 
            
            
        Example:
            >>> t = syp.symbols('t')
            >>> x, y = syp.symbols('x y', cls=syp.Function)
            >>> x, y = x(t), y(t)
            >>> eqs = [syp.diff(x, t) - y, syp.diff(y, t) + x]
            >>> result = SymbolicSystemBuilder.build_vector_field(eqs, [x, y])
            >>> result.is_autonomous  # True (no explicit t dependence)
            True
        """
        # Parse symbolic input
        parsed_equations = SymbolicSystemBuilder._parse_symbolic_input(equations)
        
        # Validate first-order system
        if not SymbolicSystemBuilder._is_first_order_system(parsed_equations, variables):
            raise ValueError(
                "Only first-order systems are currently supported. "
                "nth-order systems will be supported in future versions."
            )
        
        # Get dimension
        dimension = len(variables)
        
        # Solve for derivatives: d(x_i)/dt = F_i(x, t)
        t_sym = syp.symbols('t')
        derivatives = []
        for i, equation in enumerate(parsed_equations):
            derivative = syp.solve(equation, syp.diff(variables[i], t_sym))[0]
            derivatives.append(derivative)
        
        # Normalise parameters and substitute
        normalised_parameters = SymbolicSystemBuilder._normalise_parameters(parameters)
        
        # Convert Function objects to symbols for substitution
        state_names = [var.func.__name__ for var in variables]
        state_symbols = syp.symbols(' '.join(state_names))
        if dimension == 1:
            state_symbols = (state_symbols,)  # Ensure tuple
        
        # Build function → symbol mapping
        if dimension == 1:
            function_to_symbol_map = {variables[0]: state_symbols[0]}
        else:
            function_to_symbol_map = {var: state_symbols[i] for i, var in enumerate(variables)}
        
        # Apply all substitutions
        all_substitutions = {**function_to_symbol_map, **normalised_parameters}
        derivatives_substituted = [der.subs(all_substitutions) for der in derivatives]
        
        # Detect autonomy
        is_autonomous = SymbolicSystemBuilder._detect_autonomy(derivatives_substituted, t_sym)
        
        # Build appropriate vector field
        if is_autonomous:
            vector_field = SymbolicSystemBuilder._build_autonomous_field(
                derivatives_substituted, state_symbols, dimension
            )
        else:
            vector_field = SymbolicSystemBuilder._build_nonautonomous_field(
                derivatives_substituted, state_symbols, t_sym, dimension
            )
        
        return SymbolicToVectorFieldResult(
            vector_field=vector_field,
            dimension=dimension,
            is_autonomous=is_autonomous
        )
    
    @staticmethod
    def _parse_symbolic_input(symbolic_system: SymbolicODE) -> List[syp.Expr]:
        """
        Parse various symbolic input formats into list of expressions.
        
        Accepts:
        - Single string: "d(x)/dt - f(x, t)"
        - Single expression: syp.Expr object
        - List of strings: ["d(x)/dt - f1", "d(y)/dt - f2"]
        - List of expressions: [syp.Expr, syp.Expr, ...]
        
        Returns:
            List of sympy expressions
        """
        if isinstance(symbolic_system, str):
            # Parse string representation
            return [syp.sympify(symbolic_system)]
        elif isinstance(symbolic_system, syp.Expr):
            # Single expression
            return [symbolic_system]
        elif isinstance(symbolic_system, list):
            # List of expressions or strings
            return [syp.sympify(expr) if isinstance(expr, str) else expr for expr in symbolic_system]
        else:
            raise TypeError(
                f"Unsupported symbolic system type: {type(symbolic_system)}. "
                f"Expected str, syp.Expr, or list of these."
            )
    
    @staticmethod
    def _normalise_parameters(parameters: SystemParameters) -> Dict[syp.Symbol, float]:
        """
        Convert parameter dict to syp.subs() compatible format.
        
        Handles both string keys and Symbol keys, converting to Symbol keys
        for consistency with SymPy's subs() method.
        
        Args:
            parameters: Dict with str or syp.Symbol keys, float values
            
        Returns:
            Dict with syp.Symbol keys, float values
        """
        if not parameters:
            return {}
        
        first_key = next(iter(parameters.keys()))
        if isinstance(first_key, str):
            # Convert string keys to symbols
            return {syp.Symbol(key): value for key, value in parameters.items()}
        else:
            # Already symbols
            return parameters
    
    @staticmethod
    def _is_first_order_system(
        equations: List[syp.Expr],
        variables: List[syp.Function]
    ) -> bool:
        """
        Validate system is first-order.
        
        Checks if any equation contains derivatives of order ≥ 2.
        
        NOTE: Implementation checks up to 100th order. Not exhaustive but
        catches all practical cases. Higher-order systems are rare and would
        likely cause performance issues before reaching order 100.
        
        Args:
            equations: List of symbolic expressions
            variables: List of dependent variables
            
        Returns:
            bool: True if system is first-order, False otherwise
        """
        t_sym = syp.symbols('t')
        # Check for higher-order derivatives
        for equation in equations:
            for variable in variables:
                for order in range(2, 100):  # Check up to 99th order
                    if equation.has(syp.diff(variable, t_sym, order)):
                        return False
        
        return True
    
    @staticmethod
    def _detect_autonomy(
        derivatives: List[syp.Expr],
        t_symbol: syp.Symbol
    ) -> bool:
        """
        Detect if system is autonomous.
        
        After solving for dx_i/dt = F_i(x, t), checks if time symbol `t`
        appears in any F_i. If not, system is autonomous (∂F/∂t ≡ 0).
        
        Args:
            derivatives: List of solved derivatives [F_1, F_2, ..., F_n]
            t_symbol: Time symbol to check for
            
        Returns:
            bool: True if autonomous (no t dependence), False otherwise
            
        Example:
            >>> t = syp.symbols('t')
            >>> x, y = syp.symbols('x y')
            >>> derivatives = [y, -x]  # Harmonic oscillator: autonomous
            >>> SymbolicSystemBuilder._detect_autonomy(derivatives, t)
            True
            >>> derivatives = [y, -x + syp.sin(t)]  # Driven: non-autonomous
            >>> SymbolicSystemBuilder._detect_autonomy(derivatives, t)
            False
        """
        return not any(der.has(t_symbol) for der in derivatives)
    
    @staticmethod
    def _build_autonomous_field(
        derivatives: List[syp.Expr],
        state_symbols: Tuple[syp.Symbol, ...],
        dimension: int
    ) -> AutonomousVectorField:
        """
        Compile autonomous vector field: F(x) -> dx/dt.
        
        Uses syp.lambdify with signature (x_1, ..., x_n) -> [F_1, ..., F_n].
        Time parameter is excluded from lambdified function signature.
        
        Args:
            derivatives: Solved derivatives [F_1, ..., F_n] with no time dependence
            state_symbols: State variable symbols (x_1, ..., x_n)
            dimension: Phase space dimension n
            
        Returns:
            Autonomous vector field function F: R^n → R^n
        """
        # Lambdify each derivative component
        vector_field_funcs = [
            syp.lambdify(state_symbols, der, 'numpy') for der in derivatives
        ]
        
        def vector_field(state: NDArray[np.float64]) -> NDArray[np.float64]:
            """
            Numerical autonomous vector field function.
            
            Args:
                state: Current state vector [x_1, x_2, ..., x_n]
                
            Returns:
                Derivative vector [dx_1/dt, dx_2/dt, ..., dx_n/dt]
            """
            args = tuple(state)
            
            derivatives_evaluated = np.array([
                func(*args) for func in vector_field_funcs
            ], dtype=np.float64)
            
            return derivatives_evaluated
        
        return vector_field
    
    @staticmethod
    def _build_nonautonomous_field(
        derivatives: List[syp.Expr],
        state_symbols: Tuple[syp.Symbol, ...],
        t_symbol: syp.Symbol,
        dimension: int
    ) -> NonAutonomousVectorField:
        """
        Compile non-autonomous vector field: F(x, t) -> dx/dt.
        
        Uses syp.lambdify with signature (x_1, ..., x_n, t) -> [F_1, ..., F_n].
        Time parameter is included in lambdified function signature.
        
        Args:
            derivatives: Solved derivatives [F_1, ..., F_n] possibly depending on t
            state_symbols: State variable symbols (x_1, ..., x_n)
            t_symbol: Time symbol t
            dimension: Phase space dimension n
            
        Returns:
            Non-autonomous vector field function F: R^n x R → R^n
        """
        # Build lambda signature: (x_1, ..., x_n, t)
        if dimension == 1:
            state_vars = (state_symbols[0],) + (t_symbol,)
        else:
            state_vars = state_symbols + (t_symbol,)
        
        # Lambdify each derivative component with time
        vector_field_funcs = [
            syp.lambdify(state_vars, der, 'numpy') for der in derivatives
        ]
        
        def vector_field(state: NDArray[np.float64], time: float) -> NDArray[np.float64]:
            """
            Numerical non-autonomous vector field function.
            
            Args:
                state: Current state vector [x1, x2, ..., xn]
                time: Current time t
                
            Returns:
                Derivative vector [dx1/dt, dx2/dt, ..., dxn/dt]
            """
            args = tuple(state) + (time,)
            
            derivatives_evaluated = np.array([
                func(*args) for func in vector_field_funcs
            ], dtype=np.float64)
            
            return derivatives_evaluated
        
        return vector_field