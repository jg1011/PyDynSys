"""
Linear system property mixin.

Adds linear system functionality: dx/dt = Ax for matrix A.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .registry import _PropertyRegistry


class LinearMixin(_PropertyRegistry):
    """
    Mixin adding linear system functionality.
    
    Mathematical Property: dx/dt = Ax for matrix A in mat_n(R)
    
    Structure: System matrix A in mat_n(R)
    
    Adds methods: \n    
    - matrix: System matrix A property \n
    - solve_explicitly(): Explicit solution x(t) = exp(At)x0 
    - eigenvalues(): System eigenvalues and eigenvectors\n
    - jacobian(): Constant Jacobian matrix A (independent of x)
    
    Example:
        >>> A = np.array([[-1, 2], [-2, -1]])
        >>> sys = LinearAutonomousEuclideanDS(dimension=2, matrix=A, ...)
        >>> eigenvals, eigenvecs = sys.eigenvalues()
        >>> x_at_t = sys.solve_explicitly(x0, t=5.0)  # Explicit solution
    """
    _eigenvals: NDArray[np.complex128]
    _eigenvecs: NDArray[np.complex128]
    
    
    ### --- Constructor --- ###
    
    
    def __init__(self, matrix: NDArray[np.float64], *args, **kwargs):
        """
        Initialize linear mixin.
        
        Args:
            matrix: System matrix A where dx/dt = Ax
            *args, **kwargs: Passed to super().__init__()
            
        Raises:
            ValueError: If matrix is not square or dimension mismatch
        """
        super().__init__(*args, **kwargs)
        
        # Validate matrix structure
        if matrix.ndim != 2:
            raise ValueError(
                f"Matrix must be 2D, got {matrix.ndim}D array"
            )
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Matrix must be square, got shape {matrix.shape}"
            )
        
        # Validate dimension consistency
        if hasattr(self, 'dimension'):
            if matrix.shape[0] != self.dimension:
                raise ValueError(
                    f"Matrix dimension {matrix.shape[0]} must match "
                    f"system dimension {self.dimension}"
                )
        
        self._A = matrix.astype(np.float64)
        self.set_property('linear', True)

        
    ### --- Properties --- ###
    
    
    @property
    def A(self) -> NDArray[np.float64]:
        """
        Returns a copy of the system matrix A; dx/dt = Ax.
        
        Returns:
            (NDArray[np.float64]): Square matrix of shape (dimension, dimension)
        """
        return self._A.copy()
    
    
    @property 
    def eigenvalues(self) -> NDArray[np.complex128]:
        """
        Compute system eigenvalues and eigenvectors.
        
        Solves: A @ v = λ * v
        
        Returns:
            (eigenvalues, eigenvectors) tuple
            - eigenvalues: Array of complex eigenvalues
            - eigenvectors: Matrix where columns are eigenvectors
        """
        if not hasattr(self, '_eigenvals'):
            self._eigenvals, self._eigenvecs = np.linalg.eig(self._A)
        return self._eigenvals.copy()
    
    
    @property 
    def eigenvalues(self) -> NDArray[np.complex128]:
        """
        Compute system eigenvalues and eigenvectors.
        
        Solves: A @ v = λ * v
        
        Returns:
            (eigenvalues, eigenvectors) tuple
            - eigenvalues: Array of complex eigenvalues
            - eigenvectors: Matrix where columns are eigenvectors
        """
        if not hasattr(self, '_eigenvals'):
            self._eigenvals, self._eigenvecs = np.linalg.eig(self._A)
        return self._eigenvals.copy()
    
    
    ### --- Public Methods --- ###
    
    
    def solve_explicitly(
        self, 
        x0: NDArray[np.float64], 
        t: float
    ) -> NDArray[np.float64]:
        """
        Explicit solution: x(t) = exp(At) x0.
        
        Uses matrix exponential (much faster than numerical integration
        for linear systems).
        
        Args:
            x0: Initial condition
            t: Time
            
        Returns:
            State at time t
            
        Raises:
            ValueError: If x0 dimension doesn't match system dimension
        """
        if hasattr(self, 'dimension'):
            if x0.shape != (self.dimension,):
                raise ValueError(
                    f"Initial state dimension {x0.shape} must match "
                    f"system dimension {self.dimension}"
                )
        
        # Matrix exponential: exp(At)
        exp_At = expm(self._A * t)
        
        # Matrix-vector product
        return exp_At @ x0
    
    
    def jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Jacobian matrix (constant for linear systems).
        
        For linear systems: DF(x) = A (independent of x)
        
        Args:
            x: State (unused for linear systems, included for interface consistency)
            
        Returns:
            Constant Jacobian matrix A
        """
        return self._A.copy()

