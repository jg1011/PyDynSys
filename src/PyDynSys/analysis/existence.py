"""
Existence and uniqueness analysis for Euclidean dynamical systems, via Cauchy-Peano and Picard-Lindelöf theorems respectively.
"""

from tkinter.constants import NONE
import numpy as np
from numpy.typing import NDArray
from PyDynSys.core import EuclideanDS

def check_existence(
    system: EuclideanDS,
    initial_state: NDArray[np.float64],
    local_radius: float = 1.0
) -> NONE:
    """Check Cauchy-Peano conditions via numerical approximations"""
    
    # TODO: Implement
    pass


def check_uniqueness(
    system: EuclideanDS,
    initial_state: NDArray[np.float64]
) -> NONE:
    """Check Picard-Lindelöf conditions"""
    
    # TODO: Implement
    pass