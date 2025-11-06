"""
Base property registry mixin.

Provides property tracking infrastructure for all systems.
Property mixins inherit from this to enable property management.
"""

from typing import Dict, Set, FrozenSet

# Central registry of all possible system properties
## When adding a new property mixin, add its property name here
_ALL_PROPERTIES: FrozenSet[str] = frozenset({
    'linear',        # System is linear: dx/dt = Ax (or A(t)x + b(t))
    # Future properties (uncomment as implemented):
    # 'hamiltonian',  # System is Hamiltonian: preserves symplectic structure
    # 'planar',       # System is 2D: enables fixed-point classification
    # 'gradient',     # System is gradient: dx/dt = -∇V(x) for potential V(x)
    # 'conservative', # System conserves energy
})


class _PropertyRegistry:
    """
    Base mixin for property tracking.
    
    Enables systems to track which mathematical properties they have
    (linear, hamiltonian, planar, etc.). Properties are tracked as
    boolean flags in a dictionary.
    
    This mixin is inherited by all property mixins and by system ABCs
    to enable property tracking throughout the hierarchy.
    
    Protocol for mixins: \n
    1. Inherit from _PropertyRegistry\n
    2. Call super().__init__() in __init__\n
    3. Set property flag: self.set_property('property_name', True)\n
    4. Validate property constraints (dimension, structure, etc.)
    
    Example:
        >>> class LinearMixin(_PropertyRegistry):
            >>> def __init__(self, *args, matrix, **kwargs):
            >>>     super().__init__(*args, **kwargs)
            >>>     self._A = matrix
            >>>     self.set_property('linear', True)
    """
    
    
    ### --- Constructor --- ###
    
    
    def __init__(self, *args, **kwargs):
        """
        Initialize property registry.
        
        Initializes _properties dict with all known properties set to False.
        Properties are set to True when detected/set by mixins.
        
        Called by subclasses via super().__init__().
        """
        super().__init__(*args, **kwargs)
        
        # Handle multiple inheritance where multiple mixins call super().__init__()
        if not hasattr(self, '_properties'):
            # Initialize all properties to False
            self._properties: Dict[str, bool] = {
                prop: False for prop in _ALL_PROPERTIES
            }
            
            
    ### --- Public Methods --- ###
    
    
    def has_property(self, prop: str) -> bool:
        """
        Check if system has a property.
        
        Args:
            prop (str): Property name (e.g., 'linear', 'hamiltonian', 'planar')
            
        Returns:
            bool: True if property in property dict & true; False otherwise
        """
        return self._properties.get(prop, False)
    
    
    def set_property(self, prop: str, value: bool = True) -> None:
        """
        Set a property flag.
        
        Args:
            prop (str): Property name (must be in _ALL_PROPERTIES)
            value (bool): Property value (default True)
            
        Raises:
            ValueError: If property name is not in the registry of known properties.
                       This helps catch typos and ensures new properties are registered.
        """
        if prop not in _ALL_PROPERTIES:
            raise ValueError(
                f"Unknown property '{prop}'. Known properties are: {_ALL_PROPERTIES}. "
                f"If adding a new property, add it to _ALL_PROPERTIES in base.py"
            )
        self._properties[prop] = value
    
    
    def get_properties(self) -> Set[str]:
        """
        Get all active properties (properties set to True).
        
        Returns:
            Set[str]: Set of property names that are True
        """
        return {k for k, v in self._properties.items() if v}
    
    
    def get_all_possible_properties(self) -> FrozenSet[str]:
        """
        Get all possible properties supported by the library.
        
        Useful for discovery and documentation. Returns all properties
        that could potentially be set, regardless of whether they're
        currently True or False for this system.
        
        Returns:
            FrozenSet[str]: Set of all possible property names
        """
        return _ALL_PROPERTIES

