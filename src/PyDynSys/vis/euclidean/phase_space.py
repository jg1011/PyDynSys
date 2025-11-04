"""
Visualization utilities for PhaseSpace objects.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from ...core.euclidean.phase_space import PhaseSpace


def plot_phase_space(
    phase_space: PhaseSpace,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 200,
    ax: Optional[Axes] = None,
    **kwargs
) -> Axes:
    """
    Visualize a PhaseSpace using its constraint function.
    
    This is a general entry point for PhaseSpace visualization. Currently only
    2D phase spaces are supported via plot_planar_phase_space.
    
    Args:
        phase_space (core.euclidean.PhaseSpace): The PhaseSpace to visualize
        xlim (Tuple[float, float]): Tuple (x_min, x_max) for plot bounds
        ylim (Tuple[float, float]): Tuple (y_min, y_max) for plot bounds
        resolution (int): Number of grid points per dimension (default: 200)
        ax (Optional[matplotlib.axes.Axes]): Optional matplotlib Axes to plot on. If None, creates a new figure
        **kwargs (Any): Additional arguments passed to matplotlib's imshow function
        
    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object containing the plot
        
    Raises:
        NotImplementedError: If phase_space.dimension != 2
    """
    if phase_space.dimension == 2:
        return plot_planar_phase_space(
            phase_space=phase_space,
            xlim=xlim,
            ylim=ylim,
            resolution=resolution,
            ax=ax,
            **kwargs
        )
    else:
        raise NotImplementedError(
            f"plot_phase_space only supports 2D PhaseSpaces currently, "
            f"got dimension {phase_space.dimension}. "
            f"Support for dimensions != 2 is planned for future releases."
        )


def plot_planar_phase_space(
    phase_space: PhaseSpace,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 200,
    ax: Optional[Axes] = None,
    **kwargs
) -> Axes:
    """
    Visualize a 2D PhaseSpace using its constraint function.
    
    Creates a grid of points and tests membership using the PhaseSpace's
    constraint callable, then visualizes the result using imshow for efficient
    boolean visualization.
    
    Args:
        phase_space: The PhaseSpace to visualize (must be 2D)
        xlim: Tuple (x_min, x_max) for plot bounds
        ylim: Tuple (y_min, y_max) for plot bounds
        resolution: Number of grid points per dimension (default: 200)
        ax: Optional matplotlib Axes to plot on. If None, creates a new figure
        **kwargs: Additional arguments passed to matplotlib's imshow function
                  (e.g., cmap, alpha, interpolation)
        
    Returns:
        The matplotlib Axes object containing the plot
        
    Raises:
        ValueError: If phase_space.dimension != 2
    """
    if phase_space.dimension != 2:
        raise ValueError(f"plot_phase_space only supports 2D PhaseSpaces, got dimension {phase_space.dimension}")
    
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    membership = np.zeros_like(X, dtype=bool)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]], dtype=np.float64)
            membership[i, j] = phase_space.contains_point(point)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use imshow for simple, fast boolean visualization
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    default_kwargs = {
        'cmap': 'Blues',
        'alpha': 0.5,
        'origin': 'lower',
        'interpolation': 'nearest'
    }
    default_kwargs.update(kwargs)
    
    im = ax.imshow(membership.astype(float), extent=extent, **default_kwargs)
    ax.contour(X, Y, membership.astype(float), levels=[0.5], colors='black', linewidths=2)
    
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_title('Phase Space Visualization')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax

