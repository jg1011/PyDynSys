# Euclidean Systems

This module contains the primary user-facing classes for defining dynamical systems on Euclidean space ($\mathbb{R}^n$).

## Autonomous Systems

::: PyDynSys.core.euclidean.base.autonomous.AutDynSys
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - from_symbolic
        - from_vector_field
        - trajectory
        - evolve
        - vector_field
        - symbolic_vector_field

## Non-Autonomous Systems

::: PyDynSys.core.euclidean.base.non_autonomous.NonAutDynSys
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - from_symbolic
        - from_vector_field
        - trajectory
        - evolve
        - vector_field
        - symbolic_vector_field
        - time_horizon
