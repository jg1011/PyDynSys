# TimeHorizon

The `TimeHorizon` class provides a flexible representation of time horizons (subsets of $\mathbb{R}$) with both symbolic and callable constraint representations.

## Example Usage

See the [Phase Spaces & Time Horizons example notebook](../../examples/phase_spaces_and_time_horizons.ipynb) for practical usage examples.

## Factory Methods

### `real_line()`
Creates a time horizon representing $\mathbb{R}$ (entire real line). This is the default time horizon for non-autonomous systems.

### `closed_interval(t_min, t_max)`
Creates a closed interval time horizon $[t_{\min}, t_{\max}]$.

### `open_interval(t_min, t_max)`
Creates an open interval time horizon $(t_{\min}, t_{\max})$.

## Exposed Methods

### contains_time(t)
Verifies whether the time point $t \in \mathbb{R}$ is in the time horizon.

### contains_times(t)
Verifies whether each time $t \in T \subset \mathbb{R}$ is in the time horizon.

## Properties 

### length
Returns either an analytic length (via the Lebesgue measure) of the time horizon, if available, or a numerically estimated length. Utilises caching to avoid consistent recomputation.

**Note:** Currently not implemented - deferred to future versions.

## Dunder Methods 

We've implemented, thus far, 

- ```__str__```
- ```__repr__```
  - The dunders below are experimental. They invoke `sympy` subset logic, which is frail at best, and automatically fall back to false if `sympy` cannot determine the relevant relationships. As such, a false result should not be seen as a certainty, whereas a true result can be. 
- ```__eq__```
- ```__ne__```
- ```__le__```
- ```__lt__```
- ```__ge__```
- ```__gt__```

## Full Docs

::: PyDynSys.core.euclidean.time_horizon.TimeHorizon
    options:
      show_root_heading: true
      show_source: true
      show_root_toc_entry: true
      show_signature_annotations: true
      show_submodules: false
