# PhaseSpace

The `PhaseSpace` class provides a flexible representation of phase spaces (subsets of $\mathbb{R}^n$) with both symbolic and callable constraint representations.

## Example Usage

See the [Phase Spaces & Time Horizons example notebook](../../examples/phase_spaces_and_time_horizons.ipynb) for practical usage examples.

## Factory Methods

### `full(dimension)`
Creates a phase space representing R^n (full Euclidean space).

### `box(bounds)`
Creates a box-constrained phase space $[a_1, b_1] \times \cdots \times [a_n, b_n]$.

### `closed_hypersphere(center, radius)`
Creates a closed hypersphere $\{{\bf x} \in \mathbb{R}^n \; : \; \lVert {\bf x} - {\bf c} \rVert \leq r\}$.

### `open_hypersphere(center, radius)`
Creates an open hypersphere $\{{\bf x} \in \mathbb{R}^n \; : \; \lVert {\bf x} - {\bf c} \rVert < r\}$.

## Exposed Methods

### contains_point(x)
Verifies whether the point $x \in \mathbb{R}^n$ is in the phase space.

### contains_points(A)
Verifies whether each point $x \in A \subset \mathbb{R}^n$ is in the phase space.

## Properties 

### volume
Returns either an analytic volume (via the Lebesgue measure) of the phase space, if available, or a numerically estimated volume by considering the volume of a convex hull. Utilises caching to avoid consistent recomputation.

NOTE: Currently not implemented - deferred to future versions.

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

::: PyDynSys.core.euclidean.phase_space.PhaseSpace
    options:
      show_root_heading: true
      show_source: true
      show_root_toc_entry: true
      show_signature_annotations: true
      show_submodules: false

