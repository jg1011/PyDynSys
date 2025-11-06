# Quickstart: The Lorenz Attractor

This guide will walk you through a classic example: defining the Lorenz system, solving for a trajectory, and plotting the famous chaotic attractor. This entire process takes just a few lines of `PyDynSys` code.

## 1. Define the System Symbolically

First, we define the system's equations of motion using `sympy`. `PyDynSys` uses these symbolic expressions to build a fast, callable numerical function for the solver.

```python
import sympy as syp

t = syp.Symbol('t')
x, y, z = syp.symbols('x y z', cls=syp.Function)

lorenz_eq = [
    syp.Eq(x(t).diff(t), 10 * (y(t) - x(t))),
    syp.Eq(y(t).diff(t), x(t) * (28 - z(t)) - y(t)),
    syp.Eq(z(t).diff(t), x(t) * y(t) - (8/3) * z(t))
]
variables = [x(t), y(t), z(t)]
```

## 2. Create a System Instance

With the symbolic equations defined, we use the `AutDynSys.from_symbolic` factory method to create an instance of our dynamical system. Because the equations have no explicit dependence on the time variable `t`, this is an **autonomous** system.

```python
from PyDynSys import AutDynSys

system = AutDynSys.from_symbolic(lorenz_eq, variables)
```

## 3. Solve for a Trajectory

Now we can solve the system. We provide an initial state `x0`, and the time points `t_eval` at which we want to evaluate the solution.

The `t_span` argument tells the underlying ODE solver the interval over which to compute the solution.

```python
import numpy as np

# Define initial conditions and time evaluation points
x0 = np.array([1.0, 1.0, 1.0])
t_eval = np.linspace(0, 40, 5000)

# Solve for the trajectory
traj = system.trajectory(initial_state=x0, t_span=(0, 40), t_eval=t_eval)
```
The result, `traj`, is a `Trajectory` object that contains the solution data.

## 4. Plot the Result

Finally, we can use `matplotlib` to plot the resulting trajectory in 3D phase space. The `traj.y` attribute holds the state vectors, with shape `(dimension, num_time_points)`.

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj.y[0, :], traj.y[1, :], traj.y[2, :], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()
```

This will produce the classic Lorenz attractor image:

![Lorenz Attractor](https://i.imgur.com/b7c42nF.png)
