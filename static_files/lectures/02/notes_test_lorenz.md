---
title: "The Lorenz System: Chaos in Action"
layout: note
permalink: /static_files/lectures/00/test_lorenz/
---


This notebook demonstrates the famous Lorenz system - a simple set of differential equations that exhibits chaotic behavior.

## The Story

In 1963, Edward Lorenz was running weather simulations on an early computer. To save time, he restarted a run from the middle, typing in numbers from a printout (3 decimal places instead of 6 stored internally). The trajectory diverged completely - leading to the discovery of what we now call **chaos**.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(state, t):
    """The Lorenz system of ODEs."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]
```

## The Equations

The Lorenz system is defined by three coupled ordinary differential equations:

$$\frac{dx}{dt} = \sigma(y - x)$$

$$\frac{dy}{dt} = x(\rho - z) - y$$

$$\frac{dz}{dt} = xy - \beta z$$

With the classic parameters $\sigma = 10$, $\rho = 28$, $\beta = 8/3$.


```python
# Simulate the system
t = np.linspace(0, 50, 10000)
initial_state = [1.0, 1.0, 1.0]

# Solve ODE
solution = odeint(lorenz, initial_state, t)

# Extract components
x, y, z = solution.T
```


```python
# Plot the famous butterfly attractor
fig = plt.figure(figsize=(12, 5))

# 3D phase space
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z, lw=0.5, color='steelblue')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Lorenz Attractor (3D Phase Space)')

# Time series
ax2 = fig.add_subplot(122)
ax2.plot(t[:2000], x[:2000], lw=0.8)
ax2.set_xlabel('Time')
ax2.set_ylabel('x(t)')
ax2.set_title('X component over time')

plt.tight_layout()
plt.show()
```

## Sensitivity to Initial Conditions

The hallmark of chaos: tiny differences in initial conditions lead to completely different trajectories. Let's demonstrate:


```python
# Two trajectories with slightly different initial conditions
ic1 = [1.0, 1.0, 1.0]
ic2 = [1.0 + 1e-10, 1.0, 1.0]  # Tiny perturbation!

sol1 = odeint(lorenz, ic1, t)
sol2 = odeint(lorenz, ic2, t)

# Plot divergence
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(t, sol1[:, 0], label='Original', lw=0.8)
axes[0].plot(t, sol2[:, 0], label='Perturbed (1e-10)', lw=0.8, alpha=0.7)
axes[0].set_ylabel('x(t)')
axes[0].legend()
axes[0].set_title('Sensitivity to Initial Conditions')

# Difference
diff = np.abs(sol1[:, 0] - sol2[:, 0])
axes[1].semilogy(t, diff + 1e-15)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('|Difference|')
axes[1].set_title('Exponential divergence of trajectories')

plt.tight_layout()
plt.show()
```

## Lesson

> **Deterministic ≠ Predictable**

The Lorenz system is completely deterministic - no randomness. Yet long-term prediction is impossible because any measurement error, no matter how small, grows exponentially.

This is why weather forecasts become unreliable beyond ~10 days, despite deterministic equations.
