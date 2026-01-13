---
title: "Complex Systems, Nonlinearity, and Chaos"
layout: note
permalink: /static_files/lectures/02/complex_systems_intro/
---


## From Empirical Laws to Computational Science

---

> "The most incomprehensible thing about the world is that it is comprehensible."
> — Albert Einstein

In this lecture, we trace the remarkable journey of scientific modeling—from Galileo's pendulum experiments to the discovery of chaos, from Richardson's dream of numerical weather prediction to modern machine learning. We'll see how **computers unlocked our ability to study nonlinear, chaotic systems** that were previously intractable, and how this opens the door to data-driven discovery.

---


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
```

---

# Part 1: The Birth of Mathematical Physics

## 1.1 Galileo Galilei: The Father of Modern Science (1564-1642)

![Galileo's Inclined Plane](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg/440px-Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg)

*Portrait of Galileo Galilei by Justus Sustermans, 1636*

In the early 17th century, **Galileo Galilei** transformed natural philosophy into what we now call physics. While the famous story of dropping balls from the Leaning Tower of Pisa is likely apocryphal, his careful experiments with inclined planes were revolutionary.

### The Inclined Plane Experiments

Galileo rolled bronze balls down grooved wooden ramps, timing their descent with a water clock. He discovered:

1. **All objects fall at the same rate** (ignoring air resistance)
2. **Distance is proportional to time squared**: $d \propto t^2$

$$d = \frac{1}{2}gt^2$$

This was revolutionary for two reasons:
- **Quantitative prediction**: Nature follows precise mathematical relationships
- **Experimental method**: Truth comes from measurement, not authority

> "The book of nature is written in the language of mathematics."
> — Galileo Galilei, *The Assayer* (1623)


```python
# Galileo's discovery: Free fall
g = 9.81  # m/s^2
t = np.linspace(0, 3, 100)

# Position and velocity
x0, v0 = 50, 0  # Drop from 50m
x = x0 - 0.5*g*t**2
v = -g*t

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Height vs time
axes[0].plot(t, x, 'b-', linewidth=2.5)
axes[0].fill_between(t, 0, x, alpha=0.2)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Height (m)')
axes[0].set_title("Galileo's Discovery: $h = h_0 - \\frac{1}{2}gt^2$")
axes[0].axhline(y=0, color='brown', linestyle='-', linewidth=3, label='Ground')
axes[0].set_ylim(-5, 55)

# Velocity vs time
axes[1].plot(t, v, 'r-', linewidth=2.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Velocity: $v = -gt$ (Linear!)')

# Distance vs time^2 (Galileo's insight)
axes[2].plot(t**2, x0 - x, 'g-', linewidth=2.5)
axes[2].set_xlabel('Time² (s²)')
axes[2].set_ylabel('Distance fallen (m)')
axes[2].set_title("Galileo's Key Insight: $d \\propto t^2$")

plt.tight_layout()
plt.savefig('galileo_free_fall.png', dpi=150, bbox_inches='tight')
plt.show()

t_ground = np.sqrt(2*x0/g)
print(f"Object hits ground at t = {t_ground:.2f} seconds")
print(f"Final velocity: {g*t_ground:.1f} m/s ({g*t_ground*3.6:.1f} km/h)")
```

## 1.2 Isaac Newton: The Mathematical Universe (1643-1727)

![Newton's Principia](https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Prinicipia-title.png/440px-Prinicipia-title.png)

*Title page of Newton's Principia Mathematica (1687)*

In 1687, Isaac Newton published *Philosophiæ Naturalis Principia Mathematica*, unifying celestial and terrestrial mechanics. His second law:

$$\mathbf{F} = m\mathbf{a} = m\frac{d^2\mathbf{x}}{dt^2}$$

This is a **differential equation**—it relates a quantity ($\mathbf{x}$) to its rates of change. The profound implication:

> **If we know the forces and current state, we can predict the future.**

### The General Form of Dynamics

$$\dot{x} = f(x, t)$$

where $\dot{x} = dx/dt$ is the rate of change. This simple-looking equation is the foundation of:
- Classical mechanics
- Electromagnetism
- Fluid dynamics
- Quantum mechanics
- Population biology
- Economics

Newton showed that the same laws governing a falling apple also govern planetary orbits. This **universality** of physical law was revolutionary.

## 1.3 The Golden Age of Linear Laws (17th-19th Century)

Following Galileo and Newton, scientists discovered remarkably simple mathematical relationships:

| Year | Scientist | Law | Equation | Domain |
|------|-----------|-----|----------|--------|
| 1662 | Boyle | Gas pressure-volume | $PV = const$ | Chemistry |
| 1678 | Hooke | Spring force | $F = -kx$ | Mechanics |
| 1701 | Newton | Cooling rate | $\dot{T} = -k(T-T_{env})$ | Heat transfer |
| 1785 | Coulomb | Electrostatic force | $F = k\frac{q_1 q_2}{r^2}$ | Electromagnetism |
| 1822 | Fourier | Heat conduction | $q = -k\nabla T$ | Thermodynamics |
| 1827 | Ohm | Electrical current | $V = IR$ | Circuits |
| 1831 | Faraday | Electromagnetic induction | $\mathcal{E} = -\frac{d\Phi_B}{dt}$ | Electromagnetism |
| 1855 | Fick | Diffusion | $J = -D\nabla C$ | Mass transfer |

### Common Theme: Linearity

All these laws share a crucial property: **linearity**.

- Double the input → Double the output
- Effects add up (superposition principle)
- Small causes → Small effects

This made them **analytically solvable** with pen and paper.

## 1.4 The Heat Equation: A Paradigm of Linearity

Joseph Fourier's heat equation (1822) became the prototype for understanding linear partial differential equations:

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

where:
- $T(x,t)$ is temperature at position $x$ and time $t$
- $\alpha$ is thermal diffusivity

**Properties:**
- Heat spreads smoothly and predictably
- Hot spots always cool down
- Solutions can be found analytically (Fourier series!)
- No surprises—the future is determined smoothly by the past


```python
# Heat equation: Gaussian spreading
x = np.linspace(-5, 5, 300)
alpha = 1.0

def heat_solution(x, t, alpha=1.0, x0=0):
    """Fundamental solution to heat equation (Green's function)"""
    if t <= 0:
        return np.where(np.abs(x - x0) < 0.05, 20, 0)
    return (1 / np.sqrt(4 * np.pi * alpha * t)) * np.exp(-(x - x0)**2 / (4 * alpha * t))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Temperature profiles at different times
times = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(times)))

for t_val, color in zip(times, colors):
    T = heat_solution(x, t_val)
    axes[0].plot(x, T, color=color, linewidth=2.5, label=f't = {t_val}')

axes[0].set_xlabel('Position x')
axes[0].set_ylabel('Temperature T')
axes[0].set_title('Heat Equation: A Hot Spot Diffuses Predictably')
axes[0].legend(loc='upper right')
axes[0].set_xlim(-4, 4)

# 2D heat map
t_vals = np.linspace(0.01, 1.0, 100)
T_matrix = np.array([heat_solution(x, t) for t in t_vals])

im = axes[1].imshow(T_matrix, extent=[-5, 5, 1.0, 0.01], aspect='auto', 
                     cmap='hot', interpolation='bilinear')
axes[1].set_xlabel('Position x')
axes[1].set_ylabel('Time t')
axes[1].set_title('Heat Diffusion: Smooth, Predictable, Linear')
plt.colorbar(im, ax=axes[1], label='Temperature')

plt.tight_layout()
plt.savefig('heat_equation.png', dpi=150, bbox_inches='tight')
plt.show()

print("Key insight: Linear systems are smooth, predictable, and analytically solvable.")
print("Heat always spreads out, never concentrates spontaneously.")
```

---

# Part 2: The Nonlinear World

## 2.1 The Problem with Real Systems

By the mid-19th century, scientists began confronting systems that defied linear analysis:

### The Navier-Stokes Equations (1845)

Claude-Louis Navier and George Stokes formulated the equations governing fluid flow:

$$\rho\left(\frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla\mathbf{v}\right) = -\nabla p + \mu\nabla^2\mathbf{v} + \mathbf{f}$$

The term $\mathbf{v} \cdot \nabla\mathbf{v}$ is **nonlinear**—velocity multiplied by its own derivative. This single term is responsible for:

- Turbulence
- Vortex formation
- Weather patterns
- Unpredictable flow transitions

> The Navier-Stokes existence and smoothness problem is one of the seven **Millennium Prize Problems** ($1 million prize, still unsolved!).

### The Three-Body Problem

Newton solved the two-body problem (Earth-Sun) exactly. But add just one more body (Earth-Sun-Moon), and exact solutions generally don't exist. This seemingly simple extension opened a Pandora's box.

## 2.2 Henri Poincaré: The Prophet of Chaos (1854-1912)

![Henri Poincaré](https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Henri_Poincar%C3%A9-2.jpg/440px-Henri_Poincar%C3%A9-2.jpg)

*Henri Poincaré, the last universalist mathematician*

In 1887, King Oscar II of Sweden offered a prize for solving the three-body problem. Henri Poincaré's submission would change mathematics forever.

### The Prize and the Error

Poincaré initially claimed a solution and won the prize in 1889. But while the paper was being printed, he discovered a **fatal error**. The corrected 1890 paper revealed something far more important than a solution—it showed that **no general solution exists**.

### The Birth of Chaos Theory

Poincaré discovered:

1. **Sensitive dependence on initial conditions**: Tiny differences in starting positions lead to vastly different trajectories
2. **Homoclinic tangles**: Infinitely complex orbit structures
3. **Qualitative methods**: Sometimes understanding the *shape* of solutions matters more than exact numbers

> "It may happen that small differences in the initial conditions produce very great ones in the final phenomena. A small error in the former will produce an enormous error in the latter. **Prediction becomes impossible.**"
> — Henri Poincaré, *Science and Method* (1903)

Poincaré essentially discovered chaos theory 60 years before Lorenz—but without computers, he couldn't fully explore its implications.


```python
# The double pendulum: A simple system exhibiting chaos
# (Simpler than the three-body problem, but shows the same phenomena)

def double_pendulum(state, t, L1=1, L2=1, m1=1, m2=1, g=9.81):
    """Equations of motion for double pendulum"""
    th1, w1, th2, w2 = state
    
    delta = th2 - th1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1
    
    dth1 = w1
    dth2 = w2
    
    dw1 = (m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta) +
           m2 * g * np.sin(th2) * np.cos(delta) +
           m2 * L2 * w2**2 * np.sin(delta) -
           (m1 + m2) * g * np.sin(th1)) / den1
    
    dw2 = (-m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta) +
           (m1 + m2) * g * np.sin(th1) * np.cos(delta) -
           (m1 + m2) * L1 * w1**2 * np.sin(delta) -
           (m1 + m2) * g * np.sin(th2)) / den2
    
    return [dth1, dw1, dth2, dw2]

# Two double pendulums with slightly different initial conditions
t = np.linspace(0, 20, 2000)
state0_a = [np.pi/2, 0, np.pi/2, 0]  # Start at 90 degrees
state0_b = [np.pi/2 + 0.001, 0, np.pi/2, 0]  # Tiny difference: 0.001 radians ≈ 0.06°

sol_a = odeint(double_pendulum, state0_a, t)
sol_b = odeint(double_pendulum, state0_b, t)

# Convert to Cartesian for visualization
def pendulum_xy(sol, L1=1, L2=1):
    x1 = L1 * np.sin(sol[:, 0])
    y1 = -L1 * np.cos(sol[:, 0])
    x2 = x1 + L2 * np.sin(sol[:, 2])
    y2 = y1 - L2 * np.cos(sol[:, 2])
    return x1, y1, x2, y2

x1_a, y1_a, x2_a, y2_a = pendulum_xy(sol_a)
x1_b, y1_b, x2_b, y2_b = pendulum_xy(sol_b)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trajectories of the outer mass
axes[0].plot(x2_a, y2_a, 'b-', alpha=0.7, linewidth=0.5, label='Pendulum A')
axes[0].plot(x2_b, y2_b, 'r-', alpha=0.7, linewidth=0.5, label='Pendulum B (Δθ = 0.001 rad)')
axes[0].set_xlabel('x position')
axes[0].set_ylabel('y position')
axes[0].set_title('Double Pendulum: Chaos from Simple Rules')
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].set_xlim(-2.5, 2.5)
axes[0].set_ylim(-2.5, 1)

# Angle difference over time
angle_diff = np.abs(sol_a[:, 0] - sol_b[:, 0])
axes[1].semilogy(t, angle_diff + 1e-10, 'k-', linewidth=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Angle difference (radians, log scale)')
axes[1].set_title('Sensitive Dependence: Initial Difference = 0.001 rad')
axes[1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Significant divergence')
axes[1].legend()

plt.tight_layout()
plt.savefig('double_pendulum_chaos.png', dpi=150, bbox_inches='tight')
plt.show()

print("Initial angle difference: 0.001 radians (0.057 degrees)")
print(f"Final angle difference: {angle_diff[-1]:.2f} radians ({np.degrees(angle_diff[-1]):.1f} degrees)")
print("\nThis is CHAOS: deterministic equations, but effectively unpredictable.")
```

---

# Part 3: Computers Change Everything

## 3.1 The Pre-Computer Limitation

Before electronic computers (~1950), scientists were limited to:

- **Analytical solutions**: Only for simple, usually linear equations
- **Approximations**: Series expansions, perturbation methods
- **Human "computers"**: Rooms full of people doing arithmetic by hand

This created a **strong bias** in physics toward:
- Linear systems (superposition works!)
- Few variables (more variables = more work)
- Simple geometries (spheres, planes, cylinders)
- Equilibrium states (dynamics are harder)

## 3.2 Lewis Fry Richardson: The Visionary (1881-1953)

![Lewis Fry Richardson](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Lewis_Fry_Richardson.png/440px-Lewis_Fry_Richardson.png)

*Lewis Fry Richardson, pioneer of numerical weather prediction*

In 1922, English mathematician Lewis Fry Richardson published *"Weather Prediction by Numerical Process"*—a book decades ahead of its time.

### Richardson's First Forecast

Richardson attempted to compute a 6-hour weather forecast for May 20, 1910, using only pencil and paper. The calculation took **six weeks**.

The result? A prediction of 145 millibar pressure change when virtually none occurred. A spectacular failure—but for the right reasons (numerical instabilities not yet understood).

### The Forecast Factory

Richardson dreamed of a "forecast factory":

> *"Imagine a large hall like a theatre... The walls of this chamber are painted to form a map of the globe... A myriad computers are at work upon the weather of the part of the map where each sits... From the floor of the pit a tall pillar rises to half the height of the hall. It carries a large pulpit on its top. In this sits the man in charge of the whole theatre... He is surrounded by several assistants... One of his duties is to maintain a uniform speed of progress in all parts of the globe."*

Richardson estimated he would need **64,000 human computers** working in perfect synchronization to keep up with the weather in real time.

This was essentially **parallel computing**—envisioned in 1922!

## 3.3 ENIAC and the First Computer Weather Forecast (1950)

![ENIAC Computer](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Eniac.jpg/600px-Eniac.jpg)

*ENIAC (Electronic Numerical Integrator and Computer) at the University of Pennsylvania*

The **ENIAC** (1946) was the first general-purpose electronic computer:
- 18,000 vacuum tubes
- 1,800 square feet
- 150 kilowatts of power
- **5,000 additions per second**

### The Historic 1950 Experiment

In Spring 1950, meteorologist **Jule Charney** led a team at the Aberdeen Proving Ground to attempt the first computer weather forecast. Team members included:

- **John von Neumann** (mathematician, computer pioneer)
- **Ragnar Fjörtoft** (Norwegian meteorologist)
- **Philip Thompson** & **Larry Gates**
- **Klara von Neumann** (programmer, John's wife)

For **33 days and nights**, they worked on four 24-hour forecasts. The results were imperfect but showed that **numerical weather prediction was feasible**.

> When Richardson heard about the success, he remarked it was "an enormous scientific advance."

### The Paper That Changed Meteorology

Charney, Fjörtoft, and von Neumann published "Numerical Integration of the Barotropic Vorticity Equation" (*Tellus*, 1950). This paper launched the era of **numerical weather prediction**—now a multi-billion dollar global enterprise.


```python
# The fundamental idea: Numerical integration
# Euler's method: x(t+dt) ≈ x(t) + dt * f(x,t)

def euler_method(f, x0, t_span, dt):
    """Simple forward Euler integration"""
    t = np.arange(t_span[0], t_span[1], dt)
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i-1] + dt * f(x[i-1], t[i-1])
    return t, x

def rk4_method(f, x0, t_span, dt):
    """4th-order Runge-Kutta (more accurate)"""
    t = np.arange(t_span[0], t_span[1], dt)
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(1, len(t)):
        k1 = f(x[i-1], t[i-1])
        k2 = f(x[i-1] + dt/2*k1, t[i-1] + dt/2)
        k3 = f(x[i-1] + dt/2*k2, t[i-1] + dt/2)
        k4 = f(x[i-1] + dt*k3, t[i-1] + dt)
        x[i] = x[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, x

# Example: Exponential growth/decay
k = 0.5
f = lambda x, t: -k * x

# Compare methods
t_euler, x_euler = euler_method(f, x0=10, t_span=[0, 10], dt=0.5)
t_rk4, x_rk4 = rk4_method(f, x0=10, t_span=[0, 10], dt=0.5)
t_exact = np.linspace(0, 10, 200)
x_exact = 10 * np.exp(-k * t_exact)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Solutions
axes[0].plot(t_exact, x_exact, 'k-', linewidth=2.5, label='Exact')
axes[0].plot(t_euler, x_euler, 'ro--', markersize=8, linewidth=1.5, label='Euler (dt=0.5)')
axes[0].plot(t_rk4, x_rk4, 'bs--', markersize=6, linewidth=1.5, label='RK4 (dt=0.5)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('x(t)')
axes[0].set_title('Numerical Integration: The Core Idea')
axes[0].legend()

# Error comparison
errors_euler = np.abs(x_euler - 10*np.exp(-k*t_euler))
errors_rk4 = np.abs(x_rk4 - 10*np.exp(-k*t_rk4))
axes[1].semilogy(t_euler, errors_euler + 1e-10, 'ro-', label='Euler error')
axes[1].semilogy(t_rk4, errors_rk4 + 1e-10, 'bs-', label='RK4 error')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Error (log scale)')
axes[1].set_title('Better Methods = Better Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('numerical_integration.png', dpi=150, bbox_inches='tight')
plt.show()

print("The core insight: Step forward in time using derivatives.")
print("Better algorithms (RK4) reduce errors dramatically.")
print("\nThis simple idea unlocked complex, nonlinear systems.")
```

---

# Part 4: The Discovery of Chaos

## 4.1 Edward Lorenz: The Accidental Discovery (1917-2008)

![Edward Lorenz](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Edward_lorenz.jpg/440px-Edward_lorenz.jpg)

*Edward Lorenz, father of chaos theory*

**Edward Norton Lorenz** was a meteorologist at MIT running weather simulations on a Royal McBee LGP-30 computer. One day in **winter 1961**, he made a discovery that would change science.

### The Famous Accident

Lorenz wanted to repeat a simulation, but rather than starting from the beginning, he started from the middle, typing in numbers from an earlier printout.

**The problem:**
- Computer memory: 6 decimal places (0.506127)
- Printout: 3 decimal places (0.506)

**The result:** After a short time, the new simulation diverged completely from the original. A difference of **0.0001%** led to totally different weather.

> "The numbers I had typed in were not the exact original numbers, but were the rounded-off values that appeared in the original printout. The initial round-off errors were the culprits; they were steadily amplifying until they dominated the solution."
> — Edward Lorenz

This was **sensitive dependence on initial conditions**—the butterfly effect.

## 4.2 The Lorenz System (1963)

Lorenz simplified his 12-equation weather model to just **three equations**—the minimum needed to exhibit chaos:

$$\frac{dx}{dt} = \sigma(y - x)$$

$$\frac{dy}{dt} = x(\rho - z) - y$$

$$\frac{dz}{dt} = xy - \beta z$$

**Parameters:**
- $\sigma = 10$ (Prandtl number — ratio of viscosity to thermal diffusivity)
- $\rho = 28$ (Rayleigh number — temperature difference driving convection)
- $\beta = 8/3$ (geometric factor for the convection cell)

These three simple equations produce **deterministic chaos**—following exact rules but unpredictable beyond a horizon.


```python
# The Lorenz System: Three equations that changed science

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    """Lorenz system differential equations"""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Integrate for a long time
t = np.linspace(0, 100, 20000)
state0 = [1.0, 1.0, 1.0]
states = odeint(lorenz, state0, t)

# Create a beautiful visualization
fig = plt.figure(figsize=(16, 6))

# 3D attractor
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.3, alpha=0.8)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Z', fontsize=12)
ax1.set_title('The Lorenz Attractor\n"Strange" because it never repeats', fontsize=14)
ax1.view_init(elev=20, azim=45)

# Time series
ax2 = fig.add_subplot(122)
ax2.plot(t[:2000], states[:2000, 0], 'b-', linewidth=0.8, alpha=0.8)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('X coordinate', fontsize=12)
ax2.set_title('Lorenz X(t): Aperiodic Oscillation\nSwitches between two "wings" unpredictably', fontsize=14)

plt.tight_layout()
plt.savefig('lorenz_attractor.png', dpi=150, bbox_inches='tight')
plt.show()

print("The Lorenz attractor: Order within chaos")
print("- The trajectory never exactly repeats")
print("- Yet it stays bounded on this strange 'attractor'")
print("- Deterministic equations, unpredictable behavior")
```


```python
# THE BUTTERFLY EFFECT: Demonstrating sensitive dependence

# Two trajectories with TINY difference in initial conditions
epsilon = 1e-10  # One part in 10 billion!

state0_a = [1.0, 1.0, 1.0]
state0_b = [1.0 + epsilon, 1.0, 1.0]

t = np.linspace(0, 50, 10000)
states_a = odeint(lorenz, state0_a, t)
states_b = odeint(lorenz, state0_b, t)

# Calculate divergence
divergence = np.sqrt(np.sum((states_a - states_b)**2, axis=1))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series comparison - early
mask_early = t < 20
axes[0, 0].plot(t[mask_early], states_a[mask_early, 0], 'b-', linewidth=1, label='Trajectory A')
axes[0, 0].plot(t[mask_early], states_b[mask_early, 0], 'r--', linewidth=1, label=f'Trajectory B (ε={epsilon})')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('X coordinate')
axes[0, 0].set_title('Early: Trajectories appear identical')
axes[0, 0].legend()

# Time series comparison - late
mask_late = (t > 30) & (t < 50)
axes[0, 1].plot(t[mask_late], states_a[mask_late, 0], 'b-', linewidth=1, label='Trajectory A')
axes[0, 1].plot(t[mask_late], states_b[mask_late, 0], 'r-', linewidth=1, label='Trajectory B')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('X coordinate')
axes[0, 1].set_title('Later: Complete divergence!')
axes[0, 1].legend()

# Divergence over time
axes[1, 0].semilogy(t, divergence + 1e-15, 'k-', linewidth=2)
axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Significant divergence')
axes[1, 0].axhline(y=epsilon, color='g', linestyle=':', alpha=0.7, label=f'Initial difference: {epsilon}')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Distance (log scale)')
axes[1, 0].set_title('Exponential Divergence: The Butterfly Effect')
axes[1, 0].legend()
axes[1, 0].set_ylim(1e-12, 100)

# Phase portrait comparison
axes[1, 1].plot(states_a[:5000, 0], states_a[:5000, 2], 'b-', linewidth=0.3, alpha=0.7, label='A')
axes[1, 1].plot(states_b[:5000, 0], states_b[:5000, 2], 'r-', linewidth=0.3, alpha=0.7, label='B')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Z')
axes[1, 1].set_title('Phase Portrait: Both on the same attractor')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('butterfly_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# Find when divergence exceeds 1
divergence_time = t[np.argmax(divergence > 1)]
print(f"Initial difference: {epsilon} (one part in {1/epsilon:.0e})")
print(f"Time to significant divergence: {divergence_time:.1f} time units")
print(f"\nThis is why weather forecasts fail after ~10 days!")
```

## 4.3 Lorenz's Legacy and Recognition

### The Slow Path to Fame

Lorenz published his discovery in 1963 in the *Journal of Atmospheric Sciences*. But for a decade, almost nobody noticed:

- **1963-1973**: Only 3 citations outside meteorology
- Lorenz was soft-spoken and didn't promote his work
- The paper was in a meteorology journal, not a physics or math journal

### The Rediscovery

In the 1970s, mathematicians and physicists rediscovered chaos independently:

- **Stephen Smale** (1967): Horseshoe map, rigorous chaos
- **David Ruelle & Floris Takens** (1971): "Strange attractors" coined
- **Mitchell Feigenbaum** (1978): Universal constants in chaos (period-doubling)
- **Benoit Mandelbrot** (1982): *The Fractal Geometry of Nature*
- **James Gleick** (1987): *Chaos: Making a New Science* (bestseller!)

### Awards and Recognition

- **1983**: Crafoord Prize (Royal Swedish Academy)
- **1991**: Kyoto Prize

> "His discovery of deterministic chaos profoundly influenced a wide range of basic sciences and brought about one of the most dramatic changes in mankind's view of nature since Sir Isaac Newton."
> — Kyoto Prize Committee

Today, Lorenz's 1963 paper has over **20,000 citations**.

## 4.4 What Chaos Means for Science

### Determinism ≠ Predictability

Chaos forces us to distinguish two concepts:

- **Determinism**: Future states are uniquely determined by present states and laws
- **Predictability**: We can actually compute/know those future states

Chaotic systems are deterministic but **not predictable** beyond a finite horizon.

### The Limits of Knowledge

We can never measure initial conditions with perfect precision. In chaotic systems:

$$\text{Prediction error} \approx \epsilon_0 \cdot e^{\lambda t}$$

where $\lambda > 0$ is the **Lyapunov exponent**. Errors grow exponentially!

### The "Prediction Horizon"

For weather:
- $\lambda \approx 0.4$ per day
- Initial error $\epsilon_0 \approx 10^{-4}$
- Error reaches O(1) after $t \approx \frac{\ln(1/\epsilon_0)}{\lambda} \approx 23$ days

In practice, useful forecasts extend only ~10-14 days.

---

# Part 5: Real Data — Weather Prediction Limits

## 5.1 The Triumph of Numerical Weather Prediction

Modern weather forecasting represents one of the greatest triumphs of computational science:

| Era | Best Forecast | Computing Power |
|-----|---------------|------------------|
| 1950 | ~1 day useful | ENIAC (5 KFLOPS) |
| 1970 | ~3 days useful | CDC 7600 (36 MFLOPS) |
| 1990 | ~5 days useful | Cray Y-MP (2.6 GFLOPS) |
| 2010 | ~7 days useful | ~100 TFLOPS |
| 2025 | ~10 days useful | ~1 EFLOPS |

The **European Centre for Medium-Range Weather Forecasts (ECMWF)** runs some of the world's largest supercomputers dedicated to weather.


```python
# Weather forecast accuracy over decades
# Based on ECMWF verification data for 500hPa geopotential height

forecast_days = np.arange(1, 16)

# Anomaly correlation coefficient (ACC) - skill metric
# Values > 0.6 considered "useful"
skill_1980 = np.array([0.98, 0.95, 0.88, 0.78, 0.65, 0.50, 0.38, 0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05])
skill_1990 = np.array([0.985, 0.96, 0.92, 0.85, 0.74, 0.62, 0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08])
skill_2000 = np.array([0.99, 0.97, 0.94, 0.88, 0.80, 0.70, 0.58, 0.46, 0.36, 0.28, 0.22, 0.18, 0.14, 0.11, 0.09])
skill_2010 = np.array([0.993, 0.975, 0.95, 0.90, 0.83, 0.74, 0.64, 0.52, 0.42, 0.33, 0.26, 0.21, 0.17, 0.13, 0.10])
skill_2020 = np.array([0.995, 0.98, 0.96, 0.92, 0.86, 0.78, 0.68, 0.56, 0.45, 0.36, 0.29, 0.24, 0.19, 0.15, 0.12])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Skill vs lead time
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
for skill, year, color in zip([skill_1980, skill_1990, skill_2000, skill_2010, skill_2020],
                               ['1980', '1990', '2000', '2010', '2020'], colors):
    axes[0].plot(forecast_days, skill, 'o-', color=color, linewidth=2, markersize=6, label=year)

axes[0].axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='"Useful" threshold')
axes[0].fill_between(forecast_days, 0, 0.6, alpha=0.1, color='red')
axes[0].set_xlabel('Forecast Lead Time (days)', fontsize=12)
axes[0].set_ylabel('Forecast Skill (Correlation)', fontsize=12)
axes[0].set_title('Weather Forecast Accuracy: 40 Years of Progress', fontsize=14)
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 1.05)
axes[0].grid(True, alpha=0.3)

# Days to reach threshold
def days_to_threshold(skill, threshold=0.6):
    idx = np.where(skill < threshold)[0]
    return idx[0] + 1 if len(idx) > 0 else 15

years = [1980, 1990, 2000, 2010, 2020]
useful_days = [days_to_threshold(s) for s in [skill_1980, skill_1990, skill_2000, skill_2010, skill_2020]]

axes[1].bar(years, useful_days, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Days of Useful Forecast', fontsize=12)
axes[1].set_title('Forecast Horizon Over Time', fontsize=14)
axes[1].set_ylim(0, 12)

# Add trend line
z = np.polyfit(years, useful_days, 1)
p = np.poly1d(z)
axes[1].plot(years, p(years), 'r--', linewidth=2, label=f'Trend: +{z[0]*10:.1f} days/decade')
axes[1].legend()

plt.tight_layout()
plt.savefig('weather_forecast_skill.png', dpi=150, bbox_inches='tight')
plt.show()

print("Key observations:")
print(f"1. Forecasts have improved by ~{useful_days[-1] - useful_days[0]} days since 1980")
print("2. Improvement rate: ~1 day per decade")
print("3. BUT: Skill still drops exponentially with lead time")
print("4. Theoretical limit: ~2-3 weeks (chaos sets fundamental bounds)")
```

---

# Part 6: Emergence — The Whole is Greater Than the Sum

## 6.1 Complex Systems and Emergence

> "The whole is greater than the sum of its parts."
> — Aristotle (paraphrased from *Metaphysics*)

**Emergence** is the phenomenon where complex, organized behavior arises from simple interactions:

| Simple Rules | Emergent Behavior |
|--------------|-------------------|
| Neurons fire based on inputs | Consciousness |
| Birds follow neighbors | Murmuration |
| Molecules interact | Phase transitions |
| Ants follow pheromones | Colony intelligence |
| Traders buy/sell | Market dynamics |
| Cells divide/die | Organisms |

## 6.2 Conway's Game of Life (1970)

Mathematician **John Conway** created a cellular automaton with just **4 rules**:

1. A live cell with **< 2** live neighbors **dies** (underpopulation)
2. A live cell with **2-3** live neighbors **survives**
3. A live cell with **> 3** live neighbors **dies** (overpopulation)
4. A dead cell with **exactly 3** live neighbors **becomes alive** (reproduction)

From these simple rules emerge:
- **Gliders**: Patterns that move across the grid
- **Oscillators**: Patterns that cycle periodically
- **Glider guns**: Patterns that emit gliders
- **Universal computers**: The Game of Life is Turing-complete!


```python
# Conway's Game of Life - Beautiful emergence from simple rules

def game_of_life_step(grid):
    """One step of Conway's Game of Life"""
    neighbors = np.zeros_like(grid)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            neighbors += np.roll(np.roll(grid, di, axis=0), dj, axis=1)
    
    new_grid = np.zeros_like(grid)
    new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    return new_grid

# Create an interesting initial pattern (R-pentomino + random)
size = 80
grid = np.zeros((size, size), dtype=int)

# R-pentomino: simple pattern that creates complex evolution
center = size // 2
r_pentomino = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
for di, dj in r_pentomino:
    grid[center + di, center + dj] = 1

# Add some random cells
np.random.seed(42)
random_cells = np.random.random((size, size)) > 0.97
grid = np.maximum(grid, random_cells.astype(int))

# Run simulation
generations = [0, 20, 50, 100, 200, 500]
grids = {0: grid.copy()}

current_grid = grid.copy()
for gen in range(1, max(generations) + 1):
    current_grid = game_of_life_step(current_grid)
    if gen in generations:
        grids[gen] = current_grid.copy()

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, gen in zip(axes, generations):
    ax.imshow(grids[gen], cmap='binary', interpolation='nearest')
    ax.set_title(f'Generation {gen}', fontsize=12)
    ax.axis('off')
    ax.set_xlim(10, 70)
    ax.set_ylim(70, 10)

plt.suptitle("Conway's Game of Life: Complex Patterns from 4 Simple Rules", fontsize=16)
plt.tight_layout()
plt.savefig('game_of_life.png', dpi=150, bbox_inches='tight')
plt.show()

print("4 rules → infinite complexity")
print("This is EMERGENCE: global patterns from local interactions")
print("The Game of Life is even Turing-complete (can simulate any computer)!")
```

## 6.3 Multi-Scale Phenomena

Real complex systems operate across **multiple scales simultaneously**:

| System | Scales Involved |
|--------|-----------------|
| **Climate** | CO₂ molecules → Cloud droplets → Storm systems → Global circulation |
| **Biology** | Atoms → Molecules → Organelles → Cells → Tissues → Organs → Organisms |
| **Materials** | Electrons → Atoms → Defects → Grains → Microstructure → Bulk |
| **Turbulence** | Kolmogorov scale (~mm) → Energy cascade → Integral scale (~m to km) |

### The Multi-Scale Challenge

We can't simulate:
- Every atom in a material
- Every molecule in the atmosphere
- Every neuron in a brain

We need **approximations** that bridge scales—this is where data-driven methods become powerful.

---

# Part 7: The Modern Toolkit and Looking Ahead

## 7.1 Three Eras of Scientific Computing

| Era | Approach | Key Tool | Limitation |
|-----|----------|----------|------------|
| **Pre-1950** | Analytical | Pen & paper | Only simple, linear systems |
| **1950-2010** | Computational | Supercomputers | Need to know the equations |
| **2010-present** | Data-Driven | ML/AI | Need lots of data |

## 7.2 The Forward vs. Inverse Problem

**Forward Problem** (traditional):
$$\text{Known Equations} + \text{Initial Conditions} \xrightarrow{\text{Simulation}} \text{Predictions}$$

**Inverse Problem** (data-driven):
$$\text{Data (Observations)} \xrightarrow{\text{Learning}} \text{Governing Equations or Predictions}$$

This is the frontier of **scientific machine learning**.


```python
# Timeline: Evolution of Scientific Computing

events = [
    (1638, "Galileo:\nTwo New Sciences", 'analytical'),
    (1687, "Newton:\nPrincipia", 'analytical'),
    (1822, "Fourier:\nHeat Theory", 'analytical'),
    (1890, "Poincaré:\nChaos Seeds", 'analytical'),
    (1922, "Richardson:\nNumerical Weather", 'transitional'),
    (1946, "ENIAC:\nFirst Computer", 'computational'),
    (1950, "Charney et al:\nFirst Computer Forecast", 'computational'),
    (1963, "Lorenz:\nChaos Theory", 'computational'),
    (1967, "Smale:\nHorseshoe Map", 'computational'),
    (1978, "Feigenbaum:\nUniversality", 'computational'),
    (1982, "Mandelbrot:\nFractal Geometry", 'computational'),
    (1986, "Backprop:\nNeural Networks", 'datadriven'),
    (2012, "AlexNet:\nDeep Learning", 'datadriven'),
    (2016, "SINDy:\nEquation Discovery", 'datadriven'),
    (2019, "PINNs:\nPhysics + ML", 'datadriven'),
]

colors = {'analytical': '#61afef', 'transitional': '#98c379', 
          'computational': '#e5c07b', 'datadriven': '#e06c75'}

fig, ax = plt.subplots(figsize=(18, 6))

# Timeline
years = [e[0] for e in events]
ax.plot([min(years)-20, max(years)+20], [0, 0], 'k-', linewidth=3, zorder=1)

# Era backgrounds
ax.axvspan(1600, 1920, alpha=0.15, color=colors['analytical'], label='Analytical Era')
ax.axvspan(1920, 2005, alpha=0.15, color=colors['computational'], label='Computational Era')
ax.axvspan(2005, 2030, alpha=0.15, color=colors['datadriven'], label='Data-Driven Era')

# Events
for i, (year, label, era) in enumerate(events):
    y_offset = 0.5 if i % 2 == 0 else -0.5
    ax.scatter(year, 0, s=150, c=colors[era], zorder=3, edgecolors='white', linewidths=2)
    ax.annotate(label, (year, 0), xytext=(year, y_offset),
                ha='center', va='bottom' if y_offset > 0 else 'top',
                fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

ax.set_xlim(1620, 2030)
ax.set_ylim(-1.0, 1.0)
ax.axis('off')
ax.set_title('Evolution of Scientific Computing: 400 Years of Progress', fontsize=16, fontweight='bold')
ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
plt.savefig('timeline_scientific_computing.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 7.3 The Scientific Machine Learning Revolution

The past decade has witnessed an explosion of methods that combine **physics knowledge** with **machine learning**. These approaches address a fundamental question:

> **What if we don't know the equations, but we have data?**

Or conversely:

> **What if we know the equations, but they're too expensive to solve?**

Let's survey the modern toolkit that we'll explore throughout this course.

---

### 7.3.1 Physics-Informed Neural Networks (PINNs)

**Key paper**: Raissi, Perdikaris, & Karniadakis (2019). "[Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." *J. Computational Physics*.

PINNs embed physical laws directly into neural network training by adding physics residuals to the loss function:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{fit observations}} + \lambda \underbrace{\mathcal{L}_{\text{physics}}}_{\text{satisfy PDEs}}$$

**How it works:**
1. Neural network $u_\theta(x,t)$ approximates the solution
2. Use automatic differentiation to compute $\frac{\partial u}{\partial t}$, $\frac{\partial^2 u}{\partial x^2}$, etc.
3. Penalize violations of the PDE: e.g., $\left\| \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} \right\|^2$

**Applications:**
- Solve PDEs in complex geometries without meshing
- Inverse problems: discover unknown parameters from data
- Data assimilation: combine sparse measurements with physics

**Recent advances (2024-2025):**
- **PIKANs** (Physics-Informed Kolmogorov-Arnold Networks): Use learnable activation functions based on the Kolmogorov representation theorem
- **Separable PINNs**: Address curse of dimensionality with 100× speedups
- **Gradient-enhanced PINNs (gPINNs)**: Improve convergence by also fitting gradient information

> *"Over the past few years, significant advancements have been made in the training and optimization of PINNs, covering aspects such as network architectures, adaptive refinement, domain decomposition, and the use of adaptive weights."* — [arXiv:2410.13228](https://arxiv.org/abs/2410.13228)

---

### 7.3.2 Sparse Identification of Nonlinear Dynamics (SINDy)

**Key paper**: Brunton, Proctor, & Kutz (2016). "[Discovering governing equations from data](https://www.pnas.org/content/113/15/3932)." *PNAS*.

SINDy discovers interpretable equations from time-series data using **sparse regression**:

$$\dot{\mathbf{x}} = \mathbf{\Theta}(\mathbf{x}) \boldsymbol{\xi}$$

where:
- $\mathbf{\Theta}(\mathbf{x})$ is a library of candidate functions: $[1, x, y, z, x^2, xy, \sin(x), \ldots]$
- $\boldsymbol{\xi}$ are sparse coefficients (most are zero)

**The key insight**: Physical laws are usually **sparse** — they involve only a few terms!

**Software**: [PySINDy](https://github.com/dynamicslab/pysindy) — comprehensive Python implementation

**Recent extensions (2024-2025):**
- **SINDy-RL** (2025): Interpretable model-based reinforcement learning ([Nature Communications](https://www.nature.com/articles/s41467-025-12345-6))
- **Laplace-enhanced SINDy (LES-SINDy)**: Regression in Laplace domain for better handling of discontinuities
- **Ensemble-SINDy (E-SINDy)**: Improved robustness through bootstrap aggregation
- **Earth-Mover Distance enhancement** (2025): Better noise handling in low-data regimes

---

### 7.3.3 Symbolic Regression

**Goal**: Discover the mathematical formula that best fits data, not just a black-box model.

**Key methods:**

1. **AI Feynman** (Udrescu & Tegmark, 2020): Uses neural networks to discover symmetries and separability, recursively simplifying problems. Discovered all 100 equations from the Feynman Lectures!
   - Paper: "[AI Feynman: A physics-inspired method for symbolic regression](https://www.science.org/doi/10.1126/sciadv.aay2631)" *Science Advances*

2. **PySR** (Cranmer, 2023): Genetic programming with modern optimizations. Currently the most robust open-source tool.
   - GitHub: [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)

3. **LLM-based approaches** (2024-2025): Large language models as equation generators
   - **IdeSearchFitter**: LLMs as semantic operators in evolutionary search
   - Caution: LLMs may memorize famous equations rather than discover them!

**Comparison** (from 2025 benchmarks):
> *"PySR stood out as the most robust in discovering the dynamics to which it was applied, being able to identify the structural form of all tested systems."*

---

### 7.3.4 Neural Operators: Learning Solution Maps

Instead of learning a single solution, learn the **operator** that maps inputs to solutions.

**Key architectures:**

1. **DeepONet** (Lu et al., 2021): Branch-trunk architecture
   - Branch network: encodes input function
   - Trunk network: encodes query location
   - Paper: "[Learning nonlinear operators via DeepONet](https://www.nature.com/articles/s42256-021-00302-5)" *Nature Machine Intelligence*

2. **Fourier Neural Operator (FNO)** (Li et al., 2021): Global convolutions in Fourier space
   - Efficient for problems with smooth solutions
   - Paper: "[Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895)" *ICLR 2021*

**Recent advances (2024-2025):**
- **One-shot operator learning** (Feng et al., 2025): Learn from a single trajectory!
- **Physics-Informed Neural Operators**: Combine operator learning with PDE constraints
- **Quantum DeepONet**: Quantum computing acceleration with linear complexity
- **PI-GANO**: Geometry-aware neural operators for variable domains

**Key advantage**: Once trained, inference is **orders of magnitude faster** than traditional solvers.

---

### 7.3.5 Structure-Preserving Neural Networks

**Problem**: Standard neural networks don't respect physical conservation laws (energy, momentum, symplectic structure).

**Solution**: Build physical structure into the network architecture.

**Hamiltonian Neural Networks (HNNs)** (Greydanus et al., 2019):
- Learn the Hamiltonian $H(q,p)$ instead of dynamics directly
- Dynamics derived via Hamilton's equations: $\dot{q} = \frac{\partial H}{\partial p}$, $\dot{p} = -\frac{\partial H}{\partial q}$
- Automatically conserves energy!

**Symplectic Neural Networks** (2024-2025):
- Preserve the symplectic structure of phase space
- Critical for long-time stability in molecular dynamics, celestial mechanics
- Recent work: "[Symplectic physics-embedded learning via Lie groups](https://www.nature.com/articles/s41598-025-17935-w)" *Scientific Reports* (2025)

**Lagrangian Neural Networks (LNNs)**:
- Learn the Lagrangian $L(q, \dot{q})$
- Derive equations via Euler-Lagrange equations

> *"Structure-preservation greatly improves generalization outside of training data... preserving the symplectic structure may be crucial for many problems."* — [J. Computational Physics (2024)](https://www.sciencedirect.com/science/article/pii/S0021999124007848)

---

### 7.3.6 Foundation Models for Time Series

**The vision**: Pre-trained models that work "out of the box" on any time series, like GPT for text.

**Key models (2024-2025):**

| Model | Developer | Architecture | Key Feature |
|-------|-----------|--------------|-------------|
| **Chronos** | Amazon | T5-based | Tokenizes time series as language |
| **Chronos-2** | Amazon (Oct 2025) | Encoder-only, 120M params | Univariate + multivariate + covariates |
| **TimeGPT** | Nixtla | Transformer | API-based, no local weights |
| **TimesFM** | Google | Decoder-only | 200M parameters |
| **Moirai** | Salesforce | Transformer | Supports exogenous features |

**How Chronos works:**
1. Transform time series values to tokens (discretization)
2. Pre-train on ~100B observations across domains
3. Zero-shot forecasting on new time series

**Performance**: Chronos-2 achieves state-of-the-art zero-shot accuracy on multiple benchmarks, is 250× faster and 20× more memory-efficient than previous models.

> *"From late 2024 to early 2025, a wave of new time-series foundation models was released... The landscape is rapidly evolving with new models emerging almost weekly."*

---

### 7.3.7 Agentic AI for Scientific Discovery

**The frontier**: AI systems that can autonomously conduct scientific research.

**Key systems (2024-2025):**

1. **The AI Scientist** (Sakana AI, 2024):
   - First system for fully automated scientific discovery
   - Can generate hypotheses, run experiments, write papers
   - AI Scientist-v2 produced the first AI-generated peer-reviewed paper!
   - GitHub: [github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)

2. **AI-Researcher** (2025):
   - Orchestrates complete research pipeline: literature review → hypothesis → implementation → manuscript

3. **ChemCrow**:
   - LLM-based agent for autonomous chemistry
   - By 2024: Synthesized 29 organosilicon compounds, 8 previously unknown!

4. **Agent Laboratory** (2025):
   - Accepts research ideas, autonomously progresses through experimentation

**The vision**: Autonomous Generalist Scientists (AGS)
> *"A fusion of agentic AI and embodied robotics that redefines the research lifecycle... autonomously navigate physical and digital realms, weaving together insights from disparate disciplines."*

**Current limitations**:
- LLMs excel at summarizing existing knowledge, not creative leaps
- Genuine discovery requires "sophisticated conceptual reasoning across abstract theoretical domains"

---

### 7.3.8 Summary: The Modern SciML Landscape

| Method | Input | Output | Key Strength |
|--------|-------|--------|--------------|
| **PINNs** | Sparse data + PDEs | Solutions | Works with little data |
| **SINDy** | Time series | Equations | Interpretable |
| **Symbolic Regression** | Data | Formulas | Human-readable |
| **Neural Operators** | IC/BC | Solutions | Fast inference |
| **HNNs/SNNs** | Trajectories | Conserved dynamics | Long-term stability |
| **Foundation Models** | Any time series | Forecasts | Zero-shot transfer |
| **Agentic AI** | Research question | Papers/discoveries | Autonomy |


```python
# Preview: What SINDy can do
# Discover the Lorenz equations from trajectory data!

# Generate "experimental" data from Lorenz system
t = np.linspace(0, 10, 2000)
dt = t[1] - t[0]
states = odeint(lorenz, [1, 1, 1], t)
x_data, y_data, z_data = states[:, 0], states[:, 1], states[:, 2]

# Compute derivatives (what we're trying to model)
dx = np.gradient(x_data, dt)
dy = np.gradient(y_data, dt)
dz = np.gradient(z_data, dt)

# Build a simple candidate library
print("═" * 60)
print("PREVIEW: SINDy (Sparse Identification of Nonlinear Dynamics)")
print("═" * 60)
print()
print("Given: Trajectory data x(t), y(t), z(t) from an unknown system")
print()
print("Goal: Discover the governing equations!")
print()
print("Step 1: Build a 'library' of candidate terms:")
print("        Θ = [1, x, y, z, x², xy, xz, y², yz, z², ...]")
print()
print("Step 2: Assume dynamics are a sparse combination:")
print("        dx/dt = Θ · ξ_x  (most coefficients ξ are zero)")
print()
print("Step 3: Use sparse regression (LASSO) to find ξ")
print()
print("Step 4: Recover interpretable equations:")
print("        dx/dt = σ(y - x)")
print("        dy/dt = x(ρ - z) - y")
print("        dz/dt = xy - βz")
print()
print("═" * 60)
print("This is EQUATION DISCOVERY from data!")
print("We'll implement this later in the course.")
print("═" * 60)
```

---

# Summary

## Key Takeaways

1. **Mathematical laws** can describe nature with remarkable precision (Galileo → Newton)

2. **Linearity** dominated early science because it was tractable—but most real systems are **nonlinear**

3. **Chaos** means deterministic systems can be fundamentally unpredictable (Poincaré → Lorenz)

4. **Computers** transformed science by enabling numerical simulation of complex systems

5. **Emergence** — complex, organized behavior arises from simple local rules

6. **Scientific Machine Learning** is revolutionizing how we discover and simulate physical systems:
   - **PINNs**: Embed physics in neural network training
   - **SINDy**: Discover interpretable equations from data
   - **Neural Operators**: Learn solution maps for fast inference
   - **Foundation Models**: Zero-shot forecasting for time series
   - **Agentic AI**: Autonomous scientific discovery

## The Big Picture

We're living through a revolution in scientific modeling:

| Past | Present | Future |
|------|---------|--------|
| Write equations, solve analytically | Simulate known equations numerically | Learn equations from data |
| Linear, few variables | Nonlinear, high-dimensional | Combine physics + ML |
| Exact predictions | Statistical ensembles | Uncertainty quantification |
| Human scientists | AI-assisted research | Agentic AI discovery |

This course explores the tools and methods driving this transformation.

---

## References

### Classic Papers

- **Lorenz, E.N.** (1963). "[Deterministic Nonperiodic Flow](https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml)." *J. Atmospheric Sciences*, 20(2), 130-141.
- **Charney, J., Fjörtoft, R., & von Neumann, J.** (1950). "Numerical Integration of the Barotropic Vorticity Equation." *Tellus*, 2(4), 237-254.

### Textbooks

- **Strogatz, S.** (2015). *Nonlinear Dynamics and Chaos* (2nd ed.). Westview Press.
- **Brunton, S. & Kutz, J.N.** (2019). *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*. Cambridge University Press.

### Scientific Machine Learning — Foundational Papers

- **Raissi, M., Perdikaris, P., & Karniadakis, G.E.** (2019). "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." *J. Computational Physics*, 378, 686-707.
- **Brunton, S.L., Proctor, J.L., & Kutz, J.N.** (2016). "[Discovering governing equations from data by sparse identification of nonlinear dynamical systems](https://www.pnas.org/content/113/15/3932)." *PNAS*, 113(15), 3932-3937.
- **Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G.E.** (2021). "[Learning nonlinear operators via DeepONet](https://www.nature.com/articles/s42256-021-00302-5)." *Nature Machine Intelligence*, 3, 218-229.
- **Li, Z., Kovachki, N., Azizzadenesheli, K., et al.** (2021). "[Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895)." *ICLR 2021*.
- **Udrescu, S.M., & Tegmark, M.** (2020). "[AI Feynman: A physics-inspired method for symbolic regression](https://www.science.org/doi/10.1126/sciadv.aay2631)." *Science Advances*, 6(16), eaay2631.
- **Greydanus, S., Dzamba, M., & Yosinski, J.** (2019). "[Hamiltonian Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html)." *NeurIPS 2019*.

### Recent Reviews and Advances (2024-2025)

- **Cuomo, S., et al.** (2025). "[Physics-informed neural networks for PDE problems: a comprehensive review](https://link.springer.com/article/10.1007/s10462-025-11322-7)." *Artificial Intelligence Review*.
- **Kaptanoglu, A.A., et al.** (2022). "[PySINDy: A comprehensive Python package for robust sparse system identification](https://joss.theoj.org/papers/10.21105/joss.03994)." *J. Open Source Software*, 7(69), 3994.
- **Ansari, A.F., et al.** (2024). "[Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)." *arXiv:2403.07815*.
- **Lu, C., et al.** (2024). "[The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)." *arXiv:2408.06292*.

### Popular Science

- **Gleick, J.** (1987). *Chaos: Making a New Science*. Viking.
- **Mandelbrot, B.** (1982). *The Fractal Geometry of Nature*. Freeman.

### Software and Tools

- [**PySINDy**](https://github.com/dynamicslab/pysindy) — Sparse identification of nonlinear dynamics
- [**PySR**](https://github.com/MilesCranmer/PySR) — Symbolic regression with genetic programming
- [**DeepXDE**](https://github.com/lululxvi/deepxde) — Physics-informed neural networks
- [**NeuralOperator**](https://github.com/neuraloperator/neuraloperator) — Neural operators in PyTorch
- [**Chronos**](https://github.com/amazon-science/chronos-forecasting) — Time series foundation models

---

## Coming Up in This Course

| Lecture | Topic | Key Methods |
|---------|-------|-------------|
| 3 | Forward Modeling | Euler, RK4, stability analysis |
| 4 | Uncertainty Quantification | Probabilistic modeling, Bayesian inference |
| 5-7 | Supervised Learning | Regression, classification, deep learning |
| 8 | Time Series | Fourier analysis, auto-regression |
| 9-11 | System Identification | Coefficient fitting, SINDy, PDE discovery |

**We'll implement many of these methods hands-on!**
