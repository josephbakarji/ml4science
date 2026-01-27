---
title: "Lecture Notes: Scientific Modeling Principles"
layout: note
permalink: /static_files/lectures/04/notes_scientific_modeling_principles/
---


## 1. Introduction: Why Differential Equations?

This lecture covers principles of scientific modeling with the goal of understanding why differential equations are useful. We'll review familiar concepts like Newton's second law and identify general principles that appear repeatedly when implementing scientific machine learning—incorporating physical constraints into ML models.

### The Quest for Generalization

A central theme in both ML and science is **generalization**: building models that work beyond the training data. This naturally leads us to prefer **simple models** because:
- They enable fast inference and quick predictions
- They generalize better beyond what we've observed
- A model that only recalls training data is just a lookup table

### The Strong Inductive Bias: Space and Time

There is a powerful inductive bias in science: **space and time are special variables**. We're typically interested in functions that map points in space $(x, y, z)$ and time $t$ to measured quantities:

$$u: (x, y, z, t) \mapsto \text{measured variable}$$

This is different from the empirical laws we saw earlier (like $V = IR$), which map one measured variable to another at a single instant. Here, we're interested in how things evolve.

To quantify position, you need a **coordinate system**—some fixed reference point. To quantify time, you need a clock. These measurements have **units** (meters, seconds), which will become important later.

---

## 2. The Only Constant is Change

A profound observation: **nothing stays the same in space and time**. Everything changes. Yet in the midst of this change, we search for **laws that remain constant**—and these laws typically describe *how* things change.

### Quantifying Change: From Data to Derivatives

Given discrete measurements $u_i$ at times $t_i$, we can quantify change using **finite differences**:

$$\frac{\Delta u}{\Delta t} = \frac{u_i - u_{i-1}}{t_i - t_{i-1}}$$

This is the **slope** between two data points—an approximation of how fast $u$ is changing.

As we collect more data points (making $\Delta t$ smaller), this approximation improves. In the limit:

$$\frac{\partial u}{\partial t} = \lim_{\Delta t \to 0} \frac{u(t + \Delta t) - u(t)}{\Delta t}$$

This limiting process, formalized by Leibniz and Newton, was controversial for a long time—what does it mean for something to be "infinitely small but not zero"? But it works remarkably well for describing the physical world.

**The key insight**: We go from discrete (data) → continuous (mathematics) → back to discrete (computation). The continuous formulation eliminates dependence on specific grid spacing and reveals mathematical structure.

### Continuous vs. Discrete Systems

There's a fundamental divide:
- **Sciences**: Focus on continuous systems—smooth functions defined everywhere, with well-behaved derivatives
- **Computer Science**: Focus on discrete systems—states, transitions, graphs

These perspectives are connected: the continuous world emerges from taking limits of the discrete world. Both are idealizations of reality.

---

## 3. From Galileo to Newton: The Birth of Dynamics

### Galileo's Discovery

When Galileo dropped objects from the Tower of Pisa (or so the legend goes), he noticed something curious:
- Position $x(t)$ increases
- Velocity $v = \frac{dx}{dt}$ also increases (objects accelerate)
- But the **acceleration** $a = \frac{d^2x}{dt^2}$ is **constant**!

This constant is $g \approx 9.8 \, \text{m/s}^2$—the acceleration due to gravity. It's the same for all objects, everywhere on Earth (approximately).

### Newton's Second Law

This observation generalizes to Newton's second law:

$$F = ma = m\frac{d^2 x}{dt^2}$$

Force equals mass times acceleration. For free fall near Earth's surface: $F = mg$, so:

$$\frac{d^2 x}{dt^2} = g$$

### Solving the Equation

This is a **differential equation**—an equation involving derivatives. To solve it:

1. Integrate once: $\frac{dx}{dt} = gt + v_0$ (velocity)
2. Integrate again: $x = \frac{1}{2}gt^2 + v_0 t + x_0$ (position)

The result is a **parabola**! From a simple differential equation, we derived the shape of a projectile's trajectory.

Add drag (air resistance proportional to velocity), and you get:
$$m\frac{d^2x}{dt^2} = -mg - b\frac{dx}{dt}$$

This introduces exponential decay and modified trajectories.

---

## 4. Conservation Laws: Another Powerful Inductive Bias

### The Basic Principle

Conservation laws are everywhere in physics. The fundamental idea is simple:

$$\text{Flux in} - \text{Flux out} = \text{Accumulation}$$

Think of people in a room: if 10 enter and 3 leave, the number inside increases by 7.

### Mathematical Formulation

Consider a quantity with density $\rho(x,t)$ in a 1D domain. In a small region $[x, x+\Delta x]$:

- **Accumulation**: $\frac{\partial}{\partial t}\int_x^{x+\Delta x} \rho \, dx' \approx \frac{\partial \rho}{\partial t} \Delta x$
- **Flux in at $x$**: $F(x,t)$
- **Flux out at $x+\Delta x$**: $F(x+\Delta x, t)$

Conservation requires:
$$F(x,t) - F(x+\Delta x, t) = \frac{\partial \rho}{\partial t} \Delta x$$

Dividing by $\Delta x$ and taking the limit:
$$-\frac{\partial F}{\partial x} = \frac{\partial \rho}{\partial t}$$

Or equivalently:
$$\frac{\partial \rho}{\partial t} + \frac{\partial F}{\partial x} = 0$$

This is the **continuity equation** or **conservation law form**.

### Examples

1. **Conservation of mass**: $F = \rho v$ (density times velocity)
   $$\frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x}(\rho v) = 0$$

2. **Traffic flow**: $\rho$ = car density, $v(\rho)$ = speed (depends on congestion)

3. **Conservation of momentum**: Leads to the Navier-Stokes equations

4. **Conservation of energy**: Leads to heat equations

### Generalization to Higher Dimensions

In 3D, with velocity field $\mathbf{v} = (u, v, w)$:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

where $\nabla \cdot$ is the **divergence** operator:
$$\nabla \cdot (\rho \mathbf{v}) = \frac{\partial(\rho u)}{\partial x} + \frac{\partial(\rho v)}{\partial y} + \frac{\partial(\rho w)}{\partial z}$$

This is why we study calculus—to manipulate these derivative expressions and combine them in powerful ways.

---

## 5. Linearity and Superposition: Why We Love Linear Systems

### What is Linearity?

A function $L$ is **linear** if for any inputs $X, Y$ and scalars $a, b$:
$$L(aX + bY) = aL(X) + bL(Y)$$

### Why Linearity Matters

Linear systems allow **superposition**: if you solve for the parts, you can combine them to get the solution for the whole.

Instead of understanding a complex system all at once:
1. Break it into simpler parts
2. Solve each part separately
3. Add the solutions together

This is enormously powerful for both:
- **Physical understanding**: Decompose complex phenomena
- **Mathematical tractability**: Linear equations have systematic solution methods

### The Reality: Nonlinearity

Most real systems are **nonlinear**. But whenever possible, we try to:
- **Linearize**: Approximate near an operating point
- **Find linear subsystems**: Isolate linear components
- **Perturb about equilibria**: Study small deviations

---

## 6. Case Study: The Navier-Stokes Equations

### Building the Equations

The Navier-Stokes equations describe fluid motion. They combine:

1. **Newton's law of viscosity** (an empirical law):
   $$\tau = \mu \frac{\partial u}{\partial y}$$
   Shear stress $\tau$ is proportional to velocity gradient.

2. **Conservation of momentum** (Newton's second law for fluids):
   $$\frac{D(\rho \mathbf{v})}{Dt} = \text{forces per unit volume}$$

### The Equations

For an incompressible Newtonian fluid:

$$\rho\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}\right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}$$

Where:
- $\mathbf{u}(x,y,z,t)$ = velocity field (what we solve for)
- $p$ = pressure
- $\mu$ = dynamic viscosity
- $\mathbf{f}$ = body forces (like gravity)

### Interpreting the Terms

| Term | Physical Meaning |
|------|------------------|
| $\frac{\partial \mathbf{u}}{\partial t}$ | Local acceleration (change at a fixed point) |
| $\mathbf{u} \cdot \nabla \mathbf{u}$ | **Convective/advective acceleration** (nonlinear!) |
| $-\nabla p$ | Pressure forces (compression) |
| $\mu \nabla^2 \mathbf{u}$ | Viscous forces (shearing between layers) |

### The Lagrangian Derivative

The combination:
$$\frac{D\mathbf{u}}{Dt} = \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}$$

is the **material derivative**—it tracks the acceleration of a fluid particle moving with the flow, rather than the change at a fixed point.

### The Nonlinearity Problem

The term $\mathbf{u} \cdot \nabla \mathbf{u}$ is **nonlinear** in $\mathbf{u}$. This single term:
- Makes analytical solutions nearly impossible
- Prevents proof of existence and uniqueness (Millennium Prize Problem!)
- Gives rise to turbulence and chaos

Yet we build planes, design cars, and model weather using these equations—by solving them **numerically**.

### Why Study Navier-Stokes?

It's an archetypal equation because:
- It appears everywhere (fluid flow, weather, blood circulation, aerodynamics)
- It contains all the challenges of PDEs (nonlinearity, coupled fields, boundary conditions)
- It's full of open questions (even basic mathematical properties are unproven)

---

## 7. Building Block PDEs

Before tackling complex equations like Navier-Stokes, we should understand simpler canonical PDEs. To visualize their behavior, consider a **Gaussian initial condition**:

$$u_0(x) = e^{-\frac{(x - \mu_0)^2}{2\sigma_0^2}}$$

where $\mu_0$ is the center and $\sigma_0$ is the width.

### 7.1 The Advection Equation

$$\frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = 0$$

**Physical meaning**: A quantity $u$ is transported at constant speed $c$ without changing shape.

**Solution**: $u(x,t) = u_0(x - ct)$

For a Gaussian initial condition, the solution is simply:
$$u(x,t) = e^{-\frac{(x - \mu_0 - ct)^2}{2\sigma_0^2}}$$

The profile **translates** at speed $c$ but maintains its shape. This describes pure transport without any spreading (e.g., a wave on a string, a pulse traveling down a cable).

### 7.2 The Diffusion (Heat) Equation

$$\frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2}$$

**Physical meaning**: Quantities spread out over time. Peaks decrease, valleys fill in.

**Solution for Gaussian initial condition**:
$$u(x,t) = \frac{\sigma_0}{\sigma(t)} e^{-\frac{(x - \mu_0)^2}{2\sigma(t)^2}}$$

where the width grows as:
$$\sigma(t) = \sqrt{\sigma_0^2 + 2Dt}$$

**Key insight**: The second derivative $\frac{\partial^2 u}{\partial x^2}$ measures **curvature**:
- Positive curvature (local minimum) → $u$ increases
- Negative curvature (local maximum) → $u$ decreases

This **smooths** the solution over time. The peak height decreases as $\sigma_0/\sigma(t)$ to conserve total area (mass conservation).

### 7.3 The Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = v^2 \frac{\partial^2 u}{\partial x^2}$$

**Physical meaning**: Oscillatory disturbances propagate at speed $v$.

**d'Alembert's solution**: The general solution splits into two counter-propagating waves:
$$u(x,t) = f(x - vt) + g(x + vt)$$

**Examples**: Vibrating strings, sound waves, electromagnetic waves.

### 7.4 Advection-Diffusion: Combined Effects

$$\frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = D\frac{\partial^2 u}{\partial x^2}$$

**Solution for Gaussian initial condition**:
$$u(x,t) = \frac{\sigma_0}{\sigma(t)} e^{-\frac{(x - \mu_0 - ct)^2}{2\sigma(t)^2}}$$

The solution **translates** (due to advection) AND **spreads** (due to diffusion) simultaneously.

**Example**: Pollutant in a river
- Advection: Carried downstream by the current
- Diffusion: Disperses due to molecular and turbulent mixing

### 7.5 Reaction-Diffusion

$$\frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2} + R(u)$$

**Physical meaning**: Diffusion combined with local reactions or growth.

**Examples**:
- Chemical reactions spreading through a medium
- Population dynamics with spatial spread
- **Turing patterns**: The nonlinear reaction term $R(u)$ can create spontaneous pattern formation (spots, stripes) as seen in animal coat patterns

---

## 8. Case Study: Electronic Circuits

Circuits provide another example of differential equations arising from simple laws.

### The Building Blocks

| Component | Constitutive Law | Type |
|-----------|------------------|------|
| Resistor | $V = IR$ | Algebraic |
| Capacitor | $I = C\frac{dV}{dt}$ | Differential |
| Inductor | $V = L\frac{dI}{dt}$ | Differential |

### RC Circuit

A resistor and capacitor in series:
$$RC\frac{dV}{dt} + V = V_{\text{in}}$$

This is a first-order linear ODE. Solution: exponential decay/growth toward equilibrium.

### RLC Circuit

Add an inductor:
$$LC\frac{d^2V}{dt^2} + RC\frac{dV}{dt} + V = V_{\text{in}}$$

This is a second-order linear ODE. Solutions include:
- Oscillations (underdamped)
- Exponential decay (overdamped)
- Critical damping

### Synthesizers: Designing Sound

A fascinating application: **synthesizers** use combinations of R, L, C components to shape signals. Given the differential equations governing each component, you can design circuits that produce specific waveforms—approximating the sound of a violin, for instance.

The inverse problem is also interesting: given a desired sound, what circuit produces it?

---

## 9. Units and Dimensional Analysis

A crucial but often overlooked principle: **physical quantities have units**.

### Why Units Matter

In ML, we often normalize data (mean 0, variance 1) without thinking about what the numbers represent. But in science:
- Units encode **physical meaning**
- Relationships between quantities have **dimensional consistency**
- Scale differences reveal which effects dominate

### Dimensional Analysis

Velocity has units of $[\text{length}]/[\text{time}] = \text{m/s}$
Acceleration: $\text{m/s}^2$
Force: $\text{kg} \cdot \text{m/s}^2 = \text{N}$

By checking that equations are dimensionally consistent, we can:
- Catch errors
- Derive relationships (Buckingham Pi theorem)
- Identify which terms dominate at different scales

### Feature Engineering via Scaling

If one term in an equation is much smaller than others at the relevant scale, we might neglect it. This is **asymptotic analysis**—a form of physics-informed feature engineering.

---

## 10. Solving Differential Equations: A Preview

### Analytical Solutions

For simple, linear equations with nice boundary conditions, we can find **closed-form solutions**:
- Separation of variables
- Fourier series
- Green's functions

### Numerical Solutions

Most real equations (especially nonlinear ones) require **numerical methods**:
1. **Discretize** space and time into a grid
2. Approximate derivatives with **finite differences**
3. March forward in time, computing values at each grid point

We went from discrete → continuous (for mathematical convenience) → back to discrete (for computation).

### Why the Detour?

The continuous formulation:
- Eliminates dependence on grid size
- Reveals mathematical structure
- Allows manipulation and simplification
- Connects to conservation laws and symmetries

Then we discretize in a controlled way, understanding the approximations we're making.

---

## 11. Assumptions and Simplifications

Every model makes assumptions. Common ones include:

| Assumption | Reality |
|------------|---------|
| Newtonian fluid | Non-Newtonian fluids (toothpaste, blood) |
| Linear response | Nonlinear effects at large amplitudes |
| Continuous medium | Discrete molecules |
| Deterministic | Stochastic/noisy |
| Local interactions | Long-range forces |

### The Value of Assumptions

Assumptions **simplify** the problem. We should:
- Use the simplest model that captures the essential physics
- Be aware of what we've assumed
- Know when assumptions break down

### When Models Fail

If a model doesn't work, ask: **What assumptions did I make?**

Often we're not even aware of our assumptions until we examine the building blocks. Identifying violated assumptions is key to improving models.

### The Trend Toward "Assumption-Free" Models

Deep learning promises models that learn directly from data without explicit assumptions. But there's no such thing as truly assumption-free:
- Architecture choices are assumptions
- Training data is an assumption
- Optimization is an assumption

The **inductive biases** we discussed (space-time structure, conservation, linearity) help models generalize. Scientific ML seeks to encode these biases explicitly.

---

## 12. Summary: Key Principles

1. **Space and time are special**: Functions $u(x,t)$ are the natural language of physics

2. **Change is fundamental**: Derivatives quantify how things change locally

3. **Conservation laws**: Flux in - flux out = accumulation

4. **Linearity enables superposition**: Solve parts, combine for the whole

5. **Combine simple laws → complex equations**: Newton's viscosity + momentum conservation → Navier-Stokes

6. **Nonlinearity is the challenge**: Makes equations hard to solve, creates chaos

7. **Units matter**: They encode physical meaning and reveal scales

8. **All models make assumptions**: Know yours, and know when they break

---

## 13. Looking Ahead

**Next lecture**: Numerical methods for solving differential equations
- Finite differences
- Stability and accuracy
- What happens when we solve nonlinear equations → chaos

**Later**:
- Probabilistic modeling and uncertainty
- Deep learning for differential equations (PINNs, neural operators)
- Data-driven discovery of equations (SINDy)

The goal: build a foundation for understanding **scientific machine learning**—combining the power of deep learning with the structure of physical laws.
