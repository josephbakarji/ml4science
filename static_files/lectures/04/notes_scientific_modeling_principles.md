---
title: "Lecture Notes: Scientific Modeling Principles"
layout: note
permalink: /static_files/lectures/04/notes_scientific_modeling_principles/
---


## 1. Introduction

This lecture covers principles of scientific modeling with the goal of understanding why differential equations are useful. We review familiar concepts like Newton's second law and identify general principles that appear repeatedly when implementing scientific machine learning, particularly when incorporating physical constraints into ML models.

A central theme in both ML and science is generalization: building models that work beyond the training data. This naturally leads us to prefer simple models. Simple models enable fast inference, generalize better beyond observed data, and avoid becoming mere lookup tables of training examples.

There is a powerful inductive bias in science: space and time are special variables. We are typically interested in functions that map points in space $(x, y, z)$ and time $t$ to measured quantities:

$$u: (x, y, z, t) \mapsto \text{measured variable}$$

This differs from the empirical laws we saw earlier (like $V = IR$), which map one measured variable to another at a single instant. Here, we are interested in how things evolve.

To quantify position, you need a coordinate system with some fixed reference point. To quantify time, you need a clock. These measurements have units (meters, seconds), which will become important later.

---

## 2. Quantifying Change

A profound observation: nothing stays the same in space and time. Everything changes. Yet in the midst of this change, we search for laws that remain constant, and these laws typically describe how things change.

Given discrete measurements $u_i$ at times $t_i$, we can quantify change using finite differences:

$$\frac{\Delta u}{\Delta t} = \frac{u_i - u_{i-1}}{t_i - t_{i-1}}$$

This is the slope between two data points, an approximation of how fast $u$ is changing. As we collect more data points (making $\Delta t$ smaller), this approximation improves. In the limit:

$$\frac{\partial u}{\partial t} = \lim_{\Delta t \to 0} \frac{u(t + \Delta t) - u(t)}{\Delta t}$$

This limiting process, formalized by Leibniz and Newton, was controversial for a long time. What does it mean for something to be "infinitely small but not zero"? But it works remarkably well for describing the physical world.

The key insight is that we go from discrete (data) to continuous (mathematics) and back to discrete (computation). The continuous formulation eliminates dependence on specific grid spacing and reveals mathematical structure.

There is a fundamental divide between scientific and computational perspectives. Sciences focus on continuous systems with smooth functions defined everywhere and well-behaved derivatives. Computer science focuses on discrete systems involving states, transitions, and graphs. These perspectives connect through limits: the continuous world emerges from taking limits of the discrete world. Both are idealizations of reality.

---

## 3. From Galileo to Newton

When Galileo dropped objects from the Tower of Pisa (or so the legend goes), he noticed something curious. Position $x(t)$ increases, and velocity $v = dx/dt$ also increases as objects accelerate. But the acceleration $a = d^2x/dt^2$ is constant. This constant is $g \approx 9.8 \, \text{m/s}^2$, the acceleration due to gravity. It is the same for all objects, everywhere on Earth (approximately).

This observation generalizes to Newton's second law:

$$F = ma = m\frac{d^2 x}{dt^2}$$

Force equals mass times acceleration. For free fall near Earth's surface, $F = mg$, so:

$$\frac{d^2 x}{dt^2} = g$$

This is a differential equation, an equation involving derivatives. To solve it, integrate once to get velocity: $dx/dt = gt + v_0$. Integrate again to get position: $x = \frac{1}{2}gt^2 + v_0 t + x_0$. The result is a parabola. From a simple differential equation, we derived the shape of a projectile's trajectory.

Adding drag (air resistance proportional to velocity) gives:
$$m\frac{d^2x}{dt^2} = -mg - b\frac{dx}{dt}$$

This introduces exponential decay and modified trajectories.

---

## 4. Conservation Laws

Conservation laws appear everywhere in physics. The fundamental idea is simple:

$$\text{Flux in} - \text{Flux out} = \text{Accumulation}$$

Think of people in a room: if 10 enter and 3 leave, the number inside increases by 7.

Consider a quantity with density $\rho(x,t)$ in a 1D domain. In a small region $[x, x+\Delta x]$, the accumulation is approximately $(\partial \rho / \partial t) \Delta x$. The flux in at $x$ is $F(x,t)$, and the flux out at $x+\Delta x$ is $F(x+\Delta x, t)$. Conservation requires:

$$F(x,t) - F(x+\Delta x, t) = \frac{\partial \rho}{\partial t} \Delta x$$

Dividing by $\Delta x$ and taking the limit yields:

$$\frac{\partial \rho}{\partial t} + \frac{\partial F}{\partial x} = 0$$

This is the continuity equation, also called the conservation law form.

Conservation of mass sets $F = \rho v$ (density times velocity), giving $\partial \rho / \partial t + \partial(\rho v)/\partial x = 0$. Traffic flow uses the same form where $\rho$ represents car density and $v(\rho)$ is speed as a function of congestion. Conservation of momentum leads to the Navier-Stokes equations, while conservation of energy leads to heat equations.

In three dimensions with velocity field $\mathbf{v} = (u, v, w)$:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

where $\nabla \cdot$ is the divergence operator:

$$\nabla \cdot (\rho \mathbf{v}) = \frac{\partial(\rho u)}{\partial x} + \frac{\partial(\rho v)}{\partial y} + \frac{\partial(\rho w)}{\partial z}$$

This is why we study calculus: to manipulate these derivative expressions and combine them in powerful ways.

---

## 5. Linearity and Superposition

A function $L$ is linear if for any inputs $X, Y$ and scalars $a, b$:

$$L(aX + bY) = aL(X) + bL(Y)$$

Linear systems allow superposition: if you solve for the parts, you can combine them to get the solution for the whole. Instead of understanding a complex system all at once, break it into simpler parts, solve each part separately, and add the solutions together. This is enormously powerful for both physical understanding (decomposing complex phenomena) and mathematical tractability (linear equations have systematic solution methods).

Most real systems are nonlinear. But whenever possible, we try to linearize by approximating near an operating point, find linear subsystems by isolating linear components, or perturb about equilibria to study small deviations.

---

## 6. The Navier-Stokes Equations

The Navier-Stokes equations describe fluid motion by combining Newton's law of viscosity (an empirical law where shear stress $\tau = \mu \partial u / \partial y$ is proportional to velocity gradient) with conservation of momentum (Newton's second law for fluids).

For an incompressible Newtonian fluid:

$$\rho\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u}\right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}$$

Here $\mathbf{u}(x,y,z,t)$ is the velocity field (what we solve for), $p$ is pressure, $\mu$ is dynamic viscosity, and $\mathbf{f}$ represents body forces like gravity.

Each term has a physical meaning:

| Term | Physical Meaning |
|------|------------------|
| $\frac{\partial \mathbf{u}}{\partial t}$ | Local acceleration (change at a fixed point) |
| $\mathbf{u} \cdot \nabla \mathbf{u}$ | Convective acceleration (nonlinear) |
| $-\nabla p$ | Pressure forces (compression) |
| $\mu \nabla^2 \mathbf{u}$ | Viscous forces (shearing between layers) |

The combination $D\mathbf{u}/Dt = \partial \mathbf{u}/\partial t + \mathbf{u} \cdot \nabla \mathbf{u}$ is called the material derivative. It tracks the acceleration of a fluid particle moving with the flow, rather than the change at a fixed point.

The term $\mathbf{u} \cdot \nabla \mathbf{u}$ is nonlinear in $\mathbf{u}$. This single term makes analytical solutions nearly impossible, prevents proof of existence and uniqueness (a Millennium Prize Problem), and gives rise to turbulence and chaos. Yet we build planes, design cars, and model weather using these equations by solving them numerically.

The Navier-Stokes equations are archetypal because they appear everywhere (fluid flow, weather, blood circulation, aerodynamics), contain all the challenges of PDEs (nonlinearity, coupled fields, boundary conditions), and remain full of open questions (even basic mathematical properties are unproven).

---

## 7. Building Block PDEs

Before tackling complex equations like Navier-Stokes, we should understand simpler canonical PDEs. To visualize their behavior, consider a Gaussian initial condition:

$$u_0(x) = e^{-\frac{(x - \mu_0)^2}{2\sigma_0^2}}$$

where $\mu_0$ is the center and $\sigma_0$ is the width.

### The Advection Equation

$$\frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = 0$$

A quantity $u$ is transported at constant speed $c$ without changing shape. The solution is $u(x,t) = u_0(x - ct)$. For a Gaussian initial condition:

$$u(x,t) = e^{-\frac{(x - \mu_0 - ct)^2}{2\sigma_0^2}}$$

The profile translates at speed $c$ but maintains its shape. This describes pure transport without spreading, such as a wave on a string or a pulse traveling down a cable.

### The Diffusion Equation

$$\frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2}$$

Quantities spread out over time; peaks decrease and valleys fill in. For a Gaussian initial condition:

$$u(x,t) = \frac{\sigma_0}{\sigma(t)} e^{-\frac{(x - \mu_0)^2}{2\sigma(t)^2}}$$

where the width grows as $\sigma(t) = \sqrt{\sigma_0^2 + 2Dt}$.

The second derivative $\partial^2 u / \partial x^2$ measures curvature. Positive curvature (local minimum) causes $u$ to increase; negative curvature (local maximum) causes $u$ to decrease. This smooths the solution over time. The peak height decreases as $\sigma_0/\sigma(t)$ to conserve total area (mass conservation).

### The Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = v^2 \frac{\partial^2 u}{\partial x^2}$$

Oscillatory disturbances propagate at speed $v$. D'Alembert's solution shows that the general solution splits into two counter-propagating waves: $u(x,t) = f(x - vt) + g(x + vt)$. Examples include vibrating strings, sound waves, and electromagnetic waves.

### Advection-Diffusion

$$\frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = D\frac{\partial^2 u}{\partial x^2}$$

The solution translates (due to advection) and spreads (due to diffusion) simultaneously. A pollutant in a river, for example, is carried downstream by the current while dispersing due to molecular and turbulent mixing.

### Reaction-Diffusion

$$\frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2} + R(u)$$

Diffusion combines with local reactions or growth. Examples include chemical reactions spreading through a medium, population dynamics with spatial spread, and Turing patterns where the nonlinear reaction term $R(u)$ creates spontaneous pattern formation (spots, stripes) as seen in animal coat patterns.

---

## 8. Electronic Circuits

Circuits provide another example of differential equations arising from simple laws. Three components form the building blocks:

| Component | Constitutive Law | Type |
|-----------|------------------|------|
| Resistor | $V = IR$ | Algebraic |
| Capacitor | $I = C\frac{dV}{dt}$ | Differential |
| Inductor | $V = L\frac{dI}{dt}$ | Differential |

A resistor and capacitor in series (RC circuit) obey $RC \, dV/dt + V = V_{\text{in}}$, a first-order linear ODE with exponential decay or growth toward equilibrium. Adding an inductor (RLC circuit) gives $LC \, d^2V/dt^2 + RC \, dV/dt + V = V_{\text{in}}$, a second-order linear ODE with solutions including oscillations (underdamped), exponential decay (overdamped), and critical damping.

Synthesizers use combinations of R, L, C components to shape signals. Given the differential equations governing each component, you can design circuits that produce specific waveforms, approximating the sound of a violin for instance. The inverse problem is also interesting: given a desired sound, what circuit produces it?

---

## 9. Units and Dimensional Analysis

Physical quantities have units. In ML, we often normalize data (mean 0, variance 1) without thinking about what the numbers represent. But in science, units encode physical meaning, relationships between quantities have dimensional consistency, and scale differences reveal which effects dominate.

Velocity has units of $[\text{length}]/[\text{time}] = \text{m/s}$, acceleration has units $\text{m/s}^2$, and force has units $\text{kg} \cdot \text{m/s}^2 = \text{N}$. By checking that equations are dimensionally consistent, we can catch errors, derive relationships (Buckingham Pi theorem), and identify which terms dominate at different scales.

If one term in an equation is much smaller than others at the relevant scale, we might neglect it. This is asymptotic analysis, a form of physics-informed feature engineering.

---

## 10. Solving Differential Equations

For simple, linear equations with nice boundary conditions, we can find closed-form solutions using separation of variables, Fourier series, or Green's functions.

Most real equations (especially nonlinear ones) require numerical methods. The process involves discretizing space and time into a grid, approximating derivatives with finite differences, and marching forward in time while computing values at each grid point.

We went from discrete to continuous (for mathematical convenience) and back to discrete (for computation). The continuous formulation eliminates dependence on grid size, reveals mathematical structure, allows manipulation and simplification, and connects to conservation laws and symmetries. Then we discretize in a controlled way, understanding the approximations we are making.

---

## 11. Assumptions and Simplifications

Every model makes assumptions:

| Assumption | Reality |
|------------|---------|
| Newtonian fluid | Non-Newtonian fluids (toothpaste, blood) |
| Linear response | Nonlinear effects at large amplitudes |
| Continuous medium | Discrete molecules |
| Deterministic | Stochastic/noisy |
| Local interactions | Long-range forces |

Assumptions simplify the problem. We should use the simplest model that captures the essential physics, be aware of what we have assumed, and know when assumptions break down. If a model does not work, ask: what assumptions did I make? Often we are not even aware of our assumptions until we examine the building blocks. Identifying violated assumptions is key to improving models.

Deep learning promises models that learn directly from data without explicit assumptions. But there is no such thing as truly assumption-free: architecture choices are assumptions, training data is an assumption, and optimization is an assumption. The inductive biases we discussed (space-time structure, conservation, linearity) help models generalize. Scientific ML seeks to encode these biases explicitly.

---

## 12. Summary

Space and time are special: functions $u(x,t)$ are the natural language of physics. Change is fundamental, and derivatives quantify how things change locally. Conservation laws express the principle that flux in minus flux out equals accumulation. Linearity enables superposition, allowing us to solve parts and combine them for the whole. Combining simple laws yields complex equations: Newton's viscosity plus momentum conservation yields Navier-Stokes. Nonlinearity is the challenge, making equations hard to solve and creating chaos. Units matter because they encode physical meaning and reveal scales. All models make assumptions; know yours and know when they break.

---

## 13. Looking Ahead

The next lecture covers numerical methods for solving differential equations, including finite differences, stability and accuracy, and what happens when we solve nonlinear equations (chaos).

Later topics include probabilistic modeling and uncertainty, deep learning for differential equations (PINNs, neural operators), and data-driven discovery of equations (SINDy). The goal is to build a foundation for understanding scientific machine learning, combining the power of deep learning with the structure of physical laws.
