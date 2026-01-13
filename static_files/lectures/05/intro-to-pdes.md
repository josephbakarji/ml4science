---
title: "Understanding Data Through PDEs: From Modeling to Discovery"
layout: note
permalink: /static_files/lectures/05/intro-to-pdes/
---


## 1. Introduction

Suppose you have measurements of a quantity \(u(x,t)\) in space \(x\) and time \(t\). The problem you want to solve is:

> **How can we use these measurements to understand the system and predict its future evolution?**

One of the most powerful frameworks for tackling this problem is **partial differential equations (PDEs)**. PDEs allow us to capture how \(u(x,t)\) changes in both space and time using derivatives. By combining information about how \(u\) changes in space and time, we often uncover fundamental relationships—ranging from conservation principles to physical laws—that govern the evolution of \(u\).

In many physical systems (e.g., fluid flow, heat conduction, wave propagation), the assumptions of continuity and differentiability are reasonable, so classical PDEs serve as an excellent modeling tool. Once a PDE is identified or assumed, we can solve it—either analytically or numerically—to predict or interpret the system’s behavior.


## 2. The Role of Differential Equations

### 2.1 Why Use Differential Equations?

Differential equations link **changes** of a function \(u\) to the function itself and possibly its spatial variations. This linkage arises from the idea that **local changes** in space and/or time are driven by **local properties** of \(u\). For example:

- **Advection**: where a quantity is transported or carried along by a flow with velocity \(c\).
- **Diffusion**: where a quantity tends to spread out due to concentration gradients.
- **Waves**: where oscillatory or periodic disturbances propagate through the medium.

### 2.2 Simple Example: The Advection Equation

Consider the assumption that **the change in \(u\) with respect to time is proportional to the change in \(u\) with respect to space**. Symbolically, in discrete form, you might say:

\[
\frac{\Delta u}{\Delta t} \propto \frac{\Delta u}{\Delta x}.
\]

Taking the limit \(\Delta x \to 0\) and \(\Delta t \to 0\), we obtain

\[
\frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} \;=\; 0,
\]

where \(c\) is the proportionality constant (interpreted as a constant velocity). This is the **advection equation**, which essentially describes how the quantity \(u\) is translated with speed \(c\). 

#### Derivation from a Lagrangian Perspective

An alternative derivation: If \(u(x,t)\) is advected with speed \(c\), then in a moving coordinate system \(\xi = x - ct\), \(u\) does not change with time. Thus,

\[
\frac{d}{dt} u(x-ct,t) \;=\; 0 \;\;\Rightarrow\;\;
\frac{\partial u}{\partial t} + \left(\frac{d\xi}{dt}\right)\frac{\partial u}{\partial x} \;=\; 0,
\]

and because \(d\xi/dt = -c\), we get 

\[
\frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} \;=\; 0.
\]


### 2.3 The Diffusion Equation

The **diffusion equation** states that the rate of change in time of \(u\) is proportional to the spatial curvature (second derivative) of \(u\). In discrete terms, you might approximate the “change of the change” in \(u\) over space and equate it to the change in time:

\[
\frac{\Delta \left(\frac{\Delta u}{\Delta x}\right)}{\Delta x} \;\approx\; \frac{\Delta u}{\Delta t}.
\]

As \(\Delta x \to 0\) and \(\Delta t \to 0\), this leads to the continuous form:

\[
\frac{\partial u}{\partial t} \;=\; D\, \frac{\partial^2 u}{\partial x^2},
\]

where \(D\) is the **diffusion coefficient**. 

#### Physical Interpretation

- A positive second derivative (\(\partial^2 u/\partial x^2\)) indicates a local minimum in \(u\) which causes \(u\) to increase over time.
- A negative second derivative corresponds to a local maximum, causing \(u\) to decrease over time.

This process **smooths out** the quantity \(u\) over time.


### 2.4 Advection–Diffusion Equation

If a quantity is **both** carried along by advection **and** has a diffusive tendency, we combine the two effects:

\[
\frac{\partial u}{\partial t} \;+\; c\,\frac{\partial u}{\partial x} \;=\; D\,\frac{\partial^2 u}{\partial x^2}.
\]

Examples:

- **Heat** in a moving fluid: advected by fluid flow with velocity \(c\) and diffused by thermal conduction.
- **Pollutant** in a river: carried by the current and dispersed by molecular and turbulent diffusion.


### 2.5 The Wave Equation

The **wave equation** models oscillatory systems, such as vibrations of a guitar string, pressure waves in air, or electromagnetic waves in vacuum. Its 1D form is:

\[
\frac{\partial^2 u}{\partial t^2} \;=\; v^2\,\frac{\partial^2 u}{\partial x^2},
\]

where \(v\) is the wave speed. 

- **Physical derivation** often comes from force balance (like tension in a string) leading to a spatial derivative of tension forces and Newton’s second law for acceleration (\(\partial^2 u/\partial t^2\)).


## 3. Conservation Laws and PDE Derivations

An even more systematic approach to deriving PDEs is through **conservation laws**. The general statement is:

\[
\text{Flux in} \;-\; \text{Flux out} \;=\; \text{Accumulation}.
\]

### 3.1 Example in 1D

Let \(\rho(x,t)\) be the density of some conserved quantity (mass, momentum, number of cars, etc.). In an infinitesimal control volume \([x, x+\Delta x]\):

- **Flux in** at \(x\) is \(F(x,t)\).
- **Flux out** at \(x+\Delta x\) is \(F(x+\Delta x,t)\).
- **Accumulation** over \([x, x+\Delta x]\) in time \(\Delta t\) is the change in total amount: 

  \[
  \frac{\partial}{\partial t} \int_{x}^{x+\Delta x} \rho(x', t)\, dx'.
  \]

Hence, we write:

\[
F(x,t) - F(x+\Delta x,t) = \frac{\partial}{\partial t} \int_{x}^{x+\Delta x} \rho(x', t)\, dx'.
\]

Divide by \(\Delta x\) and take the limit \(\Delta x \to 0\). The fundamental theorem of calculus yields:

\[
-\frac{\partial F}{\partial x} = \frac{\partial \rho}{\partial t}.
\]

Or equivalently,

\[
\frac{\partial \rho}{\partial t} + \frac{\partial F}{\partial x} = 0.
\]

### 3.2 Special Cases

1. **Mass conservation**: \(F = \rho v\), giving

   \[
   \frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x} (\rho\,v) = 0.
   \]

2. **Momentum conservation**: \(F = \rho\,v^2 + \text{(pressure terms)}\), etc.

3. **Traffic flow**: \(\rho\) is car density, and \(v(\rho)\) might be a known velocity-density relationship. Then \(F = \rho\,v(\rho)\).

Such conservation arguments are the starting point for many PDEs found in fluid mechanics (the Navier–Stokes equations), gas dynamics (Euler equations), and beyond.


## 4. Analytical Solutions

Many PDEs—especially linear ones—admit **closed-form analytical solutions** under certain conditions (e.g., specific domain geometries, boundary/initial conditions).

### 4.1 Separation of Variables

A classic technique, **separation of variables**, is applicable when the PDE and boundary conditions have certain forms of linearity and homogeneity. We assume:

\[
u(x,t) = X(x)\,T(t).
\]

#### Example: 1D Diffusion Equation

\[
\frac{\partial u}{\partial t} = D\,\frac{\partial^2 u}{\partial x^2}.
\]

Plugging \(u(x,t) = X(x)\,T(t)\) in:

\[
X(x)\,\frac{d T(t)}{d t} = D\,T(t)\,\frac{d^2 X(x)}{d x^2}.
\]

Divide both sides by \(D\,X(x)\,T(t)\):

\[
\frac{1}{D}\,\frac{1}{T(t)}\;\frac{d T}{dt} = \frac{1}{X(x)}\;\frac{d^2 X}{dx^2}.
\]

The left-hand side depends only on \(t\), and the right-hand side depends only on \(x\). For them to be equal for all \(x\) and \(t\), they must each be equal to a **constant**, say \(-\lambda\). Then:

\[
\frac{1}{T} \,\frac{d T}{dt} = -\lambda D, 
\quad\quad
\frac{1}{X} \,\frac{d^2 X}{dx^2} = -\lambda.
\]

We get two ODEs:

\[
\frac{d T}{d t} = -\lambda D\, T,
\quad\quad
\frac{d^2 X}{dx^2} = -\lambda\,X.
\]

Solving these gives families of solutions, which we combine and then use boundary and initial conditions to fix constants.

### 4.2 Boundary and Initial Conditions

- **Initial condition**: \(u(x,0) = f(x)\).
- **Boundary conditions**: e.g., 
  - Dirichlet: \(u(0,t) = \alpha_0,\, u(L,t) = \alpha_L\).
  - Neumann: \(\frac{\partial u}{\partial x}(0,t) = 0,\dots\).

Conditions define how the solution behaves at the edges and at \(t=0\). Without them, the solution to a PDE is not unique (or may not exist in a classical sense).

---

## 5. Numerical Solutions

Many real-world PDEs (especially **nonlinear** ones) do not succumb to neat analytical solutions. In these cases, we turn to **numerical methods**.

### 5.1 Spatial and Temporal Discretization

Let \(x_i = x_0 + i\,\Delta x\) (\(i = 0,\dots,N\)), and \(t_j = t_0 + j\,\Delta t\) (\(j = 0,\dots,M\)). We define:

\[
u_i^j \;\equiv\; u(x_i, t_j).
\]

Then we approximate derivatives. For instance:

\[
\frac{\partial u}{\partial t}\Big|_{(x_i,\,t_j)} 
\approx \frac{u_i^{j+1} - u_i^j}{\Delta t},
\]

\[
\frac{\partial^2 u}{\partial x^2}\Big|_{(x_i,\,t_j)} 
\approx \frac{u_{i+1}^j - 2\,u_i^j + u_{i-1}^j}{(\Delta x)^2}.
\]

### 5.2 Example: Explicit Finite Difference Scheme for Diffusion

For the diffusion equation:

\[
\frac{\partial u}{\partial t} 
= D\,\frac{\partial^2 u}{\partial x^2},
\]

approximate each side:

\[
\frac{u_i^{j+1} - u_i^j}{\Delta t} 
= D\,\frac{u_{i+1}^j - 2u_i^j + u_{i-1}^j}{(\Delta x)^2}.
\]

Rearrange to find \(u_i^{j+1}\):

\[
u_i^{j+1} 
= u_i^j 
+ D\,\frac{\Delta t}{(\Delta x)^2} 
\bigl(u_{i+1}^j - 2u_i^j + u_{i-1}^j\bigr).
\]

**Python-like Pseudo-Code**:

```python
for j in range(time_steps):
    for i in range(1, N-1):  # skip boundary points if Dirichlet or handle them otherwise
        u_new[i] = (u[i] 
                     + D*(dt/dx**2) * (u[i+1] - 2*u[i] + u[i-1]))
    u[:] = u_new[:]
```

#### Stability Criterion

For the explicit diffusion scheme, a common criterion is:

\[
r = \frac{D\,\Delta t}{(\Delta x)^2} \;\leq\; \frac{1}{2} 
\quad\longrightarrow\quad
\Delta t \;\le\; \frac{(\Delta x)^2}{2\,D}.
\]

If \(r\) is too large, the numerical solution can become unstable.

## 6. Conclusion

In this chapter, we explored how PDEs provide a framework for **understanding data** that varies in space and time. Key insights:

1. **Modeling from Principles**: Basic physical/engineering assumptions (like advection or diffusion) or fundamental conservation laws (mass, momentum, energy) lead to **canonical PDEs** (advection, diffusion, wave equations, etc.).  
2. **Analytical Solutions**: For simple PDEs with standard boundary/initial conditions, we can solve analytically (e.g., via **separation of variables**).  
3. **Numerical Methods**: More complex or nonlinear PDEs typically require discretization (finite differences, finite elements, etc.) to simulate the system’s evolution in time.  
4. **Data-Driven Discovery (SINDy)**: With modern computational power and measurement techniques, we can estimate derivatives directly from data and then **infer the PDE** itself via sparse regression.

This approach—merging theoretical derivations, numerical solutions, and data-driven methods—enables engineers and scientists to **gain insight** into complex spatiotemporal phenomena, predict future states, and even **uncover new physical laws** where the governing PDEs were previously unknown.

