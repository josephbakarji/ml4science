---
title: "Least Squares Derivation"
layout: note
permalink: /static_files/lectures/02/notes_least_squares_derivation/
---


## Overview

Least squares is a method for finding the best-fit parameters of a model by minimizing the sum of squared errors between observations and predictions.

## 1. Simplest Case: y = ax (One Parameter)

### Setup

**Data**: n observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$

**Model**: $\hat{y} = ax$ (line through origin)

**Error for each point**: $e_i = y_i - ax_i$

**Total squared error (cost function)**:

$$E(a) = \sum_{i=1}^{n} (y_i - ax_i)^2$$

### Minimization

To find the minimum, take the derivative and set it to zero:

$$\frac{dE}{da} = \sum_{i=1}^{n} 2(y_i - ax_i)(-x_i) = 0$$

$$-2 \sum_{i=1}^{n} x_i(y_i - ax_i) = 0$$

$$\sum_{i=1}^{n} x_i y_i - a \sum_{i=1}^{n} x_i^2 = 0$$

### Solution

$$\boxed{a = \frac{\sum_{i=1}^{n} x_i y_i}{\sum_{i=1}^{n} x_i^2}}$$

This is a **closed-form solution**—we can compute it directly from the data.

## 2. Linear Regression: y = ax + b (Two Parameters)

### Setup

**Model**: $\hat{y} = ax + b$

**Total squared error**:

$$E(a, b) = \sum_{i=1}^{n} (y_i - ax_i - b)^2$$

### Minimization

Take partial derivatives with respect to both parameters:

$$\frac{\partial E}{\partial a} = -2 \sum_{i=1}^{n} x_i(y_i - ax_i - b) = 0$$

$$\frac{\partial E}{\partial b} = -2 \sum_{i=1}^{n} (y_i - ax_i - b) = 0$$

### The Normal Equations

Expanding these conditions:

$$\sum x_i y_i = a \sum x_i^2 + b \sum x_i \quad \text{...(1)}$$

$$\sum y_i = a \sum x_i + b \cdot n \quad \text{...(2)}$$

### Matrix Form

$$\begin{pmatrix} \sum x_i^2 & \sum x_i \\ \sum x_i & n \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} \sum x_i y_i \\ \sum y_i \end{pmatrix}$$

This is a 2×2 linear system. Solving:

$$a = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}$$

$$b = \frac{\sum y_i - a \sum x_i}{n} = \bar{y} - a\bar{x}$$

## 3. General Linear Model (k Parameters)

### Setup

**Model**: Linear combination of basis functions

$$\hat{y} = \theta_1 f_1(x) + \theta_2 f_2(x) + \cdots + \theta_k f_k(x)$$

Examples:
- Polynomial: $f_j(x) = x^{j-1}$, giving $\hat{y} = \theta_1 + \theta_2 x + \theta_3 x^2 + \cdots$
- Fourier: $f_j(x) = \sin(jx)$ or $\cos(jx)$

### Matrix Formulation

**Design matrix** $\mathbf{X}$ (n × k):

$$\mathbf{X} = \begin{pmatrix} f_1(x_1) & f_2(x_1) & \cdots & f_k(x_1) \\ f_1(x_2) & f_2(x_2) & \cdots & f_k(x_2) \\ \vdots & \vdots & \ddots & \vdots \\ f_1(x_n) & f_2(x_n) & \cdots & f_k(x_n) \end{pmatrix}$$

**Parameter vector**: $\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_k)^T$

**Observation vector**: $\mathbf{y} = (y_1, y_2, \ldots, y_n)^T$

**Model in matrix form**: $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$

### Cost Function

$$E(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})$$

Expanding:

$$E = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\theta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\theta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\theta}$$

### Minimization

Take the gradient with respect to $\boldsymbol{\theta}$ and set to zero:

$$\nabla_{\boldsymbol{\theta}} E = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} = 0$$

### The Normal Equations (General Form)

$$\boxed{\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} = \mathbf{X}^T\mathbf{y}}$$

### Solution

$$\boxed{\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

This is **the** fundamental result of linear least squares.

**Note**: The matrix $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is called the **pseudoinverse** of $\mathbf{X}$.

## 4. The Ellipse Model: Kepler's Laws and Orbital Mechanics

Before discussing nonlinear least squares, let's understand *why* orbits are ellipses and what equations govern them.

### Kepler's Laws (1609-1619)

**First Law**: Planets move in *ellipses* with the Sun at one focus.

**Second Law**: A line from the Sun to a planet sweeps equal areas in equal times.

**Third Law**: The square of the orbital period is proportional to the cube of the semi-major axis: $T^2 \propto a^3$

These were empirical laws—Kepler fit them to Tycho Brahe's observational data. Newton later derived them from his law of gravitation.

### The Ellipse Equation

An ellipse in its orbital plane can be written in polar coordinates $(r, \nu)$ where $r$ is distance from the focus (Sun) and $\nu$ is the **true anomaly** (angle from perihelion):

$$r = \frac{a(1 - e^2)}{1 + e \cos \nu}$$

where:
- $a$ = semi-major axis (size of the ellipse)
- $e$ = eccentricity (shape: 0 = circle, 0 < e < 1 = ellipse)

### The Time Problem: Kepler's Equation

Given time $t$, where is the planet? This requires solving **Kepler's equation**:

$$M = E - e \sin E$$

where:
- $M$ = **mean anomaly** = $\frac{2\pi}{T}(t - T_0)$ (proportional to time since perihelion)
- $E$ = **eccentric anomaly** (auxiliary angle)
- $T_0$ = time of perihelion passage

**This equation cannot be solved analytically for $E$!** You must use iteration (Newton-Raphson) or series expansion.

Once you have $E$, the true anomaly is:

$$\tan\frac{\nu}{2} = \sqrt{\frac{1+e}{1-e}} \tan\frac{E}{2}$$

### From 2D to 3D: The Orbital Elements

The ellipse lives in a plane, but we observe from Earth in 3D. Three angles orient the orbital plane:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Semi-major axis | $a$ | Size of ellipse |
| Eccentricity | $e$ | Shape of ellipse |
| Inclination | $i$ | Tilt of orbital plane relative to ecliptic |
| Long. of ascending node | $\Omega$ | Where orbit crosses ecliptic going "up" |
| Argument of perihelion | $\omega$ | Orientation of ellipse within its plane |
| Time of perihelion | $T_0$ | When object is closest to Sun |

### The Full Position Function

The position in 3D space at time $t$ is:

$$\mathbf{r}(t) = R_z(-\Omega) \cdot R_x(-i) \cdot R_z(-\omega) \cdot \begin{pmatrix} r\cos\nu \\ r\sin\nu \\ 0 \end{pmatrix}$$

where $R_x, R_z$ are rotation matrices, and $r, \nu$ come from solving Kepler's equation.

**This is highly nonlinear in the 6 parameters** $(a, e, i, \Omega, \omega, T_0)$!

### Why Gauss's Problem Was Hard

1. **Nonlinearity**: Position depends nonlinearly on orbital elements
2. **Transcendental equation**: Kepler's equation requires iteration
3. **Sparse data**: Only 22 observations over 9° of arc
4. **Noise**: Every measurement has errors
5. **No closed form**: Can't just "solve" for the 6 parameters

This is why Gauss needed iterative methods (linearize → solve → repeat).

---

## 5. Nonlinear Least Squares

### The Problem

When the model is nonlinear in the parameters, we cannot solve directly:

$$\hat{y} = f(x; \boldsymbol{\theta})$$

For Ceres, the "observation" is angular position (RA, Dec), and the model predicts position from orbital elements:

$$(\text{RA}, \text{Dec})_{\text{predicted}} = f(a, e, i, \Omega, \omega, T_0, t)$$

The position depends nonlinearly on the 6 orbital elements through Kepler's equation and 3D rotations.

### Gauss-Newton Method

**Key idea**: Linearize around current estimate, solve the linear problem, iterate.

**Step 1**: Start with initial guess $\boldsymbol{\theta}_0$

**Step 2**: Linearize the model using Taylor expansion:

$$f(x; \boldsymbol{\theta}) \approx f(x; \boldsymbol{\theta}_0) + \mathbf{J}(\boldsymbol{\theta} - \boldsymbol{\theta}_0)$$

where $\mathbf{J}$ is the **Jacobian matrix**:

$$J_{ij} = \frac{\partial f(x_i; \boldsymbol{\theta})}{\partial \theta_j} \bigg|_{\boldsymbol{\theta}_0}$$

**Step 3**: Define the residual $\mathbf{r} = \mathbf{y} - f(\mathbf{x}; \boldsymbol{\theta}_0)$

**Step 4**: Solve the linear least squares problem for the update $\Delta\boldsymbol{\theta}$:

$$\mathbf{J}^T\mathbf{J} \Delta\boldsymbol{\theta} = \mathbf{J}^T\mathbf{r}$$

**Step 5**: Update: $\boldsymbol{\theta}_1 = \boldsymbol{\theta}_0 + \Delta\boldsymbol{\theta}$

**Step 6**: Repeat until convergence

This is the **Gauss-Newton algorithm**—invented by Gauss for the Ceres problem!

## 6. Why Squared Errors?

### Practical Reasons

1. **Positivity**: Errors can be positive or negative; squaring ensures they don't cancel
2. **Differentiability**: $e^2$ is smooth everywhere; $|e|$ has a kink at zero
3. **Simple derivatives**: $\frac{d}{de}(e^2) = 2e$

### Statistical Reason: Maximum Likelihood

If measurement errors follow a **normal (Gaussian) distribution**:

$$P(\text{error} = e) \propto \exp\left(-\frac{e^2}{2\sigma^2}\right)$$

For independent measurements, the joint probability is:

$$P(\text{all errors}) \propto \exp\left(-\frac{\sum e_i^2}{2\sigma^2}\right)$$

Taking the log:

$$\log P = \text{const} - \frac{1}{2\sigma^2}\sum e_i^2$$

**Maximizing probability** (maximum likelihood) is equivalent to **minimizing $\sum e_i^2$**.

This is why least squares is optimal when errors are Gaussian—and errors often are Gaussian due to the Central Limit Theorem.

### Gauss's Contribution

Gauss went beyond Legendre by:
1. Connecting least squares to probability theory
2. Deriving the normal distribution from first principles
3. Proving that least squares is optimal under Gaussian errors

The normal distribution is called "Gaussian" in his honor.

## 7. Geometric Interpretation

Least squares finds the **orthogonal projection** of $\mathbf{y}$ onto the column space of $\mathbf{X}$.

```
        y
       /|
      / |
     /  | (error vector, perpendicular to column space)
    /   |
   /    v
  ---------> X*θ (projection onto column space of X)
```

The residual $\mathbf{y} - \mathbf{X}\boldsymbol{\theta}$ is perpendicular to every column of $\mathbf{X}$:

$$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}) = 0$$

This is exactly the normal equation!

## Summary

| Case | Model | Solution |
|------|-------|----------|
| 1 parameter | $y = ax$ | $a = \frac{\sum x_i y_i}{\sum x_i^2}$ |
| 2 parameters | $y = ax + b$ | Solve 2×2 system |
| k parameters (linear) | $\mathbf{y} = \mathbf{X}\boldsymbol{\theta}$ | $\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| Nonlinear | $y = f(x; \boldsymbol{\theta})$ | Iterate: $\mathbf{J}^T\mathbf{J}\Delta\boldsymbol{\theta} = \mathbf{J}^T\mathbf{r}$ |

## References

- [Wikipedia: Least Squares](https://en.wikipedia.org/wiki/Least_squares)
- [The Method of Least Squares - Georgia Tech](https://textbooks.math.gatech.edu/ila/least-squares.html)
- [Least Squares Fitting - Wolfram MathWorld](https://mathworld.wolfram.com/LeastSquaresFitting.html)
