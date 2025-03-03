# Sparse Identification of Differential Equations (SINDy)

## 1. Introduction

Many physical, biological, and engineering systems admit differential equation
representations. For a one-dimensional system, we often write:

$$
\dot{x} \;=\; f(x),
$$

where 

$$
\dot{x} \;=\; \frac{d x}{d t}.
$$

However, suppose you only have time-series data $$\{x^{(i)}, t_i\}$$, without
knowing $$f$$. Sparse Identification of Nonlinear Dynamics (SINDy) is one way to
address this scenario by assuming that $$f$$ is a sparse linear combination of
candidate functions that are predefined as a feature library. Concretely, if 

$$
\Phi(x) \;=\; 
\begin{bmatrix}
\phi_1(x) \\
\phi_2(x) \\
\vdots     \\
\phi_M(x)
\end{bmatrix}
$$

is a feature library, we write:

$$
\dot{x} \;=\; \Phi(x) \cdot \,\boldsymbol{w} \;=\; 
\sum_{j=1}^M \phi_j(x)\, w_j,
$$

so that $$f(x)$$ is nonlinear in $$x$$ but **linear in** the parameters
$$\boldsymbol{w}$$.


## 2. From Time Series to Derivative Approximation

Suppose we measure 

$$
x^{(1)} = x(t_1),\; x^{(2)} = x(t_2),\;\dots,\; x^{(N)} = x(t_N).
$$

We thus have $$N$$ snapshots $$\bigl\{(x^{(i)},t_i)\bigr\}_{i=1}^{N}$$.

SINDy requires an estimate of $$\dot{x}(t_i)$$. A basic forward difference is

$$
\hat{\dot{x}}^{(i)} \;=\; \frac{x^{(i+1)} - x^{(i)}}{\,t_{i+1} - t_i\,}, 
\quad i=1,\dots,(N-1).
$$

Here, $$\hat{\dot{x}}^{(i)}$$ approximates $$\dot{x}(t_i)$$. One may also use
centered differences, polynomial fitting, or other techniques to improve
accuracy. A **feature library** $$\Phi$$ is a set of candidate functions:

$$
\Phi(x) \;=\;
\begin{bmatrix}
\phi_1(x)\\
\phi_2(x)\\
\vdots\\
\phi_M(x)
\end{bmatrix}
\quad
\Longrightarrow
\quad
\dot{x} \;=\; \Phi(x) \cdot \,\boldsymbol{w}.
$$

Common examples:
1. **Polynomials**: $$1$$, $$x$$, $$x^2$$, $$x^3$$, $$\dots$$.
2. **Trigonometric**: $$\sin(x)$$, $$\cos(x)$$.
3. **Exponentials**: $$e^x$$, etc.


For each measurement $$x^{(i)}$$ ($$i = 1,\dots,N-1$$), we evaluate every
candidate function in $$\Phi$$. This yields a **design matrix** $$X \in
\mathbb{R}^{(N-1) \times M}$$. For example, if $$M=4$$ with $$\bigl\{1,\, x,\,
x^2,\, \sin(x)\bigr\}$$:

$$
X \;=\;
\begin{bmatrix}
1 & x^{(1)} & \bigl(x^{(1)}\bigr)^2 & \sin\bigl(x^{(1)}\bigr)\\
1 & x^{(2)} & \bigl(x^{(2)}\bigr)^2 & \sin\bigl(x^{(2)}\bigr)\\
\vdots & \vdots & \vdots & \vdots\\
1 & x^{(N-1)} & \bigl(x^{(N-1)}\bigr)^2 & \sin\bigl(x^{(N-1)}\bigr)
\end{bmatrix}.
$$

We arrange our time-derivative approximations $$\hat{\dot{x}}^{(i)}$$ into a
vector

$$
\hat{\mathbf{\dot{x}}}
\;=\;
\begin{bmatrix}
\hat{\dot{x}}^{(1)}\\
\hat{\dot{x}}^{(2)}\\
\vdots\\
\hat{\dot{x}}^{(N-1)}
\end{bmatrix}.
$$

Then the SINDy problem in its simplest form is:

$$
\min_{\boldsymbol{w}} 
\;\bigl\|\hat{\mathbf{\dot{x}}} - X\,\boldsymbol{w}\bigr\|_2^2.
$$

Equivalently, for each data row $$i$$, we want:

$$
\hat{\dot{x}}^{(i)} \;\approx\; 
\underbrace{
[\;1,\; x^{(i)},\;\bigl(x^{(i)}\bigr)^2,\;\sin\bigl(x^{(i)}\bigr)\;]
}_{\text{the row } i \text{ of }X}
\;\cdot\;
\underbrace{
[w_1,\; w_2,\; w_3,\; w_4]
}_{\boldsymbol{w}}.
$$

---


Let’s say $$N=6$$. We have 6 derivative approximations $$\hat{\dot{x}}^{(1)},
\dots, \hat{\dot{x}}^{(6)}$$. With 4 candidate features ($$1, x, x^2,
\sin(x)$$), the design matrix $$X$$ is $$6\times4$$. In full:

$$
\hat{\mathbf{\dot{x}}} 
=
\begin{bmatrix}
\hat{\dot{x}}^{(1)}\\
\hat{\dot{x}}^{(2)}\\
\hat{\dot{x}}^{(3)}\\
\hat{\dot{x}}^{(4)}\\
\hat{\dot{x}}^{(5)}\\
\hat{\dot{x}}^{(6)}
\end{bmatrix}, 
\quad
X = 
\begin{bmatrix}
1 & x^{(1)} & (x^{(1)})^2 & \sin\bigl(x^{(1)}\bigr)\\
1 & x^{(2)} & (x^{(2)})^2 & \sin\bigl(x^{(2)}\bigr)\\
1 & x^{(3)} & (x^{(3)})^2 & \sin\bigl(x^{(3)}\bigr)\\
1 & x^{(4)} & (x^{(4)})^2 & \sin\bigl(x^{(4)}\bigr)\\
1 & x^{(5)} & (x^{(5)})^2 & \sin\bigl(x^{(5)}\bigr)\\
1 & x^{(6)} & (x^{(6)})^2 & \sin\bigl(x^{(6)}\bigr)
\end{bmatrix},
\quad
\boldsymbol{w} = 
\begin{bmatrix}
w_1\\[3pt]
w_2\\[3pt]
w_3\\[3pt]
w_4
\end{bmatrix}.
$$

Hence,

$$
\hat{\mathbf{\dot{x}}} - X\,\boldsymbol{w}
=
\begin{bmatrix}
\hat{\dot{x}}^{(1)}\\
\hat{\dot{x}}^{(2)}\\
\hat{\dot{x}}^{(3)}\\
\hat{\dot{x}}^{(4)}\\
\hat{\dot{x}}^{(5)}\\
\hat{\dot{x}}^{(6)}
\end{bmatrix}
-\;
\begin{bmatrix}
1 & x^{(1)} & \bigl(x^{(1)}\bigr)^2 & \sin\bigl(x^{(1)}\bigr)\\
1 & x^{(2)} & \bigl(x^{(2)}\bigr)^2 & \sin\bigl(x^{(2)}\bigr)\\
\vdots & \vdots & \vdots & \vdots\\
1 & x^{(6)} & \bigl(x^{(6)}\bigr)^2 & \sin\bigl(x^{(6)}\bigr)
\end{bmatrix}
\cdot
\begin{bmatrix}
w_1\\[4pt]
w_2\\[4pt]
w_3\\[4pt]
w_4
\end{bmatrix}
\;=\;\mathbf{0}.
$$

SINDy aims to **minimize** the norm of this residual to find the best
$$\boldsymbol{w}$$.

---

### 2.1 Enforcing Sparsity

Real systems often have **only a few dominant terms** in $$f$$. Thus, we impose
sparsity on $$\boldsymbol{w}$$. Common approaches:

1. **LASSO (L1 penalty)**:

$$
\min_{\boldsymbol{w}} 
\Bigl(
\|\hat{\mathbf{\dot{x}}} - X\,\boldsymbol{w}\|_2^2 
\;+\;\lambda \|\boldsymbol{w}\|_1
\Bigr).
$$

2. **Sequential Thresholding**:
   1. Solve regular least squares $$\|\hat{\mathbf{\dot{x}}} -
X\,\boldsymbol{w}\|_2^2$$.
   2. Set small coefficients in $$\boldsymbol{w}$$ (below a threshold
$$\epsilon$$) to 0.
   3. Repeat until convergence.

This isolates the few relevant terms, yielding a concise model.

---

## 3. Multiple Variables $$(x,y,z)$$

When the system has **multiple measurements** $$\bigl(x(t),y(t),z(t)\bigr)$$,
each variable can obey a distinct equation:

$$
\dot{x} 
\;=\; f_x\bigl(x,y,z\bigr), 
\quad 
\dot{y} 
\;=\; f_y\bigl(x,y,z\bigr),
\quad
\dot{z} 
\;=\; f_z\bigl(x,y,z\bigr).
$$

In matrix form, define a **combined** library $$\Phi(x,y,z)$$ for each data
point. Then we can “stack” the equations. One approach is to solve three
separate linear regressions:

$$
\begin{aligned}
\hat{\mathbf{\dot{x}}} &\approx X\,\boldsymbol{w}_x,\\
\hat{\mathbf{y}} &\approx X\,\boldsymbol{w}_y,\\
\hat{\mathbf{z}} &\approx X\,\boldsymbol{w}_z,
\end{aligned}
$$

where $$\boldsymbol{w}_x$$, $$\boldsymbol{w}_y$$, $$\boldsymbol{w}_z$$ are
independent coefficient vectors. Each corresponds to one of the three equations
($$\dot{x}, \dot{y}, \dot{z}$$). Alternatively, one may form a larger matrix
equation if it is convenient, but conceptually, it remains a set of **multiple
linear regressions**.

---

## 4. Extending SINDy to PDEs

For a system described by a **partial differential equation** in one spatial
dimension $$x$$ and time $$t$$:

$$
u_t 
\;\;=\;\; f\bigl(u,\,u_x,\,u_{xx},\dots\bigr),
$$
we have measurements 
$$
u(x_j,\,t_k)\quad \text{for} \quad j=1,\dots,J;\; k=1,\dots,K.
$$

We want to discover the function $$f$$. 

1. **Time Derivative $$\hat{u}_t$$**:  
   For fixed $$x_j$$, approximate the partial derivative $$\partial u/\partial
t$$ via a finite difference in $$t$$:
   $$
   \hat{u}_t(x_j,t_k)
   \;\approx\;
   \frac{u(x_j,\,t_{k+1}) - u(x_j,\,t_k)}{t_{k+1} - t_k}.
   $$
   
2. **Spatial Derivatives**:  
   For fixed $$t_k$$, approximate $$u_x$$ or $$u_{xx}$$ using finite
differences in $$x$$. For instance,

   $$
   \hat{u}_x(x_j,t_k)
   \;\approx\;
   \frac{u(x_{j+1},t_k) - u(x_{j-1},t_k)}{2\Delta x},
   \quad\;
   \hat{u}_{xx}(x_j,t_k)
   \;\approx\;
   \frac{u(x_{j+1},t_k) - 2u(x_j,t_k) + u(x_{j-1},t_k)}{(\Delta x)^2}.
   $$

   Other variants (central differences, higher-order schemes, etc.) can improve
accuracy.

At each grid point $$(x_j,t_k)$$, we evaluate a **PDE feature library**
$$\Phi$$ that might include
$$\bigl\{u,\,u^2,\,u_x,\,u_{xx},\,u\,u_x,\dots\bigr\}$$. Then the PDE assumption
is:

$$
\hat u_t(x_j,t_k) 
\;\approx\; 
\Phi\bigl(\hat u,\, \hat u_x,\, \hat u_{xx},\dots\bigr)
\,\boldsymbol{w},
$$

which is **linear** in the parameters $$\boldsymbol{w}$$. Stacking all grid
points (or a subset) gives a **large linear system**. Enforcing sparsity on
$$\boldsymbol{w}$$ reveals which terms are truly present in the PDE.

This extension can uncover PDEs—like advection-diffusion, Korteweg–de Vries, or
wave equations—from spatio-temporal data.

---

## 5. Pseudo-Code Example

Below is a concise Python pseudo-code illustrating **SINDy for a single
time-series**. The same logic extends to multiple variables or PDEs—only the
library construction (and derivative approximations) becomes more elaborate.

```python
import numpy as np
from sklearn.linear_model import Lasso

def compute_derivatives_1D(x, t):
    # Simple forward difference
    return (x[1:] - x[:-1]) / (t[1:] - t[:-1])

def build_feature_library_1D(x):
    # Example library: [1, x, x^2, sin(x)]
    return np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        np.sin(x)
    ])

# Example data: x(t) at 7 time points
x_data = np.array([1.0, 2.1, 3.0, 4.2, 5.1, 5.8, 6.2])  
t_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# 1) Numerically approximate derivatives (length = 6)
x_dot_approx = compute_derivatives_1D(x_data, t_data)

# 2) Build design matrix (length = 6 x 4)
X = build_feature_library_1D(x_data[:-1])  # match dimension

# 3) Fit with LASSO for sparsity
lasso = Lasso(alpha=0.1)
lasso.fit(X, x_dot_approx)
w_spar = lasso.coef_

print("Identified sparse coefficients:", w_spar)
```

- **`build_feature_library_1D`** could be extended to include more candidate functions.
- For **multi-variable** or **PDE** data, we would similarly:
  1. Approximate partial derivatives $$\hat{u}_t,\, \hat{u}_x,\,
\hat{u}_{xx},\dots$$.
  2. Build a matrix with columns for each candidate term.
  3. Solve a **sparse regression**.

---

## 9. Summary

1. **Formulate a Library**: List candidate functions of the system variables
(e.g. $$\{1, x, x^2, \dots\}$$ for ODEs; $$\{u, u^2, u_x, u_{xx}, \dots\}$$ for
PDEs).  
2. **Approximate Derivatives**: Use discrete data to numerically approximate
$$\hat{\dot{x}}$$ or $$\hat{u}_t, \hat{u}_x, \hat{u}_{xx}, \dots$$.  
3. **Build a Design Matrix**: Evaluate each candidate function at each data
point, yielding $$X$$.  
4. **Sparse Regression**: Solve  
   $$
   \min_{\boldsymbol{w}} \bigl\|\hat{\mathbf{\dot{x}}} -
X\,\boldsymbol{w}\bigr\|_2^2 
   \;+\;\lambda\|\boldsymbol{w}\|_1 
   \quad\text{(or with thresholding)}.
   $$
   The resulting $$\boldsymbol{w}$$ picks out a small set of nonzero terms.  

