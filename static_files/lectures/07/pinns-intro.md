## Physics-Informed Neural Networks (PINNs) and Residual Minimization

### Introduction: Residual Minimization from a Basis Function Expansion

Consider the classical problem of finding a function $$ u(x, t) $$ that
satisfies a partial differential equation (PDE). A common strategy is to propose
a solution as a linear combination of known basis functions:

$$
u(x, t) = \sum_{i=1}^N a_i \phi_i(x, t)
$$

where:

- $$ \phi_i(x, t) $$ are known basis functions (often eigenfunctions or Fourier modes).
- $$ a_i $$ are unknown coefficients to be determined from data.

Given a PDE, for example, the heat equation:

$$
u_t = \alpha \frac{\partial^2 u}{\partial x^2},
$$

we can substitute the hypothesis directly into the PDE, obtaining a
**residual**:

$$
\mathcal{R}(x, t; a_i) = \frac{\partial u}{\partial t} - \alpha
\frac{\partial^2 u}{\partial x^2}
$$

Explicitly, using our decomposition:

$$
\mathcal{R}(x, t; a_i) = \sum_{i=1}^N a_i \frac{\partial \phi_i(x, t)}{\partial
t} - \alpha \sum_{i=1}^N a_i \frac{\partial^2 \phi_i(x, t)}{\partial x^2}
$$

### Residual Minimization with Data

Given observed data points $$(x_j, t_j, u_j^{\text{data}})$$, we find the
coefficients $$ a_i $$ by minimizing the following loss function:

$$
\mathcal{L}(a_i) = 
\underbrace{\sum_{j=1}^{N_d} \left[ u^{\text{data}}_j - \sum_{i=1}^N a_i
\phi_i(x_j, t_j) \right]^2}_{\text{Data-fitting term}} 
+ 
\underbrace{\sum_{k=1}^{N_r} \mathcal{R}(x_k, t_k; a_i)^2}_{\text{Residual
term}}
$$

- Minimizing the residual term ensures the solution respects the governing PDE.
- Minimizing the data-fitting term ensures the solution matches observed data.

This approach, known as **residual minimization** or **collocation methods**,
has a long tradition in applied mathematics and numerical analysis, closely
related to **spectral methods**, **Galerkin methods**, and **method of weighted
residuals**.



### Relationship to Physics-Informed Neural Networks (PINNs)

PINNs generalize this concept by replacing the linear combination of basis
functions with a **neural network** as a universal approximator for the solution
$$ u(x,t) $$:

$$
u(x, t) \approx u_\theta(x, t)
$$

- Here, $$\theta$$ are the parameters (weights and biases) of the neural network.
- The neural network acts as a flexible hypothesis function without explicitly selecting basis functions.
- **Automatic differentiation** computes derivatives efficiently, allowing easy evaluation of the PDE residual.

Thus, the residual is now defined as:

$$
\mathcal{R}_\theta(x, t) = \frac{\partial u_\theta}{\partial t}(x,t) - \alpha
\frac{\partial^2 u_\theta}{\partial x^2}(x, t)
$$

And the optimization problem becomes:

$$
\mathcal{L}(\theta) = 
\underbrace{\sum_{j=1}^{N_d}\left[ u_\theta(x_j, t_j) -
u_j^{\text{data}}\right]^2}_{\text{Data fitting term}}
+
\underbrace{\sum_{k=1}^{N_r}\left| \frac{\partial u_\theta}{\partial t}(x_k,
t_k) - \alpha \frac{\partial^2 u_\theta}{\partial x^2}(x_k,
t_k)\right|^2}_{\text{Residual minimization term}}
$$

The parameters $$\theta$$ are then optimized through standard deep-learning
techniques (gradient-based optimization, backpropagation, etc.).


### Literature Context

This neural-network-based method, introduced prominently by Raissi et al. in
their seminal work:

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*.

The general approach of using basis expansions and optimizing coefficients via
residuals is classical and falls under the umbrella of **collocation methods**
or **Galerkin methods**. The specific neural-network-based variant is precisely
what is referred to today as **Physics-Informed Neural Networks (PINNs)**.


### PyTorch Example (Pseudo-code):

Here's a concise, clear example in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Neural network architecture
class PINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.stack([x, t], dim=1)
        return self.model(inputs).squeeze()

# PDE residual computation (e.g., heat equation)
def residual(model, x, t, alpha):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    return u_t - alpha * u_xx

# Loss function combining residual and data
def loss_fn(model, x_data, t_data, u_data, x_r, t_r, alpha):
    u_pred = model(x_d, t_d)
    loss_data = torch.mean((u_pred - u_data)**2)

    res = residual(model, x_r, t_r, alpha)
    loss_res = torch.mean(res**2)

    return loss_res + loss_data

# Training loop (simplified)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_fn(model, x_res, t_res, x_data, t_data, u_data, alpha)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```


### Key Takeaways

- **Residual minimization** is foundational to physics-informed modeling.
- PINNs generalize classical collocation methods with neural network flexibility.
- PINNs leverage automatic differentiation, simplifying the handling of derivatives.
- PINNs inherently integrate physical laws and data-driven learning, making them powerful tools in scientific machine learning.