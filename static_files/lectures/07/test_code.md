---
title: "------------------------------"
layout: note
permalink: /static_files/lectures/07/test_code/
---

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Neural Network
# ------------------------------
class PINN(nn.Module):
    def __init__(self, hidden_dim=20, hidden_layers=2):
        super(PINN, self).__init__()
        # Input layer
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

```


```python

# ------------------------------
# 2. Create the PINN instance
# ------------------------------
pinn = PINN(hidden_dim=20, hidden_layers=2)
optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)

# ------------------------------
# 3. Define Loss Functions
# ------------------------------
def physics_residual(x):
    """
    ODE: du/dx + u = 0
    The residual is r(x) = du/dx + u.
    """
    # Enable gradient calculation for x
    x.requires_grad = True
    
    # Forward pass: compute u(x)
    u = pinn(x)
    
    # Compute derivative du/dx via autograd
    grad_u = torch.autograd.grad(u, x, 
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
    
    # ODE residual
    r = grad_u + u
    return r

def loss_function(x_interior, x_boundary):
    # Physics loss
    residuals = physics_residual(x_interior)
    loss_physics = torch.mean(residuals**2)
    
    # Boundary/initial condition loss: u(0) = 1
    u_boundary = pinn(x_boundary)
    loss_boundary = torch.mean((u_boundary - 1.0)**2)
    
    return loss_physics + loss_boundary


```


```python

# ------------------------------------------
# 4. Training Loop
# ------------------------------------------
# Sample points in domain [0,1]
N_interior = 50
x_interior_np = np.random.rand(N_interior, 1)  # Random points in [0,1]
x_interior = torch.tensor(x_interior_np, dtype=torch.float32)

# Boundary point
x_boundary = torch.tensor([[0.0]], dtype=torch.float32)

num_iterations = 2000
loss_history = []

for i in range(num_iterations):
    optimizer.zero_grad()
    loss = loss_function(x_interior, x_boundary)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if (i+1) % 200 == 0:
        print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")


```

    Iteration 200/2000, Loss: 0.000066
    Iteration 400/2000, Loss: 0.000045
    Iteration 600/2000, Loss: 0.000034
    Iteration 800/2000, Loss: 0.000024
    Iteration 1000/2000, Loss: 0.000016
    Iteration 1200/2000, Loss: 0.000010
    Iteration 1400/2000, Loss: 0.000006
    Iteration 1600/2000, Loss: 0.000005
    Iteration 1800/2000, Loss: 0.000003
    Iteration 2000/2000, Loss: 0.000002



```python
# ------------------------------------------
# 5. Evaluate the Trained PINN
# ------------------------------------------
x_test_np = np.linspace(0, 1, 100)[:, None]
x_test = torch.tensor(x_test_np, dtype=torch.float32)
u_pred = pinn(x_test).detach().numpy()

# Analytical solution
u_true = np.exp(-x_test_np)

# Plot the PINN solution vs. analytical
plt.figure()
plt.plot(x_test_np, u_true, label='Analytical: e^{-x}')
plt.plot(x_test_np, u_pred, 'o', label='PINN Prediction', markersize=3)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('PINN Solution for du/dx + u = 0')
plt.show()

# Plot the loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.show()
```


    
![png](/static_files/lectures/07/test_code/output_3_0.png)
    



    
![png](/static_files/lectures/07/test_code/output_3_1.png)
    

