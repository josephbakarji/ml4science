# Symbolic Regression with Genetic Algorithms

## 1. Introduction and Motivation

### 1.1 What is Symbolic Regression?

In **symbolic regression**, we aim to discover a **mathematical expression**
(symbolic function) that best fits observed data
$$\{(x^{(i)},\,y^{(i)})\}_{i=1}^N$$. In contrast to conventional regression,
which fits parameters to a predetermined functional form (e.g., a polynomial of
fixed degree or a neural network architecture), **symbolic regression** searches
over the space of possible **functional forms**—often an extremely large space.

For example, suppose you have input $$x$$ and measured output $$y$$. A symbolic
regression tool might propose:

$$
y \;\approx\; \sin(x) + \frac{1}{2}\log(x + 2),
$$

or more generally, any expression from a predefined set of operations (e.g.,
$$+,\; -,\; \times,\; \div,\; \sin,\; \log,\; \exp,\dots$$).

### 1.2 Why Use Symbolic Regression?

1. **Interpretability**: The discovered expressions are explicit formulas that
are often far more interpretable than a black-box model (like a random forest or
deep neural network).
2. **Compactness**: Symbolic regression can lead to **parsimonious** solutions,
capturing the essence of relationships in a relatively small
expression—potentially revealing underlying physics or domain insights.
3. **Generalization**: If the derived formula is correct (or close to correct),
it can extrapolate better to unseen data than purely data-driven methods that do
not assume a structural form.


## 2. Genetic Algorithms for Symbolic Regression

### 2.1 Basic Idea of Genetic Programming

Symbolic regression via **genetic programming** (GP) uses evolutionary concepts:

1. **Representation**: Each candidate solution is typically represented as an
**expression tree**. For instance, an expression like $$\sin(x) +
0.5\,\log(x+2)$$ can be written as a tree with `+` at the root and two subtrees
for $$\sin(x)$$ and $$0.5\,\log(x+2)$$.
2. **Population**: Maintain a population (a set) of such candidate expression
trees.
3. **Fitness**: Evaluate how well each tree fits the data (e.g., mean squared
error).
4. **Selection**: Choose the better candidates for reproduction.
5. **Crossover**: Randomly exchange subtrees between two parent trees.
6. **Mutation**: Randomly alter parts of a tree (e.g., replace a function node
or numeric constant).
7. **Iteration**: Over many generations, the population evolves to (hopefully)
produce simpler and more accurate expressions.

### 2.2 Key Components

- **Search space**: Defined by a “grammar” of functions/operators (e.g., `+`, `-`, `*`, `/`, `sin`, `cos`, etc.) plus ephemeral random constants (numeric values).
- **Fitness metric**: Often the **error** on the training set, possibly regularized to encourage simplicity (e.g., penalize large trees).
- **Selection strategy**: e.g., **tournament selection** or **roulette wheel selection**.
- **Crossover & mutation**: Make sure that the resulting offspring remain valid expression trees.



## 3. Outline of a Genetic Algorithm for Symbolic Regression

Below is **pseudo-code** capturing the main loop of a **genetic programming**
approach to symbolic regression:

```text
Initialize population P with random expression trees 
for generation in [1..Gmax]:
    # Evaluate Fitness
    for individual in P:
        y_pred = evaluate_expression_tree(individual, X)  
        individual.fitness = compute_mse(y_pred, Y)
                         + alpha * complexity_penalty(individual)
    
    # Select parents
    P_selected = selection_operator(P)  # e.g., tournament
    
    # Create offspring via Crossover & Mutation
    P_offspring = []
    while len(P_offspring) < population_size:
        parent1, parent2 = choose_two_parents(P_selected)
        child1, child2 = crossover(parent1, parent2)
        child1 = maybe_mutate(child1, mutation_rate)
        child2 = maybe_mutate(child2, mutation_rate)
        P_offspring.add(child1)
        P_offspring.add(child2)
    
    # Form new population
    P = best_elites_from(P) + P_offspring  # keep top individuals (elitism)
    
# Return best individual in final population
best_solution = argmin_fitness_in(P)
```

### Explanation

- **Initialization**: Start with random trees of a limited depth, using random functions and constants.
- **Evaluation**:
  - `evaluate_expression_tree(...)` interprets the tree to produce predictions
$$\hat{y}$$.
  - `compute_mse(...)` calculates mean squared error (or another error metric).
  - `complexity_penalty(...)` can measure expression tree size to promote
simpler solutions.
- **Selection**: Based on fitness (lower error = better).
- **Crossover**: Exchange random subtrees between parent expressions.
- **Mutation**: Randomly change node types (e.g., replace `x` with `sin(x)`, or replace `-` with `+`, etc.).
- **Elitism**: Often keep a fraction of the best solutions from one generation to the next to preserve high-fitness individuals.


## 4. Example Python-Like Code Snippet

In Python, frameworks such as **[DEAP](https://deap.readthedocs.io/)** or
**[PySR](https://github.com/MilesCranmer/PySR)** implement symbolic regression
with a genetic or evolutionary approach. Below is a **minimal** illustrative
code snippet (not fully functional but capturing essential steps).

```python
import random
import numpy as np

# Suppose we have a set of unary/binary functions
FUNCTIONS = [("add", 2), ("sub", 2), ("mul", 2), ("div", 2),
             ("sin", 1), ("cos", 1), ("log", 1)]
CONSTANT_RANGE = (-1.0, 1.0)

class Node:
    def __init__(self, func=None, children=None, value=None):
        self.func = func        # e.g. ("mul", 2) or ("sin", 1)
        self.children = children if children else []
        self.value = value      # For leaf constants or x

def generate_random_tree(depth=3):
    """Recursively create a random expression tree."""
    if depth == 0 or random.random() < 0.3:
        # 50% chance to be a variable or random constant
        if random.random() < 0.5:
            node = Node(value="x")  # variable
        else:
            node = Node(value=random.uniform(*CONSTANT_RANGE))
        return node
    else:
        func = random.choice(FUNCTIONS)
        node = Node(func=func)
        arity = func[1]
        for _ in range(arity):
            node.children.append(generate_random_tree(depth-1))
        return node

def evaluate_tree(node, x):
    """Evaluate expression tree for input x."""
    if node.func is None:
        # Leaf node
        if isinstance(node.value, str) and node.value == "x":
            return x
        else:
            return float(node.value)
    else:
        # Internal function node
        name, arity = node.func
        vals = [evaluate_tree(child, x) for child in node.children]
        if name == "add":
            return vals[0] + vals[1]
        elif name == "sub":
            return vals[0] - vals[1]
        elif name == "mul":
            return vals[0] * vals[1]
        elif name == "div":
            return vals[0] / vals[1] if abs(vals[1])>1e-9 else 1.0
        elif name == "sin":
            return np.sin(vals[0])
        elif name == "cos":
            return np.cos(vals[0])
        elif name == "log":
            return np.log(abs(vals[0])+1e-9)
        # ... etc.

# Example usage:
X = np.linspace(-2, 2, 20)
Y = 1.0 + 0.5 * np.sin(X)  # Example "true" function

# Generate a random expression tree
individual = generate_random_tree(depth=2)
# Evaluate MSE
preds = np.array([evaluate_tree(individual, xval) for xval in X])
mse = np.mean((Y - preds)**2)
print("Initial random tree MSE:", mse)
```

**Note**:  
- **Mutation and crossover** would be implemented similarly, e.g., picking a random subtree to swap or random node to alter.  
- A real GA would maintain a population of trees, run multiple generations, etc.


## 5. PySR: Symbolic Regression in Python

### 5.1 Overview

- **PySR** is a Python package combining **high-performance** searching (written in Julia) with a user-friendly Python interface.
- It uses a combination of techniques, including evolutionary search and gradient-based refinement (optional).
- **PySR** specifically tries to keep track of **Pareto fronts** of solutions, balancing complexity (size of the expression) vs. loss.

### 5.2 Minimal Usage Example

```python
!pip install pysr

import numpy as np
from pysr import pysr, best

# Sample dataset
X = np.linspace(-5, 5, 100)
Y = 1.0 + 0.5 * np.sin(X)  # True function + noise if desired

# PySR expects 2D arrays for X
X_2D = X.reshape(-1, 1)

equations = pysr(
    X_2D, Y,
    niterations=1000,
    unary_operators=["sin", "cos", "exp", "log"],
    binary_operators=["+", "-", "*", "/"],
    procs=4,
)

print(equations)
best_equation = best(equations)  # Returns best discovered formula
print("Best equation found by PySR:", best_equation)
```

**Key arguments**:

- `X_2D`: input features of shape $$(N,\; d)$$.
- `Y`: target of shape $$(N,)$$.
- `niterations`: number of evolutionary steps.
- `unary_operators`, `binary_operators`: define the function set.
- `procs`: parallelism for speed.

**Output**: The library prints a table of discovered equations with their complexity and score (MSE or a chosen metric). A typical best solution might be something like `0.9999 + 0.50*sin(x_0)`.


## 6. Discussion and Practical Tips

1. **Initialization**: The depth of initial trees strongly impacts performance.
Deeper trees can explore more complex expressions early, but also risk higher
initial complexity.
2. **Premature Convergence**: Genetic algorithms can converge to suboptimal
solutions. Including diversity-preserving strategies (e.g., random mutations)
helps maintain exploration.
3. **Regularization**: Add a penalty for large trees (complexity) to favor
interpretable, simpler expressions.
4. **Runtime**: Symbolic regression is computationally expensive.
Parallelization or GPU acceleration can help.
5. **Noise**: Real-world data may have noise/outliers. Consider robust fitness
metrics (e.g., median error, or a combined cost function with an outlier
penalty).


## 7. Conclusion

**Symbolic regression** via **genetic algorithms** offers a powerful, interpretable approach for discovering mathematical relationships directly from data. By representing candidate solutions as expression trees and applying evolutionary operators (selection, crossover, mutation), these methods iteratively refine functional forms that balance **accuracy** and **simplicity**. Tools like **PySR** make these methods more accessible, providing a high-level interface for exploring symbolic models.

- Symbolic regression **searches** over functional forms, not just parameters.
- **Genetic programming** is a flexible technique for evolving expressions.
- Modern packages (e.g., **PySR**) combine evolutionary methods with efficient implementations to handle moderate data sizes.
- Symbolic formulas can yield **interpretable** and often **parsimonious** solutions, valuable for many scientific and engineering applications.

---

### Further Reading

- **John R. Koza**, *Genetic Programming: On the Programming of Computers by Means of Natural Selection*, 1992.
- **Michael Schmidt and Hod Lipson**, “Distilling Free-Form Natural Laws from Experimental Data”, *Science*, 2009.
- **PySR** GitHub: [https://github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
- **DEAP**: A general framework for evolutionary algorithms in Python: [https://deap.readthedocs.io/](https://deap.readthedocs.io/)
