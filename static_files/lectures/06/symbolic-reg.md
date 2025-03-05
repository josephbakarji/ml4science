#lize lecture: symbolic regression with genetic algorithms

## 1. introduction and motivation

### 1.1 what is symbolic regression?

in **symbolic regression**, we aim to discover a **mathematical expression**
(symbolic function) that best fits observed data
$$\{(x^{(i)},\,y^{(i)})\}_{i=1}^n$$. in contrast to conventional regression,
which fits parameters to a predetermined functional form (e.g., a polynomial of
fixed degree or a neural network architecture), **symbolic regression** searches
over the space of possible **functional forms**—often an extremely large space.

for example, suppose you have input $$x$$ and measured output $$y$$. a symbolic
regression tool might propose:

$$
y \;\approx\; \sin(x) + \frac{1}{2}\log(x + 2),
$$

or more generally, any expression from a predefined set of operations (e.g.,
$$+,\; -,\; \times,\; \div,\; \sin,\; \log,\; \exp,\dots$$).

### 1.2 why use symbolic regression?

1. **interpretability**: the discovered expressions are explicit formulas that
are often far more interpretable than a black-box model (like a random forest or
deep neural network).
2. **compactness**: symbolic regression can lead to **parsimonious** solutions,
capturing the essence of relationships in a relatively small
expression—potentially revealing underlying physics or domain insights.
3. **generalization**: if the derived formula is correct (or close to correct),
it can extrapolate better to unseen data than purely data-driven methods that do
not assume a structural form.


## 2. genetic algorithms for symbolic regression

### 2.1 basic idea of genetic programming

symbolic regression via **genetic programming** (gp) uses evolutionary concepts:

1. **representation**: each candidate solution is typically represented as an
**expression tree**. for instance, an expression like $$\sin(x) +
0.5\,\log(x+2)$$ can be written as a tree with `+` at the root and two subtrees
for $$\sin(x)$$ and $$0.5\,\log(x+2)$$.
2. **population**: maintain a population (a set) of such candidate expression
trees.
3. **fitness**: evaluate how well each tree fits the data (e.g., mean squared
error).
4. **selection**: choose the better candidates for reproduction.
5. **crossover**: randomly exchange subtrees between two parent trees.
6. **mutation**: randomly alter parts of a tree (e.g., replace a function node
or numeric constant).
7. **iteration**: over many generations, the population evolves to (hopefully)
produce simpler and more accurate expressions.

### 2.2 key components

- **search space**: defined by a “grammar” of functions/operators (e.g., `+`, `-`, `*`, `/`, `sin`, `cos`, etc.) plus ephemeral random constants (numeric values).
- **fitness metric**: often the **error** on the training set, possibly regularized to encourage simplicity (e.g., penalize large trees).
- **selection strategy**: e.g., **tournament selection** or **roulette wheel selection**.
- **crossover & mutation**: make sure that the resulting offspring remain valid expression trees.



## 3. outline of a genetic algorithm for symbolic regression

below is **pseudo-code** capturing the main loop of a **genetic programming**
approach to symbolic regression:

```text
initialize population p with random expression trees 
for generation in [1..gmax]:
    # evaluate fitness
    for individual in p:
        y_pred = evaluate_expression_tree(individual, x)  
        individual.fitness = compute_mse(y_pred, y)
                         + alpha * complexity_penalty(individual)
    
    # select parents
    p_selected = selection_operator(p)  # e.g., tournament
    
    # create offspring via crossover & mutation
    p_offspring = []
    while len(p_offspring) < population_size:
        parent1, parent2 = choose_two_parents(p_selected)
        child1, child2 = crossover(parent1, parent2)
        child1 = maybe_mutate(child1, mutation_rate)
        child2 = maybe_mutate(child2, mutation_rate)
        p_offspring.add(child1)
        p_offspring.add(child2)
    
    # form new population
    p = best_elites_from(p) + p_offspring  # keep top individuals (elitism)
    
# return best individual in final population
best_solution = argmin_fitness_in(p)
```

### explanation

- **initialization**: start with random trees of a limited depth, using random functions and constants.
- **evaluation**:
  - `evaluate_expression_tree(...)` interprets the tree to produce predictions
$$\hat{y}$$.
  - `compute_mse(...)` calculates mean squared error (or another error metric).
  - `complexity_penalty(...)` can measure expression tree size to promote
simpler solutions.
- **selection**: based on fitness (lower error = better).
- **crossover**: exchange random subtrees between parent expressions.
- **mutation**: randomly change node types (e.g., replace `x` with `sin(x)`, or replace `-` with `+`, etc.).
- **elitism**: often keep a fraction of the best solutions from one generation to the next to preserve high-fitness individuals.


## 4. example python-like code snippet

in python, frameworks such as **[deap](https://deap.readthedocs.io/)** or
**[pysr](https://github.com/milescranmer/pysr)** implement symbolic regression
with a genetic or evolutionary approach. below is a **minimal** illustrative
code snippet (not fully functional but capturing essential steps).

```python
import random
import numpy as np

# suppose we have a set of unary/binary functions
functions = [("add", 2), ("sub", 2), ("mul", 2), ("div", 2),
             ("sin", 1), ("cos", 1), ("log", 1)]
constant_range = (-1.0, 1.0)

class node:
    def __init__(self, func=none, children=none, value=none):
        self.func = func        # e.g. ("mul", 2) or ("sin", 1)
        self.children = children if children else []
        self.value = value      # for leaf constants or x

def generate_random_tree(depth=3):
    """recursively create a random expression tree."""
    if depth == 0 or random.random() < 0.3:
        # 50% chance to be a variable or random constant
        if random.random() < 0.5:
            node = node(value="x")  # variable
        else:
            node = node(value=random.uniform(*constant_range))
        return node
    else:
        func = random.choice(functions)
        node = node(func=func)
        arity = func[1]
        for _ in range(arity):
            node.children.append(generate_random_tree(depth-1))
        return node

def evaluate_tree(node, x):
    """evaluate expression tree for input x."""
    if node.func is none:
        # leaf node
        if isinstance(node.value, str) and node.value == "x":
            return x
        else:
            return float(node.value)
    else:
        # internal function node
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

# example usage:
x = np.linspace(-2, 2, 20)
y = 1.0 + 0.5 * np.sin(x)  # example "true" function

# generate a random expression tree
individual = generate_random_tree(depth=2)
# evaluate mse
preds = np.array([evaluate_tree(individual, xval) for xval in x])
mse = np.mean((y - preds)**2)
print("initial random tree mse:", mse)
```

**note**:  
- **mutation and crossover** would be implemented similarly, e.g., picking a random subtree to swap or random node to alter.  
- a real ga would maintain a population of trees, run multiple generations, etc.


## 5. pysr: symbolic regression in python

### 5.1 overview

- **pysr** is a python package combining **high-performance** searching (written in julia) with a user-friendly python interface.
- it uses a combination of techniques, including evolutionary search and gradient-based refinement (optional).
- **pysr** specifically tries to keep track of **pareto fronts** of solutions, balancing complexity (size of the expression) vs. loss.

### 5.2 minimal usage example

```python
!pip install pysr

import numpy as np
from pysr import pysr, best

# sample dataset
x = np.linspace(-5, 5, 100)
y = 1.0 + 0.5 * np.sin(x)  # true function + noise if desired

# pysr expects 2d arrays for x
x_2d = x.reshape(-1, 1)

equations = pysr(
    x_2d, y,
    niterations=1000,
    unary_operators=["sin", "cos", "exp", "log"],
    binary_operators=["+", "-", "*", "/"],
    procs=4,
)

print(equations)
best_equation = best(equations)  # returns best discovered formula
print("best equation found by pysr:", best_equation)
```

**key arguments**:

- `x_2d`: input features of shape $$(n,\; d)$$.
- `y`: target of shape $$(n,)$$.
- `niterations`: number of evolutionary steps.
- `unary_operators`, `binary_operators`: define the function set.
- `procs`: parallelism for speed.

**output**: the library prints a table of discovered equations with their complexity and score (mse or a chosen metric). a typical best solution might be something like `0.9999 + 0.50*sin(x_0)`.


## 6. discussion and practical tips

1. **initialization**: the depth of initial trees strongly impacts performance.
deeper trees can explore more complex expressions early, but also risk higher
initial complexity.
2. **premature convergence**: genetic algorithms can converge to suboptimal
solutions. including diversity-preserving strategies (e.g., random mutations)
helps maintain exploration.
3. **regularization**: add a penalty for large trees (complexity) to favor
interpretable, simpler expressions.
4. **runtime**: symbolic regression is computationally expensive.
parallelization or gpu acceleration can help.
5. **noise**: real-world data may have noise/outliers. consider robust fitness
metrics (e.g., median error, or a combined cost function with an outlier
penalty).


## 7. conclusion

**symbolic regression** via **genetic algorithms** offers a powerful, interpretable approach for discovering mathematical relationships directly from data. by representing candidate solutions as expression trees and applying evolutionary operators (selection, crossover, mutation), these methods iteratively refine functional forms that balance **accuracy** and **simplicity**. tools like **pysr** make these methods more accessible, providing a high-level interface for exploring symbolic models.

- symbolic regression **searches** over functional forms, not just parameters.
- **genetic programming** is a flexible technique for evolving expressions.
- modern packages (e.g., **pysr**) combine evolutionary methods with efficient implementations to handle moderate data sizes.
- symbolic formulas can yield **interpretable** and often **parsimonious** solutions, valuable for many scientific and engineering applications.

---

### further reading

- **john r. koza**, *genetic programming: on the programming of computers by means of natural selection*, 1992.
- **michael schmidt and hod lipson**, “distilling free-form natural laws from experimental data”, *science*, 2009.
- **pysr** github: [https://github.com/milescranmer/pysr](https://github.com/milescranmer/pysr)
- **deap**: a general framework for evolutionary algorithms in python: [https://deap.readthedocs.io/](https://deap.readthedocs.io/)
