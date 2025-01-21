```python
import numpy as np

# Measurement data 
inputs = [1, .5, 2, .3, .2]  # Weight in Kgs
outputs = [55, 37.5, 89.8, 31.8, 31]  # Length in cm 

# Learning rate
eta = 0.01

# Initial guess for the weights
w = np.array([0, 0])

# Number of iterations
n_iterations = 500

# Define the design matrix
X = np.array([np.ones(len(inputs)), inputs]).T

def gd(X, outputs, w, eta, n_iterations):
    # Gradient descent
    w_history = []  
    for i in range(n_iterations):
        gradients = -2 * X.T.dot(outputs - X.dot(w))
        w = w - eta * gradients
        w_history.append(w)
    return w_history

def model(inputs, w):
    X = np.array([np.ones(len(inputs)), inputs]).T
    # Linear regression model
    return X.T.dot(w) 

w_history = gd(X, outputs, w, eta, n_iterations)
w = w_history[-1]

# Predicted lengths for new weights
new_weights = [1.5, .8]
inputs_query = np.array([np.ones(len(new_weights)), new_weights])
predicted_lengths = inputs_query.T.dot(w)

print("Predicted Lengths:", predicted_lengths)
```

    Predicted Lengths: [72.4635645  49.02000743]



```python
# Plot predicted data (in red) and original data (in blue)

range_data = [0.2, 2]
inputs_query_rd = np.array([np.ones(len(range_data)), range_data])
pred_range_data = inputs_query_rd.T.dot(w)

import matplotlib.pyplot as plt
# plt.plot(range_data, pred_range_data, 'k--', label='Model')
plt.plot(inputs, outputs, 'bo')
# plt.plot(new_weights, predicted_lengths, 'ro', label='Test Data')
plt.xlabel("Weight (Kgs)")
plt.ylabel("Length (cm)")
# plt.legend()
plt.show()
```


    
![png](output_1_0.png)
    



```python
# Plot predicted data (in red) and original data (in blue)

range_data = [0.2, 2]
inputs_query_rd = np.array([np.ones(len(range_data)), range_data])
pred_range_data = inputs_query_rd.T.dot(w)

import matplotlib.pyplot as plt
plt.plot(range_data, pred_range_data, 'k--', label='Model')
plt.plot(inputs, outputs, 'bo', label='Measurement (Training) data')
plt.plot(new_weights, predicted_lengths, 'ro', label='Test Data')
plt.xlabel("Weight (Kgs)")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

```


    
![png](output_2_0.png)
    



```python
# Plot history of weights on two subplots side by side
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot([w[0] for w in w_history], label='w0', lw=2)
ax2.plot([w[1] for w in w_history], label='w1', lw=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$$$w_0$$$')
ax2.set_xlabel('Iteration')
ax2.set_ylabel(r'$$$w_1$$$')

plt.show()

```


    
![png](output_3_0.png)
    



```python
## perform gradient descent for polynomial features of degrees 1, 2, and 3, and compare

# Polynomial features of degree 1
X = np.array([np.ones(len(inputs)), inputs]).T
w = np.array([0, 0])
w_history = gd(X, outputs, w, eta, n_iterations)
w_1 = w_history[-1]
print("Weights for degree 1:", w_1, 'total loss: ', np.sum((X.dot(w_1) - outputs)**2))

# Polynomial features of degree 2
X = np.array([np.ones(len(inputs)), inputs, np.array(inputs)**2]).T
w = np.array([0, 0, 0])
w_history = gd(X, outputs, w, eta, n_iterations)
w_2 = w_history[-1]
print("Weights for degree 2:", w_2, 'total loss: ', np.sum((X.dot(w_2) - outputs)**2))

# Polynomial features of degree 3
X = np.array([np.ones(len(inputs)), inputs, np.array(inputs)**2, np.array(inputs)**3]).T
w = np.array([0, 0, 0, 0])
w_history = gd(X, outputs, w, eta, n_iterations)
w_3 = w_history[-1]
print("Weights for degree 3:", w_3, 'total loss: ', np.sum((X.dot(w_3) - outputs)**2))

# Polynomial features of degree 4
X = np.array([np.ones(len(inputs)), inputs, np.array(inputs)**2, np.array(inputs)**3, np.array(inputs)**4]).T
w = np.array([0, 0, 0, 0, 0])
w_history = gd(X, outputs, w, eta, n_iterations)
w_4 = w_history[-1]
print("Weights for degree 4:", w_4, 'total loss: ', np.sum((X.dot(w_4) - outputs)**2))

# Polynomial features of degree 5
X = np.array([np.ones(len(inputs)), inputs, np.array(inputs)**2, np.array(inputs)**3, np.array(inputs)**4, np.array(inputs)**5]).T
w = np.array([0, 0, 0, 0, 0, 0])
w_history = gd(X, outputs, w, eta, n_iterations)
w_5 = w_history[-1]
print("Weights for degree 5:", w_5, 'total loss: ', np.sum((X.dot(w_5) - outputs)**2))


```

    Weights for degree 1: [22.22737077 33.49079582] total loss:  7.562816515981742
    Weights for degree 2: [26.54254583 20.3970508   5.75462664] total loss:  8.011928541796815
    Weights for degree 3: [25.84970985 19.3962097  11.80085843 -2.74490989] total loss:  2.3314007953559988
    Weights for degree 4: [nan nan nan nan nan] total loss:  nan
    Weights for degree 5: [nan nan nan nan nan nan] total loss:  nan


    /var/folders/wq/rd7c2mhn7fs9y313qjs3c58r0000gn/T/ipykernel_67677/3743879246.py:23: RuntimeWarning: overflow encountered in multiply
      gradients = -2 * X.T.dot(outputs - X.dot(w))
    /var/folders/wq/rd7c2mhn7fs9y313qjs3c58r0000gn/T/ipykernel_67677/3743879246.py:24: RuntimeWarning: invalid value encountered in subtract
      w = w - eta * gradients



```python
# Plot predicted data (in red) and original data (in blue)

range_data = np.linspace(0.2, 2, 10)
inputs_query_rd = np.array([np.ones(len(range_data)), range_data])
pred_range_data_1 = inputs_query_rd.T.dot(w_1)

inputs_query_rd = np.array([np.ones(len(range_data)), range_data, np.array(range_data)**2])
pred_range_data_2 = inputs_query_rd.T.dot(w_2)

inputs_query_rd = np.array([np.ones(len(range_data)), range_data, np.array(range_data)**2, np.array(range_data)**3])
pred_range_data_3 = inputs_query_rd.T.dot(w_3)

import matplotlib.pyplot as plt
plt.plot(range_data, pred_range_data_1, 'k-', label='Model (degree 1)')
# plt.plot(range_data, pred_range_data_2, 'g-', label='Model (degree 2)')
plt.plot(range_data, pred_range_data_3, 'r-', label='Model (degree 3)')
plt.plot(inputs, outputs, 'bo', label='Measurement (Training) data')
# plt.plot(new_weights, predicted_lengths, 'ro', label='Test Data')
plt.xlabel("Weight (Kgs)")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()


```


    
![png](output_5_0.png)
    

