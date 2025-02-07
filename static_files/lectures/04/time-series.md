# Time Series Analysis

## 1. Context and Motivation

Over the past two weeks, we focused on the fundamental paradigm of **supervised learning**. The central objective is to approximate a function
\[
y \;=\; f_{\mathbf{w}}(\mathbf{x}),
\]
where \(\mathbf{x}\) denotes the input (or *independent*) variables and \(y\) the output (or *dependent*) variable. The model parameters \(\mathbf{w}\) are typically fitted by **minimizing a loss function**, for example:

\[
\mathcal{L}(\mathbf{w}) 
\;=\; \sum_{i=1}^{N} 
\bigl\|\,y^{(i)} - f_{\mathbf{w}}\bigl(\mathbf{x}^{(i)}\bigr)\bigr\|^{2}.
\]

### Linear Models and Feature Vectors

A *special* yet immensely important case is the **linear model**, where the function \(f_{\mathbf{w}}\) can be expressed as
\[
f_{\mathbf{w}}(\mathbf{x}) 
\;=\; \phi(\mathbf{x})^\top \mathbf{w}.
\]
Here, \(\phi(\mathbf{x})\) is called a **feature vector** (or **feature map**), and \(\mathbf{w}\) is the parameter vector. Crucially,

1. The map \(\phi(\mathbf{x})\) extracts relevant **features** from \(\mathbf{x}\).
2. The model is *linear* in the parameters \(\mathbf{w}\).  
3. Fitting \(\mathbf{w}\) often becomes a **least-squares** problem:
   \[
   \min_{\mathbf{w}} \sum_{i=1}^{N} 
   \Bigl(y^{(i)} - \phi\bigl(\mathbf{x}^{(i)}\bigr)^\top \mathbf{w}\Bigr)^2.
   \]

Despite its apparent simplicity, this framework underlies many powerful methods in **regression**, **classification**, and **time-series analysis** when the features \(\phi(\mathbf{x})\) are chosen or learned effectively.

### The Role of \(\mathbf{x}\) and \(y\) in Real-World Problems

In the physical world, the **input** \(\mathbf{x}\) might be:

- **Time** (for forecasting tasks).
- **Spatial coordinates** (e.g., for temperature or pressure fields).
- **Previous measurements** in a time series (e.g., past stock prices).
- **Control inputs** we can manipulate (in engineering or robotics).

Meanwhile, \(y\) (or \(\mathbf{y}\) if multivariate) could be:

- Future predictions of those same time-varying quantities (e.g., future temperature).
- Observed data in space and time (e.g., pollutant concentration, velocity fields, etc.).
- Outputs of interest (e.g., the predicted price of a stock).

Thus, **supervised learning** boils down to learning a mapping from measured (or controllable) inputs \(\mathbf{x}\) to desired outputs \(y\). By choosing appropriate **feature maps** \(\phi\) or by making \(\phi\) part of a deeper network (as in **deep learning**), we can capture a wide variety of phenomena.


In subsequent sections, we will see how these ideas apply specifically to **time-series** scenarios, where our “\(\mathbf{x}\)” often consists of *past observations* or *time indices*, and how we can leverage techniques from **Fourier transforms**, **auto-regression**, and **hidden-state** models to tackle the challenges of sequential data.

## 2. Fourier and Modal Decompositions

### 2.1 Approximating Functions via Orthogonal Bases

A classical strategy for understanding or predicting the behavior of a function \(x(t)\) (or a time series \(\{x_1, x_2, \dots\}\)) is to **represent** it as a sum of known basis functions. Formally, suppose you have an *orthonormal* set of functions \(\{\phi_1(t), \phi_2(t), \dots\}\) satisfying

\[
\langle \phi_i, \phi_j \rangle 
\;=\;
\int \phi_i(t)\,\phi_j(t)\,dt
\;=\;
\delta_{ij},
\]
where \(\delta_{ij}\) is the Kronecker delta (or 1 if \(i=j\), 0 otherwise). This is the generalization of a dot product being zero if the vectors are orthogonal.

In general, we often seek an **approximation** \(x(t)\) of the form:
\[
x(t) 
\;\approx\;
\sum_{k=1}^{K} a_k \,\phi_k(t),
\]
where \(a_k\) are coefficients to be determined. If you have measurements \(\{x(t_i)\}\) at discrete times \(\{t_i\}_{i=1}^N\), you can fit the coefficients \(a_k\) by **linear regression** (or least squares):

\[
\min_{\{a_k\}} \;\sum_{i=1}^{N} \Bigl[x(t_i) \;-\; \sum_{k=1}^{K} a_k\,\phi_k(t_i)\Bigr]^2.
\]

### 2.2 Fourier Decomposition as Linear Regression

A **Fourier series** is a specific example where each \(\phi_k\) is either \(\sin\bigl(2\pi f_k t\bigr)\) or \(\cos\bigl(2\pi f_k t\bigr)\). For simplicity, let’s define a discrete set of frequencies \(\{f_1, \dots, f_K\}\). Then, our model is:

\[
x(t) \;\approx\; 
\sum_{k=1}^{K} \Bigl[
w_k\,\sin\bigl(2\pi f_k\, t\bigr) 
\;+\;
v_k\,\cos\bigl(2\pi f_k\, t\bigr)
\Bigr].
\]

In **matrix form**, this is precisely a **linear regression** problem. You can construct a ``design matrix'' \(\mathbf{\Phi}\) of size \(N \times (2K)\) for \(N\) data points and \(2K\) basis functions (sine + cosine pairs). Specifically,

\[
\mathbf{\Phi}_{i,\,2k-1} 
\;=\; \sin\bigl(2\pi f_k\, t_i\bigr),
\qquad
\mathbf{\Phi}_{i,\,2k} 
\;=\; \cos\bigl(2\pi f_k\, t_i\bigr).
\]

Then let \(\boldsymbol{\alpha} = [\,w_1, v_1, \dots, w_K, v_K\,]^\top\). The approximation becomes:

\[
x(t_i) \;\approx\; 
\mathbf{\Phi}_{i,\,:}\;\boldsymbol{\alpha}.
\]

Fitting \(\boldsymbol{\alpha}\) is done via **least squares**:

\[
\widehat{\boldsymbol{\alpha}}
\;=\;
\arg\min_{\boldsymbol{\alpha}} 
\bigl\|\mathbf{x} \;-\; \mathbf{\Phi}\,\boldsymbol{\alpha}\bigr\|^2,
\]
where \(\mathbf{x} = [\,x(t_1), \ldots, x(t_N)\,]^\top\).

### 2.3 The Fast Fourier Transform (FFT)

In practice, when we use *standard* equally spaced time samples and a standard set of frequencies, we can efficiently compute the **Fourier transform** via the **Fast Fourier Transform (FFT)** in \(\mathcal{O}(N \log N)\) time. Historically, the FFT revolutionized many fields by making it computationally feasible to:

- Decompose signals into frequency components.
- Solve partial differential equations (PDEs) using spectral methods.
- Apply convolution-based operations quickly.

Even though our *linear regression* perspective is mathematically correct and instructive, the FFT is **much faster** than a naive \(\mathcal{O}(N \times K)\) regression when you use the *canonical* set of frequencies and uniform sampling.

### 2.4 Beyond Standard Fourier: Advanced Modal Techniques

Modern applied math and machine learning exploit **modal decomposition** ideas in many advanced methods. For example:

- **POD (Proper Orthogonal Decomposition)**: Finds orthonormal modes that best capture the variance of data (often used in fluid dynamics).  
- **DMD (Dynamic Mode Decomposition)**: A data-driven method for extracting spatial-temporal coherent structures and their linear evolution.  
- **DeepONets (Deep Operator Networks)**: Learn operators (mappings from function spaces to function spaces), can be viewed as generalizing classical expansions into neural frameworks.  
- **FNO (Fourier Neural Operator)**: A neural network approach that applies spectral (Fourier) transformations in intermediate layers to model PDE solutions.

These methods **extend** the notion of “expanding into bases,” whether in a data-driven manner (like POD/DMD) or by mixing neural networks with Fourier-like transforms (like FNO). They rely on the same principle of constructing “modal expansions” to capture complex systems, but leverage **deep learning** to handle nonlinearities and high-dimensional problems more flexibly.

**Key Takeaways**:
1. **Fourier expansions** give us a direct handle on *oscillatory signals* through sines and cosines.  
2. **Implementing a Fourier series** can be done via **linear regression** on sine/cosine features—this is conceptually straightforward, though the **FFT** is typically more efficient.  
3. **Modal decompositions** generalize Fourier expansions to **other orthonormal bases** or data-driven modes.  
4. **Advanced methods** (POD, DMD, DeepONets, FNO, etc.) show that *modal thinking* still powers cutting-edge research in dynamical systems and operator learning.


## 3. Auto-Regressive (AR) Models

### 3.1 Motivation: Shortcomings of Direct Function Fitting

In the **previous section**, we saw how one can fit \(x(t)\) with sines, cosines, polynomials, or other basis functions. However, **these approaches rely on us choosing the correct features**—for instance, if the signal is not strictly periodic (or if its frequency drifts over time), a fixed set of sines/cosines will typically **extrapolate poorly**.

Furthermore, even when the data exhibits some clear periodicity, **explicitly picking** those functions (e.g., \(\sin(\omega t)\)) is a strong assumption. If the system has additional complexity (trends, noise, changing frequencies), the model can fail to generalize.

### 3.2 The Autoregressive Perspective

Rather than modeling \(x_i\) as a function of the continuous variable \(t\), an **autoregressive (AR)** approach models \(x_i\) directly as a function of its **previous** values:

\[
x_i \;=\; f\!\bigl(x_{i-1},\, x_{i-2},\, \dots,\, x_{i-p}\bigr).
\]

This way, we **learn** how each new value depends on the **past** rather than on \emph{time} itself. A simple yet powerful case is the **AR(\(p\))** model **with linear dependence**:

\[
x_i 
\;=\; \sum_{\ell=1}^p w_\ell \, x_{i-\ell} 
\;+\; w_0 
\;+\; \varepsilon_i,
\]
where \(\varepsilon_i\) is noise. For \(p=2\) (AR(2)), the model reads:
\[
x_i 
\;=\; w_2\,x_{i-2} 
\;+\; w_1\,x_{i-1} 
\;+\; w_0 
\;+\; \varepsilon_i.
\]

One can fit \(\{w_2,\,w_1,\,w_0\}\) by **linear regression** on the lagged data \(\{(x_{i-2}, x_{i-1})\}\).

### 3.3 Example: AR(2) Can Approximate a Sine Wave

To illustrate the flexibility of AR(2) models, consider the time series \(\,x_i = \sin\bigl(\omega\, i\bigr)\). We can generate data for \(i = 0, 1, \dots, N-1\), then attempt to fit an AR(2):

\[
x_i 
\;=\; w_2\,x_{i-2} + w_1\,x_{i-1} + w_0.
\]
*(For simplicity, assume no noise \(\varepsilon_i\).)*

Below is a short **Python** snippet demonstrating how to **simulate** the data and **fit** the AR(2) parameters using standard linear regression:

```python
import numpy as np
import matplotlib.pyplot as plt

# 1) Generate a sine wave
N = 200
omega = 0.2  # e.g., 0.2 radians per step
i_vals = np.arange(N)
x_vals = np.sin(omega * i_vals)

# 2) Create lagged data for AR(2)
# We'll form a design matrix: [ [x_{i-1}, x_{i-2}, 1], ...]
X = []
y = []
for i in range(2, N):
    X.append([x_vals[i-1], x_vals[i-2], 1.0])  # lags + constant
    y.append(x_vals[i])

X = np.array(X)  # shape (N-2, 3)
y = np.array(y)  # shape (N-2,)

# 3) Fit w2, w1, w0 via least squares: w* = (X^T X)^{-1} X^T y
w_hat = np.linalg.pinv(X) @ y
w2, w1, w0 = w_hat

print("Fitted AR(2) parameters:")
print("w2 =", w2, ", w1 =", w1, ", w0 =", w0)

# 4) Compare predicted vs. original
# We can do a 1-step-ahead simulation
x_pred = x_vals.copy()
for i in range(2, N):
    x_pred[i] = w2*x_pred[i-2] + w1*x_pred[i-1] + w0

plt.plot(i_vals, x_vals, label="True Sine Wave")
plt.plot(i_vals, x_pred, '--', label="AR(2) Prediction", alpha=0.8)
plt.legend()
plt.show()
```

When you run this code, you will typically see that **AR(2)** *tracks* the sinusoidal behavior quite closely (especially for small noise and a stable frequency \(\omega\)).

#### Analytical Derivation

We can see why **AR(2)** can represent a sine wave by recalling a **trigonometric identity**. For \(x_i = \sin(\omega\, i)\), we have
\[
\sin(\omega\, i)
\;=\;
2\cos(\omega)\,\sin(\omega\, (i-1))
\;-\;
\sin(\omega\, (i-2)).
\]
Hence, in the form \(x_i = w_2\,x_{i-2} + w_1\,x_{i-1}\), we would identify
\[
w_1 \;=\; 2\,\cos(\omega),
\quad
w_2 \;=\; -\,1,
\quad
w_0 = 0.
\]
Thus, the sinusoidal behavior is **encoded** by the specific values of \((w_1, w_2)\). 

**Challenge**: From real (or simulated) data, can you **estimate** \((w_1, w_2)\) and then **infer** \(\omega\)? Notice that:
\[
w_1 = 2\,\cos(\omega)
\quad\Longrightarrow\quad
\omega = \arccos\!\bigl(w_1/2\bigr).
\]


### Key Takeaways

1. **Motivation**: Directly fitting \(x(t)\) with chosen features may generalize poorly in time if the chosen basis is inaccurate.  
2. **Autoregressive Approach**: AR models shift perspective to predicting \(x_i\) from **previous** samples \((x_{i-1}, x_{i-2}, \dots)\).  
3. **AR(2) and Sine Waves**: Even a simple AR(2) can capture purely oscillatory signals like \(\sin(\omega\, i)\). The parameters relate neatly to the frequency.  
4. **Open Question**: If you have real or noisy data that looks oscillatory, can you estimate \(\omega\) from the fitted AR(2) parameters? This is a neat link between **autoregression** and **spectral** (frequency) analysis.


In the **next sections**, we will see how to extend these ideas to handle *multi-step predictions*, exogenous inputs (ARX, ARMAX), and eventually **nonlinear** or **deep learning** models that go beyond the simple linear form.

## 4. Deep Learning Autoregression

### 4.1 Neural Networks for Sequence Data

We already saw that a neural network can replace the linear function in:
\[
x_{i} 
\;=\; f_{\mathbf{w}}(x_{i-1},\, x_{i-2},\, \dots),
\]
leading to **Nonlinear AutoRegressive** models. However, when the sequence is long (potentially thousands of steps), passing all past values \(\{x_{i-1}, \dots, x_{i-p}\}\) into a dense network becomes:

1. **Parametrically large** (since each input dimension is a time step),  
2. Potentially **inefficient**, and  
3. Lacking **structured weight sharing** across time.

**Recurrent Neural Networks (RNNs)** (including LSTMs, GRUs) solve some of these issues by maintaining a hidden state over time. Yet RNNs can be **sequentially hard to parallelize** because each new hidden state depends on the previous one.

### 4.2 Time Convolutional Networks (TCNs)

A **Time Convolutional Network (TCN)** provides an alternative that:

- Uses **1D convolutions** over the time dimension,  
- Maintains a **causal structure**, so the prediction at time \(i\) depends only on times \(\le i\),  
- Often employs **dilated** convolutions to achieve a large “receptive field” without incurring huge filter sizes,  
- Can be **parallelized** across time steps because convolutions can be computed in parallel.

#### 4.2.1 Basic Architecture

A TCN is built from **convolutional blocks**, each typically containing:

1. A **1D convolution** with a certain **kernel size** \(k\).  
2. **Dilation**, which “skips” certain time points to expand the receptive field exponentially with the number of layers.  
3. **Nonlinearity** (e.g., ReLU).  
4. **Residual connections**, allowing gradients to flow more easily.  
5. **Dropout** or other regularization strategies.


To ensure the model doesn’t “peek” into the future, TCN layers are **causal**: the convolution filter at time \(t\) covers only \(\{t, t-1, \dots\}\) (no future indices). **Dilated** convolution effectively uses a “stride” in the filter, so if the dilation is \(d\), the filter taps input points \(\{t, t-d, t-2d, \dots\}\). By increasing \(d\) in deeper layers, TCNs can cover a large temporal context with relatively short filter kernels.

### 4.3 Mathematical Detail

Let \(\mathbf{x} \in \mathbb{R}^{T}\) be the **time series** (for simplicity, scalar). A single **dilated convolution** layer with **kernel size** \(k\) and **dilation** \(d\) computes:

\[
y_t 
\;=\;
\sum_{j=0}^{k-1}
w_j \; x_{t - j,d}
\;+\; b,
\]
for each valid \(t\). In a **causal** TCN, we only define \(y_t\) for \(t \ge (k-1)\,d\) to avoid referencing future samples (we might pad earlier values with zeros or replicate boundary conditions).

#### Multiple Filters / Channels

In practice, we have **multiple filters** (channels), so \(\mathbf{x}_t\) might be mapped to an **output tensor** \(\mathbf{y}_t\). Summation is over channels in \(\mathbf{x}\). Stacking multiple TCN layers deepens the transformation:

\[
\mathbf{y}_{t}^{(\ell)}
\;=\;
\mathrm{ReLU}\Bigl(
\mathrm{Conv1D}\bigl(\mathbf{y}_{t}^{(\ell-1)}\bigr)
\Bigr),
\]
with appropriate **dilation** and **residual** connections. Summaries of these outputs can be fed to a **final layer** for prediction or classification.


### 4.4 Applications to Time-Series Forecasting

TCNs can handle:

1. **Single-step predictions**: Predict \(x_{t+1}\) given \(\{x_1, \dots, x_t\}\).  
2. **Multi-step predictions**: Predict \(\{x_{t+1}, \dots, x_{t+k}\}\) using the entire history \(\{x_1, \dots, x_t\}\).  
3. **Multivariate** or **multi-channel** time series: Each channel is a different sensor or variable.

Because of the convolutional structure: 1) Each layer’s convolution can be done efficiently on GPUs/TPUs across all \(t\). This often trains faster than purely recurrent models (which process one step at a time), 2) Dilation allows the TCN to incorporate information from far in the past (e.g., thousands of steps) if we stack enough layers with exponentially growing dilation factors, 3)TCNs can be combined with attention mechanisms or augmented with exogenous inputs (extra channels) for advanced forecasting tasks.

Example **domains** include:

- **Financial time series**: Multi-step forecasting of stock prices or volatility.  
- **Energy load forecasting**: Handling daily/weekly seasonalities via dilated convolutions.  
- **Speech synthesis / audio generation**: TCN-like architectures (WaveNet) are used for high-quality audio.  
- **Sensor networks**: If you have a large set of sensors measuring temperature/humidity, TCNs can learn spatiotemporal dependencies if you stack 1D time convolutions for each sensor channel (or combine with spatial graphs).

### 4.5 Summary and Outlook

- **TCN** layers use *dilated*, *causal* **1D convolutions** to capture **long-range time dependencies** without sequential recurrences.  
- **Residual connections** and **parallelizable** operations allow TCNs to train efficiently and handle large time-series.  
- **Applications** span speech (WaveNet), forecasting (energy, finance, weather), sensor data analysis, and more.  
- **Hybrid** approaches can merge TCN blocks with attention, graph neural networks (for spatial structure), or recurrent modules.

In the **next sections**, we will look at how TCNs (or other deep models) can be extended to **higher-dimensional** data (e.g., \(\mathbf{x}_i \in \mathbb{R}^d\)) and how **hidden variables** or *state-space* concepts appear in advanced sequence models.

## 5. Higher-Dimensional \(\mathbf{x}\) and State-Space Models

### 5.1 Moving from Scalar to Vector-Valued \(\mathbf{x}_i\)

Thus far, we have implicitly treated \(x_i\) as a **scalar**. However, in many applications (e.g., multivariate time series, control systems, or sensor arrays), each time step \(i\) yields a **vector**:

\[
\mathbf{x}_i \;\in\; \mathbb{R}^d,
\]
possibly with **exogenous inputs** \(\mathbf{u}_i \in \mathbb{R}^r\). The goal is still to predict or model how \(\mathbf{x}_{i+1}\) (or some output \(\mathbf{y}_i\)) evolves over time, but now the dynamics are **matrix-based** rather than scalar.

### 5.2 Linear State-Space Equations

A widely used formalism is the **(discrete-time) state-space model**, which can be written as:

\[
\begin{aligned}
\mathbf{x}_{i+1} &= A\,\mathbf{x}_i \;+\; B\,\mathbf{u}_i \;+\; \boldsymbol{\varepsilon}_i, \\[6pt]
\mathbf{y}_i     &= C\,\mathbf{x}_i \;+\; \mathbf{r}_i,
\end{aligned}
\]
where

- \(\mathbf{x}_i \in \mathbb{R}^d\) is the (hidden) *state* at time \(i\).  
- \(\mathbf{u}_i \in \mathbb{R}^r\) represents *known inputs* or controls.  
- \(\mathbf{y}_i \in \mathbb{R}^m\) is the *observed output* (e.g., sensor measurements).  
- \(A \in \mathbb{R}^{d\times d},\,B \in \mathbb{R}^{d\times r},\,C \in \mathbb{R}^{m\times d}\) are model matrices.  
- \(\boldsymbol{\varepsilon}_i\) and \(\mathbf{r}_i\) are *noise* or *disturbances* in the process and measurements, respectively.

#### 5.2.1 Known vs. Unknown Matrices

In many engineering or physics settings, \(A\), \(B\), and \(C\) might be **partially known** from first principles (e.g., linearized dynamics around an operating point). In other scenarios, these matrices (or some of their parameters) are **unknown** and must be **learned** from data.

##### Parameter Estimation

If \(\mathbf{x}_i\) and \(\mathbf{u}_i\) are fully observed, but \(\mathbf{y}_i\) is not used, the simplest regression problem is:

\[
\min_{A,\,B}
\sum_{i=1}^{T-1}
\left\|
\mathbf{x}_{i+1} 
\;-\;
A\,\mathbf{x}_i 
\;-\;
B\,\mathbf{u}_i
\right\|^2.
\]
This yields a **least-squares** solution if the noise \(\boldsymbol{\varepsilon}_i\) is assumed i.i.d. Gaussian. In matrix form, one can stack \(\mathbf{x}_{i+1}\) and the corresponding \(\mathbf{x}_i, \mathbf{u}_i\) across \(i\) to solve for \(\widehat{A}, \widehat{B}\).

### 5.3 Partial Observations and the Kalman Filter

Often, the state \(\mathbf{x}_i\) is **not** directly measured. Instead, we observe

\[
\mathbf{y}_i 
\;=\; 
C\,\mathbf{x}_i 
\;+\; 
\mathbf{r}_i,
\]
where \(\mathbf{r}_i\) is measurement noise. Estimating (or “filtering”) \(\mathbf{x}_i\) from the **indirect** observations \(\{\mathbf{y}_1,\dots,\mathbf{y}_i\}\) is the task of a **Kalman filter** (in the linear-Gaussian case).

#### Kalman Filter Assumptions

1. **Linear dynamics**: \(\mathbf{x}_{i+1} = A\,\mathbf{x}_i + B\,\mathbf{u}_i + \boldsymbol{\varepsilon}_i\).  
2. **Linear measurement**: \(\mathbf{y}_i = C\,\mathbf{x}_i + \mathbf{r}_i\).  
3. **Gaussian noise**: \(\boldsymbol{\varepsilon}_i \sim \mathcal{N}(\mathbf{0}, Q)\) and \(\mathbf{r}_i \sim \mathcal{N}(\mathbf{0}, R)\).  
4. **Initial state** \(\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)\).

Under these conditions, the **Kalman filter** provides an **optimal** (minimum-variance) estimate \(\widehat{\mathbf{x}}_i\) of the hidden state \(\mathbf{x}_i\) after seeing measurements up to time \(i\). The filter recurses in two steps:
1. **Predict**: \(\widehat{\mathbf{x}}_{i\mid i-1} = A\,\widehat{\mathbf{x}}_{i-1\mid i-1} + B\,\mathbf{u}_{i-1}\).  
2. **Update**: Incorporate \(\mathbf{y}_i\) to correct the prediction.

Additionally, the **covariance** of the estimation error is updated, allowing uncertainty quantification. For **nonlinear** versions, the **Extended Kalman Filter** or **Unscented Kalman Filter** are used.

### 5.4 Relating to Previous Sections

#### 5.4.1 AR(\(p\)) vs. State-Space

- An **AR(\(p\))** model with \(\mathbf{x}_i \in \mathbb{R}^d\) can sometimes be *rewritten* in state-space form, but **state-space** emphasizes the concept of an internal **state** that evolves, possibly with input \(\mathbf{u}_i\).  
- The dimension \(d\) can represent *fundamental states* of the system (e.g., position, velocity in 2D), rather than simply stacking past outputs.

#### 5.4.2 Noise and Uncertainty

- **State-space** explicitly includes process noise \(\boldsymbol{\varepsilon}_i\) and measurement noise \(\mathbf{r}_i\).  
- This is natural for **probabilistic** or **Bayesian** interpretations, where we keep track of the distribution of \(\mathbf{x}_i\).

### 5.5 Outlook: Nonlinear and Learned State-Space Models

1. **Nonlinear**: In many real systems, \(A\) and \(B\) could be replaced by **nonlinear** transformations, leading to \(\mathbf{x}_{i+1} = f(\mathbf{x}_i, \mathbf{u}_i)\). Extended Kalman filtering or fully **learned** models (e.g., RNN-based, “Neural ODE,” etc.) may be employed.  
2. **Partially Known**: Sometimes partial physics is known, but some parameters or nonlinearities are learned from data (a “gray box” approach).  
3. **Deep State-Space**: Recent **deep learning** approaches treat the hidden state \(\mathbf{h}_i\) as an **abstract** representation, bridging ideas from **state-space** and **recurrent** or **convolutional** architectures.

Hence, the linear state-space model is a **cornerstone** of classical control and signal processing. Its generalizations—involving learning unknown parameters or adopting nonlinear transitions—blend naturally with the concepts introduced in our earlier autoregressive sections and upcoming **hidden variable** sections.


**Key Takeaways**:
1. **State-space** models handle **vector** states \(\mathbf{x}_i\) and exogenous inputs \(\mathbf{u}_i\).  
2. When matrices \(A\), \(B\), \(C\) are unknown, we can solve **least-squares** to fit them (assuming direct or partial observations).  
3. **Kalman filters** provide a principled way to estimate hidden states \(\mathbf{x}_i\) given noisy measurements \(\mathbf{y}_i\).  
4. This framework extends naturally to **nonlinear** dynamics and more **data-driven** methods that approximate \(f(\cdot)\) or combine known physics with learned components.


Below is the **Section 6** text with the **same numbering** as our previous sections. The content remains in a **narrative** style, but now includes **sub-section numbering** to be consistent with the earlier format.


## 6. Hidden Variables

### 6.1 Motivation: Memory Without Large Lag

Imagine you have a time series that depends on events stretching far into the past—perhaps there are seasonal effects or slow drifts over time. If you attempt to capture these dependencies by extending an **autoregressive** model to include many past lags \(\{x_{i-1}, x_{i-2}, \ldots, x_{i-p}\}\), you can quickly face an explosion in the number of parameters. Not only may you lack the data to reliably estimate so many parameters, but you also lose the simplicity of having a concise “state” that summarizes the system’s behavior at each step.

A more powerful approach is to posit that, at each time \(i\), the system maintains an **internal** or **hidden** representation \(\mathbf{h}_i\). This representation, in principle, can encode all relevant information from the past. Instead of enumerating a long list of previous observations, the model simply updates \(\mathbf{h}_i\) to \(\mathbf{h}_{i+1}\) based on the new observation \(\mathbf{x}_i\).

### 6.2 General Hidden-State Formulation

Formally, a hidden-variable model often looks like this:

\[
\mathbf{h}_{i+1} 
\;=\; 
g\!\bigl(\mathbf{h}_i,\;\mathbf{x}_i;\;\boldsymbol{\theta}\bigr),
\quad
\mathbf{x}_{i} 
\;=\; 
\ell\!\bigl(\mathbf{h}_i;\;\boldsymbol{\theta}\bigr).
\]

The function \(g(\cdot)\) defines how the hidden state evolves over time, while the function \(\ell(\cdot)\) maps the hidden state to the observed output \(\mathbf{x}_i\). The parameters \(\boldsymbol{\theta}\) govern both the state transition and the emission from hidden state to observed variables. By maintaining \(\mathbf{h}_i\) as a **memory** that encapsulates past information, the model frees itself from the need to keep all prior time steps explicitly.

### 6.3 Example: A Vanilla RNN

A famous instance of such a hidden-variable model in deep learning is the **vanilla Recurrent Neural Network (RNN)**. If \(\mathbf{x}_i\) represents the observation at time \(i\) (say, a word in a sentence or a scalar measurement in a time series), and \(\mathbf{h}_i\) is the network’s hidden state, the update might look like:

\[
\mathbf{h}_{i+1} 
\;=\; 
\tanh\Bigl(
W_{hh}\,\mathbf{h}_i 
\;+\; 
W_{hx}\,\mathbf{x}_i 
\;+\; 
\mathbf{b}
\Bigr),
\quad
\mathbf{x}_{i+1}
\;=\;
W_{xh}\,\mathbf{h}_{i+1}.
\]

Here, \(\tanh\) provides the nonlinearity that lets the model learn intricate temporal dependencies. The hidden state \(\mathbf{h}_i\) can, at least in theory, carry forward information from **all** previous time steps. This avoids the blow-up in dimension you would get if you tried to store many explicit lags.

An everyday example is **language modeling**: if \(\mathbf{x}_i\) is the vector embedding of the \(i\)-th word in a sentence, the hidden state \(\mathbf{h}_i\) might accumulate context about the sentence’s subject, verb tense, or other cues that help predict the next word. Instead of referencing every single past word individually, the network uses the hidden state as a **summary**.

### 6.4 Parallels to State-Space Models

Hidden variables also connect naturally to **state-space** ideas in control and signal processing. In a classical (linear) state-space, we might write

\[
\mathbf{x}_{i+1} 
\;=\; 
A\,\mathbf{x}_i 
\;+\; 
B\,\mathbf{u}_i 
\;+\; 
\boldsymbol{\varepsilon}_i,
\quad
\mathbf{y}_i 
\;=\; 
C\, \mathbf{x}_i 
\;+\; 
\mathbf{r}_i,
\]

where \(\mathbf{x}_i\) is the “true” hidden state, \(\mathbf{y}_i\) the observed measurement, and \(\mathbf{u}_i\) an external input. The main difference is that the RNN’s update function \(g(\cdot)\) is learned from data rather than derived from first-principles physics, and it need not be linear. Nonetheless, the conceptual link is the same: a **latent state** evolves over time and helps explain or generate the observable outputs.

### 6.5 Advantages and Challenges

Placing memory into a hidden variable \(\mathbf{h}_i\) offers a major advantage: we can keep \(\mathbf{h}_i\) at a tractable dimension, no matter how long the time series. This approach also opens up many possible architectures (such as GRUs and LSTMs) that address the well-known issue of **vanishing or exploding gradients** in long-sequence training. 

However, there are challenges too. Simple RNNs can still struggle with very extended time dependencies, leading to techniques like gating (in LSTMs/GRUs) or attention-based mechanisms (as in Transformers). Another challenge is **interpretability**: unlike a linear state-space where \(\mathbf{x}_i\) might correspond to physically meaningful quantities like position or velocity, a learned hidden state can be more opaque.

### 6.6 Hidden Variables Beyond RNNs

Although RNNs are a prominent deep-learning example, hidden variables are fundamental in many sequential models:

- **Hidden Markov Models (HMMs)** rely on a discrete hidden state \(h_i\) that transitions with certain probabilities \(p(h_{i+1}\mid h_i)\) and produces observations \(x_i\) with probabilities \(p(x_i\mid h_i)\).  
- **Nonlinear Kalman Filters** or **Extended/Unscented Kalman Filters** can handle approximate Bayesian updates of a continuous-valued hidden state when the system is only partially observed.  
- **Deep Markov Models** combine the HMM perspective with neural networks to parameterize transitions and emissions more flexibly.

All of these share the same conceptual thread: that a **latent state** can simplify modeling by capturing everything from the past that matters for the future, allowing the model or filtering algorithm to concentrate on the most relevant historical information.

### 6.7 Conclusion

Hidden-variable models address the fundamental question of **how** to represent memory in a time-series process without enumerating a large set of past values directly. Whether you opt for a neural approach like an RNN, a classical approach like a linear state-space with a Kalman filter, or a probabilistic approach like an HMM, the logic is the same: introduce a hidden state \(\mathbf{h}_i\) that evolves in time. This perspective can be more compact, often easier to train (in the sense of capturing long-range patterns), and potentially more interpretable if the hidden variables are chosen or discovered in a way that maps to real-world phenomena.


## 7. Probabilistic Models

### 7.1 When Deterministic Predictions Aren’t Enough

So far, many of our discussions—be it autoregressive methods or neural networks—have been framed in terms of a **deterministic** function that maps past observations (or hidden states) to the next value. Yet real-world processes are often noisy or uncertain. In many scenarios, you want not only a **point prediction** but also a sense of how likely different outcomes are. For instance, if you’re modeling the spread of a disease, you might care about the distribution of possible infection trajectories more than a single, best-guess curve.

**Probabilistic models** speak directly to this desire by positing that each observation \(x_i\) arises from *random variables* governed by certain probability distributions. Rather than saying “\(x_i = f(x_{i-1}, \dots)\),” we say “\(x_i\) is drawn from a **conditional** distribution** \(p(x_i \mid x_{i-1}, \dots)\).” This distribution can have parameters that depend on past states, or in more complex setups, on a hidden state.

### 7.2 Markov Chains and Hidden Markov Models

A simple illustration is the **Markov chain**, where the next state depends only on the current one:

\[
x_{i+1} 
\;\sim\; 
p\bigl(x_{i+1} \mid x_i\bigr).
\]

If these states are directly observed, this is a basic (or “visible”) Markov chain. But more commonly in real applications, the Markov property is buried inside some **hidden** variables. That is, you might observe \(y_i\) at each step, while the true Markovian transitions happen in an unobserved state \(h_i\). This perspective leads to the **Hidden Markov Model (HMM)**, where the process is:

\[
h_{i+1}
\;\sim\;
p\bigl(h_{i+1} \mid h_i\bigr),
\quad
x_i
\;\sim\;
p\bigl(x_i \mid h_i\bigr).
\]

You can think of each \(h_i\) as the “mode” or “regime” of the system at step \(i\). For example, in **speech recognition**, each \(h_i\) might represent a phoneme being uttered, and \(x_i\) the acoustic signal. Because the hidden states are rarely known ahead of time, one typically uses specialized inference algorithms—like **Forward-Backward** or **Viterbi**—to estimate which hidden states are likely given the entire observed sequence. The **Expectation-Maximization (EM)** algorithm is then used to learn the transition probabilities \(p(h_{i+1}\mid h_i)\) and emission distributions \(p(x_i\mid h_i)\).

**Where does the noise come in?** In an HMM, the randomness in \(p(h_{i+1}\mid h_i)\) accounts for the uncertain transitions (maybe you stay in the same regime with probability 0.9, or jump to a new one with probability 0.1), while \(p(x_i\mid h_i)\) handles measurement or emission variability.


### 7.3 Beyond Discrete States: Continuous and Nonlinear Models

Hidden Markov Models often assume a finite set of hidden states. Yet you can also formulate **probabilistic** models with **continuous** hidden states—an example being the **Kalman filter** where states and measurements follow Gaussian distributions. That can be considered a linear-Gaussian variant of a **state-space** model. More modern incarnations might let the transition and emission distributions be parameterized by **neural networks**, creating a class of **deep generative models** for time series. Terms like **Deep Markov Models**, **Variational Autoencoders** for sequences, or **Neural ODEs** with uncertainty all inhabit this space.


### 7.4 Why Probabilistic?

One chief advantage of specifying the process in terms of probability distributions is that it becomes natural to perform **inference** and **uncertainty quantification**. Instead of just predicting a single next value, you can estimate a **distribution** over possible next values, along with confidence intervals or credible intervals. In control problems, it can be vital to know not only your best estimate of the system’s state but also how uncertain you are—particularly if you have to plan actions that might fail if your estimate is off.

A **probabilistic** approach also unifies many tasks within one conceptual framework. For example, you can:

1. **Generate samples** from the model, simulating plausible futures.  
2. **Condition** on partial observations to perform smoothing or filtering.  
3. **Compute likelihoods** to compare different model hypotheses.  

This broadens the scope from mere prediction into a more general **generative** perspective on sequence data.


### 7.5 Comparisons with RNNs and TCNs

In practice, **recurrent neural networks** (RNNs) or **time convolutional networks** (TCNs) can be cast in probabilistic terms if you equip them with output distributions. For instance, an RNN that predicts a mean and variance for a Gaussian distribution at each step is basically saying:

\[
x_i
\;\sim\;
\mathcal{N}\Bigl(\mu_{\theta}(x_{i-1}, \ldots),\;\sigma_{\theta}^2(x_{i-1}, \ldots)\Bigr).
\]

Hence, the line between “probabilistic” and “deterministic with noise” can sometimes blur: you can wrap a neural net inside a **probabilistic** output layer to create a model that yields both point estimates and uncertainty. Alternatively, you might fully encode a **latent** (hidden) state with an RNN, then define a distribution over the observations given that latent state (as in **Variational Autoencoders for time-series**).


### 7.6 Conclusion and Outlook

Probabilistic models remind us that **uncertainty** and **stochasticity** are inherent in many real systems—particularly when data is noisy, dynamics are partially observed, or processes have multiple regimes. Whether in the form of HMMs for discrete states, linear-Gaussian filters like the Kalman model, or advanced neural-based generative models, these methods provide a robust framework to **not only** predict or classify but also **understand** the random variability in sequential data.

In practical applications, combining **probabilistic** viewpoints with the **deep** or **autoregressive** techniques discussed earlier can yield powerful hybrid models—ones that excel at capturing complex dynamics while also quantifying how certain (or uncertain) they are about their predictions. Such synergy is increasingly essential in safety-critical or high-stakes domains (healthcare, finance, climate modeling, etc.) where purely deterministic approaches may understate the risks and variability inherent in the real world.

NOTE: These lecture notes were based on two lectures of MECH 798K at AUB, and were partially edited using GPT-o1.