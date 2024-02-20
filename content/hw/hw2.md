---
layout: page
title: "Problem Set #2 - Differential Equations And Probabilistic Models"
permalink: /content/hw/hw2/
---

Submit your coding problems in separate folders. If you use notebooks, provide a brief explanation for each question using the markdown functionality. I encourage you to discuss the problems, but when you write down the solution please do it on your own. You can raise and answer each other’s questions on Moodle; but please don’t provide exact solutions. Submit Jupyter notebooks (or scripts) for all problems.

## Problem 0: The Logistic Map (10 points)

The [logistic map](https://en.wikipedia.org/wiki/Logistic_map) is a simple mathematical model of population growth. It is defined by the following difference equation:

$$x_{n+1} = r x_n (1 - x_n)$$

where $$x_n$$ is the population at time $$n$$ and $$r$$ is a parameter that controls the growth rate.

(a) Generate a time series of the logistic map for $$r = 2.5$$ and $$x_0 = 0.5$$ for $$n = 0, 1, \ldots, 100$$. And plot the time series.

(b) Use linear regression to find $$r$$.

(c) Bonus: Use [PySR](https://github.com/MilesCranmer/PySR)'s symbolic regression to find the logistic map equation.


## Problem 1: The SIR Model (10 points)

When COVID-19 hit, the [SIR (Susceptible, Infectious, or Recovered)](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) model was used to model infectious diseases. It is defined by the following system of differential equations:

$$\frac{dS}{dt} = -\beta S I$$

$$\frac{dI}{dt} = \beta S I - \gamma I$$

$$\frac{dR}{dt} = \gamma I$$

where $$S$$ is the number of susceptible individuals, $$I$$ is the number of infected individuals, and $$R$$ is the number of recovered individuals. The parameters $$\beta$$ and $$\gamma$$ control the transmission rate of the desease and the recovery rate of individuals, respectively.

(a) Solve the SIR model for $$\beta = 0.3$$, $$\gamma = 0.1$$, $$S(t=0) = 0.99$$, $$I(t=0) = 0.01$$, and $$R(t=0) = 0$$. And plot the time series (you should decide what's a good time range to plot).

(b) Define a library of functions using polynomials in $$S$$ and $$I$$ up to quadratic order. 

(c) Use Lasso to find $$\beta$$ and $$\gamma$$. Tune the regularization term $$\alpha$$ (by hand) until you find a good fit.

(d) Compare the solution of the discovered model with the solution of the original model.

## Problem 3: Brownian Motion and the Diffusion Equation (30 points)

In 1827, botanist Robert Brown observed that pollen grains suspended in water moved in a random zig-zag motion. This phenomenon later came to be known as [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion). At the time, the atomistic nature of matter was not yet widely accepted, and Brownian motion provided some of the first evidence that matter was composed of atoms. 

In 1905, Albert Einstein published a paper that provided a quantitative explanation of Brownian motion, and its relationship with the [diffusion equation](https://en.wikipedia.org/wiki/Diffusion_equation); thus, providing strong theoretical evidence for the existence of discrete atoms. 

Later, Norbert Wiener and Paul Lévy developed the mathematical theory of Brownian motion, which laid the foundation for the field of stochastic calculus, and had a profound impact on fields as diverse as finance, physics, and biology. The equation for a Brownian particle is given by:

$$dX = v dt + \gamma dW$$

where $$X$$ is the position of the particle, $$v$$ is the drift velocity (do to bulk motion; e.g. water particle drifting with a river), $$\gamma$$ is related to the diffusion coefficient, $$dt$$ is the time step, and $$dW$$ is a Wiener process (a mathematical object that represents a continuous-time random walk). The Wiener process has the property that $$dW \sim \mathcal{N}(0, dt)$$; in other words, it is a normally distributed random variable with mean 0 and variance $$dt$$. To generate a Wiener process, you can use the normal distribution to generate a random number at each time step, and then sum the random numbers to get the time series Wiener process.

We want to show, like Einstein did in the early 20th century, that the diffusion equation models the statistical evolution of the random particles. But instead of doing it analytically, we will use a Monte Carlo simulation.

(a) Generate a time series of the Brownian motion for $$v = 0.1$$, $$\gamma = 0.1$$, and $$X(t=0) = 0$$ for 1000 particles. Plot the time series for the first 10 particles. Based on the plot decide whether you need a longer time series.

(b) Use the time series data to estimate the probability density function (PDF) of the positions of the particles at every time step using Kernel Density Estimation (KDE). Plot the PDFs at 5 different time steps on the same plot.

(c) Use the KDE data to find the advection and diffusion coefficients in the Advection-Diffusion equation:

$$\frac{\partial p}{\partial t} = - a \frac{\partial p}{\partial x} + b \frac{\partial^2 p}{\partial x^2} $$

where $$p$$ is the probability density function of the position of the particles.

(d) Discretize the equation you obtained and solve it numerically using the finite difference method. Compare the solution with the PDF data you obtained in part (b).

(e) What is the relationship between $$v$$, $$\gamma$$, $$a$$, and $$b$$? 

## Problem 4: Open Problem

Go through the list of differential equations in different fields on this [Wikipedia page](https://en.wikipedia.org/wiki/List_of_named_differential_equations), and pick one that you find interesting. 

(a) Write a brief explanation of the equation, its applications, and how you can collect data for the variable it is solving for. Once you have the data, what would be a good scenario where you would be interested in estimating its coefficients from data? 

(b) Solve the equation using a numerical method of your choice; and compare the solution with the analytical solution if it exists. 

(c) Define the library of functions you should use to discover the equation from data. Is there a larger set of functions that you can define in the absence of knowledge of the equation? (e.g. polynomials, trigonometric functions, etc. In the Lorenz System example, we used general polynomials of order 2 that contained actual terms.) 

(d) Use the generated data to estimate the coefficients of the equation using the [PySINDy library](https://pysindy.readthedocs.io/en/latest/index.html).

