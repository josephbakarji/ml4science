---
layout: page
title: "Problem Set #0 - Linear Algebra and Calculus Recap"
permalink: /content/hw/hw0/
---

These questions are inspired by those given in the introduction to machine learning course at Stanford (CS229). They require some thinking but brief answers. The purpose of this homework is to brush up on your linear algebra and multivariate calculus. Some of them may be useful for subsequent problem sets. If you have questions, I encourage you to post them on the forum. This homework is not graded, but it doesn't mean that you should do it.

## Problem 1: Gradients and Hessians

A matrix $ A \in \mathbb{R}^{n \times n} $ is symmetric if $ A^T = A $, that is $ A_{ij} = A_{ji} $ for all $ i, j $. Recall the gradient $ \nabla f(x) $ of a function $ f : \mathbb{R}^n \rightarrow \mathbb{R} $ which is the n-vector of partial derivatives:

$$ \nabla f(x) = \begin{bmatrix} \frac{\partial}{\partial x_1} f(x) \\ \vdots \\ \frac{\partial}{\partial x_n} f(x) \end{bmatrix} $$

where

$$ x = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} $$

The Hessian $ \nabla^2 f(x) $ of a function $ f : \mathbb{R}^n \rightarrow \mathbb{R} $ is the $ n \times n $ symmetric matrix of twice partial derivatives:

$$ \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2}{\partial x_1^2} f(x) & \cdots & \frac{\partial^2}{\partial x_1 \partial x_n} f(x) \\ \vdots & \ddots & \vdots \\ \frac{\partial^2}{\partial x_n \partial x_1} f(x) & \cdots & \frac{\partial^2}{\partial x_n^2} f(x) \end{bmatrix} $$

(a) Let $ f(x) = \frac{1}{2} x^T Ax + b^T x $ where $ A $ is a symmetric matrix and $ b \in \mathbb{R}^n $ is a vector. What is $ \nabla f(x) $? Hint: write down the element-wise multiplication and deduce the expression from the resulting matrix. 

(b) Let $ f(x) = g(h(x)) $ where $ g : \mathbb{R} \rightarrow \mathbb{R} $ is differentiable and $ h : \mathbb{R}^n \rightarrow \mathbb{R} $ is differentiable. What is $ \nabla f(x) $?

(c) What is $ \nabla^2 f(x) $ for the $ f(x) $ from part (a)?

(d) Let $ f(x) = g(a^T x) $ where $ g : \mathbb{R} \rightarrow \mathbb{R} $ is continuously differentiable and $ a \in \mathbb{R}^n $ is a vector. What are $ \nabla f(x) $ and $ \nabla^2 f(x) $? (Hint: your expression for $ \nabla^2 f(x) $ may have as few as 11 symbols including $ \nabla $ and parentheses.)

<br>

## Problem 2: Positive Definite Matrices

A matrix $ A \in \mathbb{R}^{n \times n} $ is positive semi-definite (PSD), denoted $ A \succeq 0 $, if $ A = A^T $ and $ x^T Ax \geq 0 $ for all $ x \in \mathbb{R}^n $. A matrix $ A $ is positive definite, denoted $ A \succ 0 $, if $ A = A^T $ and $ x^T Ax > 0 $ for all non-zero $ x \in \mathbb{R}^n $.The simplest example of a positive definite matrix is the identity $ I $ (the diagonal matrix with 1s on the diagonal and 0s elsewhere), which satisfies $ x^T Ix = \|x\|^2 = \sum_{i=1}^n x_i^2 $.

(a) Let $ z \in \mathbb{R}^n $ be an n-vector. Show that $ A = zz^T $ is positive semidefinite.

(b) Let $ z \in \mathbb{R}^n $ be a non-zero n-vector. Let $ A = zz^T $. What is the null-space of $ A $? What is the rank of $ A $?

(c) Let $ A \in \mathbb{R}^{n \times n} $ be positive semidefinite and $ B \in \mathbb{R}^{m \times n} $ be arbitrary, where $ m, n \in \mathbb{N} $. Is $ BAB^T $ PSD? If so, prove it. If not, give a counterexample with explicit $ A, B $.

<br>

## Problem 3: Eigenvectors, Eigenvalues, and the Spectral Theorem

The eigenvalues of an $ n \times n $ matrix $ A \in \mathbb{R}^{n \times n} $ are the roots of the characteristic polynomial $ p_A(\lambda) = \det(\lambda I - A) $, which may (in general) be complex. They are also defined as the values $ \lambda \in \mathbb{C} $ for which there exists a vector $ x \in \mathbb{C}^n $ such that $ Ax = \lambda x $. We call such a pair $ (x, \lambda) $ an eigenvector-eigenvalue pair. In this question, we use the notation $ \text{diag}(\lambda_1, ..., \lambda_n) $ to denote the diagonal matrix with diagonal entries $ \lambda_1, ..., \lambda_n $.


(a) Suppose that the matrix $ A \in \mathbb{R}^{n \times n} $ is diagonalizable, that is $ A = T \Lambda T^{-1} $ for an invertible matrix $ T \in \mathbb{R}^{n \times n} $ where $ \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n) $ is diagonal. Use the notation $ t^{(i)} $ for the columns of $ T $ so that $ T = [t^{(1)} \cdots t^{(n)}] $ where $ t^{(i)} \in \mathbb{R}^n $. Show that $ A t^{(i)} = \lambda_i t^{(i)} $ so that the eigenvalues/eigenvector pairs of $ A $ are $ (t^{(i)}, \lambda_i) $.

**Note:** A matrix $ U \in \mathbb{R}^{n \times n} $ is orthogonal if $ U^T U = I $. The spectral theorem, a crucial theorem in linear algebra, states that if $ A \in \mathbb{R}^{n \times n} $ is symmetric ($ A = A^T $), then $ A $ is diagonalizable by a real orthogonal matrix. In other words, there exists a diagonal matrix $ \Lambda \in \mathbb{R}^{n \times n} $ and an orthogonal matrix $ U \in \mathbb{R}^{n \times n} $ such that $ U^T A U = \Lambda $, or equivalently, 

$$ A = U \Lambda U^T $$

Let $ \lambda_i = \lambda_i(A) $ denote the $ i $th eigenvalue of $ A $.


(b) Let $ A $ be symmetric. Show that if $ U = [u^{(1)} \cdots  u^{(n)}] $ is orthogonal where $ u^{(i)} \in \mathbb{R}^n $ and $ A = U \Lambda U^T $ then $ u^{(i)} $ is an eigenvector of $ A $ and $ A u^{(i)} = \lambda_i u^{(i)} $ where $ \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n) $.

(c) Show that if $ A $ is PSD then $ \lambda_i(A) \geq 0 $ for each $ i $.


## Problem 4: Explore Machine Learning in Science and Engineering

How is machine learning being used in science and engineering? What are the challenges and opportunities? Do some research and write a short report (maximum of 2 pages). If you want to use an LLM (such as ChatGPT) to help you research and write the report, please include the conversation with the LLM as an appendix. Include the following in your report:

- A list of 10 applications of machine learning in science and engineering with references.
- Deduce what are the 3 most important questions in the field of machine learning for science and engineering.
- Conclude with what you believe is a good research question that you'd like to tackle in this course. 