---
layout: page
title: Project
permalink: project/
---

## Overview

The course project is your opportunity to ask a scientific question and answer it using the methods from this course. You will define a problem, collect or generate data, apply machine learning methods informed by scientific knowledge, iterate on your approach, and write up the results as a conference-quality paper.

Projects are done in teams of 2 students. The project constitutes 40% of your course grade.


## Requirements

Your project must combine two elements:

**Scientific modeling.** Choose a system governed by physical, biological, or engineering principles. Identify the inductive biases that constrain the system: conservation laws, symmetries, interaction terms, spatio-temporal structure. These biases must actively inform your machine learning model. Including an interaction term because you know two species interact, or constraining a neural network to respect energy conservation, is what distinguishes this project from a generic ML exercise.

**Machine learning.** Apply one or more data-driven methods from this course or beyond: SINDy, symbolic regression, PINNs, neural ODEs, DMD, Koopman analysis, autoencoders, or others. Handle training and test data properly. Tune hyperparameters. Compare approaches where appropriate.


## Data

You may use synthetic data, real data, or both.

**Synthetic data** means simulating a known model, adding noise, and attempting to recover the model or make predictions. The contribution here is methodological: you are testing, comparing, or improving algorithms under controlled conditions.

**Real data** means working with measurements from labs, sensors, public repositories, or field observations. The contribution is applied: you are discovering something about a real system. Real data is preferred and will be weighted more favorably in grading, but synthetic data with a genuine methodological contribution is also valuable.

Sources for real data include your own research lab, public repositories (UCI ML Repository, Kaggle, PhysioNet, NOAA, NASA Earthdata), or data shared alongside published papers.

Temporal, spatial, or spatio-temporal data is ideal for this course.


## Iteration and Documentation

The central requirement of this project is **documented iteration**.

Every time you work on the project, record what you tried, what worked, what failed, and what you decided to do next. If you consulted an LLM, note what you asked, what it suggested, and whether you followed its recommendation. If you made a decision at a junction point (change the features, switch datasets, try a different model), explain why.

This iteration log goes in the appendix of your final paper. It may span 10 to 20 pages. It is your research notebook. It serves three purposes: it makes the learning process visible for assessment, it creates a reference for your future work, and it forces reflection on your own reasoning.

I evaluate the quality of the process, not just the quality of the result. A project that tries three approaches, fails twice, and succeeds on the third attempt with a clear explanation of why the first two failed is more valuable than a project that gets the right answer on the first try with no explanation.


## The Paper

The final deliverable is a paper of 6 to 9 pages (main body) plus an appendix with your iteration log and any supplementary material.

The paper should follow a standard structure: Abstract, Introduction, Methods, Results, Discussion, and References. Use a conference LaTeX template (NeurIPS, ICML, or similar).

**Writing quality matters.** Go read papers at top venues (NeurIPS, ICLR, Nature, PNAS, Journal of Computational Physics) and study how they write. Your paper should be clear, precise, and engaging. The standard is: could this be submitted to a workshop or conference?

You are encouraged to use LLMs to help with writing, editing, and debugging code. Use them as editors, not authors. The paper should read as your voice, informed by your thinking. If a sentence sounds impressive but you cannot explain what it means, delete it. If I read something and it is clear that neither you nor the LLM understood what was written, it will cost you points. If I read something and it captivates me, that is what earns a high grade.

The baseline has shifted. Ten years ago, a well-written paper from a course like this was impressive. Now you have tools that help with every aspect of the writing process. A merely competent paper is no longer impressive. What impresses is depth, originality, and evidence of genuine engagement with the problem.


## Presentation

At the end of the semester, you will present your work to the class. This is an oral presentation followed by questions. You should be able to explain every decision in your paper, defend your approach, and discuss what you would do differently. If your iteration log says you made a decision, you should be able to explain it live.


## Timeline

| Week | Milestone |
|------|-----------|
| 1 | **Pre-proposal** posted on Slack. Brief idea, dataset, rough methods, 3+ references. I give feedback in class. |
| 2 to 6 | Weekly check-ins. You present your plan for the week, report progress, get feedback. Begin iteration log. First results expected. Pivots are normal. |
| 7 | **Progress report** submitted. Current state, preliminary results, updated plan. Pre-proposal and progress report graded together (5%). |
| 8 to 12 | Continued iteration and writing. Weekly check-ins continue. Paper takes shape. |
| Final | **Paper and presentation** submitted. 6 to 9 pages plus appendix. Oral presentation with Q&A. Graded (30%). |


## Project Ideas

These are starting points, not prescriptions. The best projects come from your own questions.

**Equation discovery.** Apply SINDy to real ecological or epidemiological data. Use PDE-FIND on fluid flow measurements. Compare SINDy and symbolic regression on the same system. Discover scaling laws from experimental data.

**Physics-informed ML.** Use PINNs for inverse problems (estimating parameters of a known PDE from noisy data). Train a neural ODE with conservation constraints. Build a Hamiltonian or Lagrangian neural network for a mechanical system. Create a surrogate model for an expensive simulation.

**Data-driven dynamics.** Apply DMD or Koopman analysis to video data. Infer governing equations from pendulum or double-pendulum video. Forecast weather or climate time series. Analyze sensor data from robotics or biomechanics labs.

**Hybrid approaches.** Combine a learned component with a known model structure. Use an autoencoder to discover latent variables, then apply SINDy in the latent space. Train a PINN where part of the PDE is known and part is learned.


## Reading Papers

One of the most important skills you will develop in this project is reading research papers. There are no courses on how to read papers. You learn by practice.

Start with the papers assigned in the course (Brunton et al. 2016, Rudy et al. 2017, Raissi et al. 2019). Follow their citation graphs forward using Google Scholar, Semantic Scholar, or Connected Papers. Find the latest work that builds on these methods.

On a first pass: read the abstract and introduction carefully. Skim the method to understand inputs, outputs, and assumptions. Look at the figures and tables. Read the conclusion. You do not need to understand every equation. Get the gist. Understand what question they asked and how they answered it.

You may use an LLM to help navigate the literature. It is a useful search tool. But the LLM summary is not the paper. You still need to read the papers themselves.


## Grading Rubric

| Criterion | Weight | What I look for |
|-----------|--------|-----------------|
| Scientific framing | 20% | Clear question, relevant inductive biases, well-motivated |
| ML methodology | 20% | Correct application, proper data handling, hyperparameter tuning |
| Iteration quality | 20% | Documented process, meaningful pivots, evidence of learning from failure |
| Paper quality | 25% | Clear writing, good figures, proper references, conference-level presentation |
| Oral presentation | 15% | Can explain and defend every decision; responds thoughtfully to questions |
