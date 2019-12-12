## QP

This repo explores general formulation of **LP** (linear programs)  and **QP** (quadratic programs) with applications in modeling, prediction, and control. These formulations have well-known solution methods that scale and are fully developed technologies integrated into most programming languages (making them readily available for embedded applications).  State-of-the-art implamentation uses hardware accelerated distributed optimization specialized for sparse representations and parallelization.


Mathematical Background:
- A high-level **writeup** can be found 
[here](https://github.com/Luke-A-Wendt/QP/blob/master/tex/main.pdf).

[The Bitter Lesson:](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- Simple models with scalable computation are outperforming complex models with 'baked-in heuristics'
- The focus needs to be on hardware-centric optimization, e.g., GPU acceleration

Project Goals:
- Develop programatic way of adding convex objectives and constraints.
- Explore large-scale optimization with sparsity and ADMM
- Develop tools for generalized nonlinear and stochastic MPC problems
- Develop Sparse ADMM neural network tools
- Early development is being done in Matlab and Python.
- Further optimization of data structures will be done in C++ with a Python API.
- GPU acceleration will be added for large problems and/or real-time control application.
