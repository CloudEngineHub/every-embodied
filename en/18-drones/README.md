# Drone Systems

This chapter uses a quadrotor to connect coordinate transforms, dynamics, controllers, and trajectory planning. Both tutorials combine derivations with Python verification and should be studied in the order “control first, planning second.”

## Prerequisites

- Linear algebra, ordinary differential equations, and basic optimization.
- Python, NumPy, and Matplotlib.
- Robot coordinate frames and feedback-control foundations.

## Chapters

1. [Quadrotor Control](./drone-control-tutorial.md): rotation matrices, dynamics, differential flatness, PID, model-based control, 3D tracking, SE(3) control, and MPC.
2. [Quadrotor Trajectory Planning](./drone-planning-tutorial.md): polynomial trajectories, minimum snap, corridor constraints, time allocation, and closed-form solutions.

## Learning Outcome

After completing this chapter, explain the interface between a planner and a controller, derive the basic quadrotor dynamics, and run the trajectory-generation and tracking examples.
