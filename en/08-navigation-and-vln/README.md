# Embodied Navigation and Vision-Language Navigation

This chapter progresses from classical map-based navigation to Habitat simulation, continuous-environment vision-language navigation, and active perception. The goal is to connect observations, maps, goals, planners, controllers, and metrics into one navigation loop.

## Learning Sequence

1. [Navigation Algorithm Foundations](./01-navigation-algorithms-practical-guide/basic-introduction-to-embodied-navigation-algorithms.md)
2. [Habitat-Sim Environment and Datasets](./02-basics-of-simulation-environment/habitat-navigation-environment/habitat-sim-environment-setup-and-dataset-introduction.md)
3. [Habitat-Sim Basic Practice](./02-basics-of-simulation-environment/habitat-navigation-environment/habitat-sim-basic-practices.md)
4. [Habitat-Lab Foundations](./02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-base.md)
5. [Habitat-Lab Setup and Configuration](./02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-environment-setup-and-configuration.md)
6. [Habitat-Lab Automatic Navigation](./02-basics-of-simulation-environment/habitat-navigation-environment/habitat-lab-basic-practices.md)
7. [Vision-Language Navigation in Continuous Environments](./03-frontier-vln-reproduction/01VLNCE/01-overview-of-visual-language-navigation-vln-ce-method-in-continuous-environment.md)
8. [ETPNav Reproduction](./03-frontier-vln-reproduction/01VLNCE/02-etpnav-code-reproduction.md)
9. [NBS Active-Perception Guide](./03-frontier-vln-reproduction/03-actively-perceive-nbs-guide/README.md)

## Completion Criteria

- Distinguish localization, mapping, global planning, local planning, and motion control.
- Load a scene, configure sensors, generate navigation trajectories, and read metrics in Habitat.
- Explain how vision-language navigation differs from point-goal navigation in goal representation and policy inputs.
- Compare navigation policies using success rate, success weighted by path length, and distance to goal.
