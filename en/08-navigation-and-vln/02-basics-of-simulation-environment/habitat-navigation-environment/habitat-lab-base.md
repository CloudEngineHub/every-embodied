# Habitat-Lab Concepts, Tasks, and APIs

Habitat-Lab is an algorithm layer based on Habitat-Sim, encapsulating standardized tasks, evaluation metrics, and APIs. This allows researchers to focus on the design, training, and evaluation of embodied AI algorithms without having to worry about the details of the underlying simulation.

## 1. Introduction to Habitat-lab Features

### 1. Predefined embodied tasks

* Navigation types: PointGoalNav (point target navigation), ObjectNav (object target navigation), VLN (visual language navigation).
* Interaction types: Rearrange (object rearrangement), PickPlace (pick and place), OpenDoor (open door).
* Support for custom tasks (such as multi-agent collaboration, long-term planning).

### 2. Standardized Evaluation System

* Built-in core metrics (success rate SR, path length efficiency SPL, DTW alignment score, etc.) allow direct comparison of algorithm performance.
* Supports alignment with public benchmarks (such as Habitat Challenge), facilitating paper reproduction and result comparison.

### 3. Modular Agent Architecture

* Decouple “sensor → encoder → policy → controller”, enabling quick replacement of components (e.g., use CNN or Transformer for visual encoding, and RL/IL for decision-making);
* Deeply integrate PyTorch, supporting mainstream algorithms such as reinforcement learning (RL), imitation learning (IL), and end-to-end deep learning.

### 4. Easy-to-use API

* Provides a simple Python interface for one-click loading of scenarios, initialization of agents, and execution of simulation loops.

## II. Habitat-lab Environment Setup

The conda environment created by habitat-sim can be used directly here.

```python
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

When downloading habitat-lab, make sure it is consistent with the habitat-sim version installed. For example, if the habitat-sim version used in this tutorial is 0.2.5, then you need to download the 0.2.5 version of habitat-lab. When reproducing other papers, also ensure that the habitat-sim and habitat-lab versions are the same.

```python
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

Install habitat-baselines simultaneously

```python
pip install -e habitat-baselines
```

Download 3D scene data and point navigation data

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

```python
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
```
