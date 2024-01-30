# Solving the Quadratic Assignment Problem using Deep Reinforcement Learning

## Overview
This repository contains an implementation for the [preprint](https://arxiv.org/abs/2310.01604) by Puneet S. Bagga and Dr. Arthur Delarue. Houses the code for the Dual Pointer model (the model used in the paper) and an additional Linear Terms model.

**Note**: This codebase is research-oriented and unrefined, as it was developed over an extensive period. The primary focus was on research and experimentation rather than clean code practices. Please feel free to reach out to us if you have any questions or issues.

## Models
There are two main models in this repository:
1. **Dual Pointer**: RNN-based model, where at each step, go from (a) placing a facility in a location or (b) picking a location for the next facility. Train a separate pointer net for both (a) and (b), with an attention mechanism to possible selections.
The primary model used in the paper.

2. **Linear-Terms**: Attention based model. Use cross-attention and self-attention from flow and distance embeddings to select a facility-location mapping ((i,j) for permutation matrix). Distance embeddings are a combination of distances between remaining locations to be picked and a linear cost of possible mappings left to pick (derived from previous mappings). Flow embeddings are computed for the remaining facilities to be placed.
An alternative model developed to explore and test different ideas.


## Data
All data used in this project is synthetic and managed under `QAP.py`. Default data is symmetric flow and distance matrices, which can be modified via hyperparameters.

`QAP.py` also contains reward functions for the QAP, which are used in the training process.

## Setup and Run
### Prerequisites

Use `conda create --name [your_env_name] --file spec-file.txt` to set up the environment.

Alternatively,  install the following packages directly. This may offer more flexibility regarding package versions, especially for CUDA versions.
- PyTorch
- PyTorch Geometric
- Weights & Biases (wandb)

### Running the Models
- To run the Dual Pointer model, use `setup.py`.
- To use the Linear-Terms model, execute `trainer.py`.


## Support
For any questions or issues, please feel free to email any of the following:

[psbagga@gatech.edu](mailto:psbagga@gatech.edu)
[psbagga17@gmail.com](mailto:psbagga17@gmail.com)
[arthur.delarue@isye.gatech.edu](mailto:arthur.delarue@isye.gatech.edu)


