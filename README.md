# LunaPilot AI

### Training an Autonomous Rocket Landing Agent with Reinforcement Learning

LunaPilot AI is an AI engineering project exploring how an autonomous agent can learn to land a rocket through trial and error, using reward signals, penalties, and policy optimization.

The project uses a simulated Lunar Lander environment to investigate Reinforcement Learning and reward shaping.

## Objectives

- Train a rocket landing agent using Proximal Policy Optimization (PPO).
- Understand how observations, actions, rewards, and policies interact.
- Experiment with custom reward functions for landing speed, stability, and fuel efficiency.
- Investigate reward hacking and develop better reward designs.
- Evaluate model performance using episode rewards, landing success rates, and learning curves.
- Develop an interactive visualization dashboard for rocket simulation and training progress.

## Tech Stack

- Python 3.11
- Gymnasium
- Stable-Baselines3
- PPO (Proximal Policy Optimization)
- PyTorch
- Matplotlib and TensorBoard
- FastAPI (planned)
- React and Three.js (planned)

## Development Roadmap

- [ ] Set up the simulation environment
- [ ] Implement and evaluate a random agent
- [ ] Train the baseline PPO agent
- [ ] Record and evaluate landing attempts
- [ ] Implement custom reward shaping
- [ ] Investigate reward hacking
- [ ] Compare baseline and modified reward strategies
- [ ] Build the interactive visualization dashboard
- [ ] Document experiments and results

## Project Status

In development.

This repository documents the implementation, experiments, challenges, and lessons learned throughout the project.


## Initial Evaluation Results

The PPO agent was evaluated over 20 episodes in the Gymnasium LunarLander-v3 environment.

- **Average episode reward:** 246.58
- **Successful landings:** 16/20 (80%)
- **Other terminal episodes:** 2
- **Time-limit endings:** 2

Success was defined using episode termination, contact from both landing legs, and a non-negative episode reward. This is a preliminary success criterion and will be refined with additional outcome checks.

**Status:** In development. Further evaluation and reward analysis are planned.

## Learning Approach

The goal is not simply to maximize reward, but to investigate whether the learned policy produces safe and reliable landings.

Training results and performance claims will be documented as experiments are completed.

## Author

Muhammad Furqan Iftikhar

AI Engineer | AI & Data Science Trainer
