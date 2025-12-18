# PPO CartPole Implementation

A minimal and clean Proximal Policy Optimization (PPO) implementation in PyTorch for the CartPole-v1 environment, designed for academic demonstrations and research.

## Features

- **Actor-Critic Architecture**: Shared feature extractor with separate actor and critic heads
- **PPO Algorithm**: Clipped surrogate objective with configurable epsilon values
- **GAE**: Generalized Advantage Estimation (λ=0.95) for advantage computation
- **Epsilon Sensitivity Study**: Systematic comparison of clipping parameters (ε = 0.1, 0.2, 0.3, 0.5)
- **Comprehensive Metrics**: Tracks returns, KL divergence, losses, entropy, value estimates, and success rates
- **Visualization**: Generates 6 research plots for analysis
- **Video Recording**: Record trained agent demonstrations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train PPO agents with different epsilon values and generate all plots:

```bash
python ppo_cartpole.py
```

This will:
1. Train 4 PPO agents (ε = 0.1, 0.2, 0.3, 0.5) for 100,000 steps each
2. Generate 6 research plots saved to `./results/`
3. Save trained models to `./results/models/`
4. Record video demonstrations

### Recording Videos

Record videos of a trained agent:

```bash
python record_video.py --model_path ./results/models/ppo_epsilon_0.2.pth --epsilon 0.2 --num_episodes 5
```

### Architecture

- **Feature Extractor**: 2-layer MLP (64 hidden units, tanh activation)
- **Actor**: Linear layer → softmax over discrete actions
- **Critic**: Linear layer → state value estimate

### Training Configuration

- Total steps: 100,000
- Update frequency: 2,048 steps
- PPO epochs: 4 per update
- Batch size: 64
- Learning rate: 3e-4
- Discount (γ): 0.99
- GAE lambda (λ): 0.95

## Outputs

### Generated Plots

1. **training_curves.png**: Episode return vs training steps
2. **kl_divergence.png**: KL divergence between policies vs updates
3. **value_estimates.png**: Value function estimates over training
4. **policy_entropy.png**: Policy entropy (exploration measure)
5. **loss_curves.png**: Policy and value losses
6. **success_rate.png**: Success rate (Return ≥ 195) over training


## References

- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms". arXiv:1707.06347

