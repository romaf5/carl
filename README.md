# CarRacing PPO Baseline

A baseline reinforcement learning implementation using Proximal Policy Optimization (PPO) for the Gymnasium CarRacing-v3 environment.

## Overview

This project implements a complete training pipeline for training an RL agent to play the CarRacing game using PPO from Stable-Baselines3. The CarRacing environment presents a continuous control challenge where the agent must learn to drive a car around a track by observing RGB images.

## Project Structure

```
carsim/
├── requirements.txt          # Project dependencies
├── .gitignore               # Git ignore patterns
├── README.md                # Project documentation
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── config.py                # Hyperparameter configuration
├── models/                  # Saved model checkpoints (gitignored)
└── logs/                    # Tensorboard logs (gitignored)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd carsim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `gymnasium[box2d]` - The CarRacing environment
- `stable-baselines3[extra]` - PPO implementation with TensorBoard support
- `torch` - Deep learning backend
- `numpy` - Numerical operations
- `pillow` - Image processing

## Usage

### Training

Start training with default hyperparameters:

```bash
python train.py
```

The training script will:
- Create the CarRacing-v3 environment with frame stacking
- Initialize a PPO agent with a CNN policy
- Train for 1 million timesteps (configurable in `config.py`)
- Save model checkpoints every 50,000 steps to `./models/`
- Log training metrics to `./logs/` for TensorBoard

### Monitoring Training

View training progress in real-time with TensorBoard:

```bash
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your browser to see:
- Episode rewards over time
- Policy loss and value loss
- Entropy (exploration measure)
- Learning rate schedule

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model-path models/ppo_carracing_final.zip
```

Options:
- `--model-path`: Path to the model file (required)
- `--n-episodes`: Number of episodes to evaluate (default: 10)
- `--render`: Add this flag to visualize the agent playing

Example with rendering:
```bash
python evaluate.py --model-path models/ppo_carracing_final.zip --n-episodes 5 --render
```

## Configuration

Hyperparameters can be adjusted in `config.py`:

### Key Hyperparameters

- **Learning Rate**: 3e-4 (standard for PPO)
- **Steps per Update**: 2048 episodes collected before each policy update
- **Batch Size**: 64 for gradient descent
- **Epochs**: 10 optimization epochs per update
- **Discount Factor (γ)**: 0.99
- **GAE Lambda (λ)**: 0.95 for advantage estimation
- **Total Timesteps**: 1,000,000 (sufficient for baseline performance)

### Environment Settings

- **Parallel Environments**: 8 environments running simultaneously for faster training
- **Frame Stack**: 4 consecutive frames stacked to provide temporal information
- **Action Space**: Continuous 3D [steering, gas, brake]
- **Observation Space**: Stacked RGB images (96, 96, 3) × 4

## Expected Results

- **Initial Performance**: Negative rewards as the agent learns basic controls
- **Learning Progress**: Gradual improvement visible within 500k-1M timesteps
- **Baseline Performance**: 700-900+ average reward with proper training
- **Training Time**: 1-3 hours for 1M timesteps with GPU and 8 parallel environments

## Environment Details

### CarRacing-v3

- **Type**: Continuous control task
- **Observation**: 96×96 RGB images from top-down view
- **Actions**:
  - Steering: [-1, 1] (left to right)
  - Gas: [0, 1]
  - Brake: [0, 1]
- **Rewards**:
  - -0.1 for each frame
  - +1000/N for each track tile visited (N = total tiles)
  - Negative reward for going off track
- **Episode End**: Track completion or going off track

## Troubleshooting

### Common Issues

1. **Box2D Installation Error**
   ```bash
   pip install box2d-py
   ```

2. **CUDA/GPU Issues**
   - Ensure PyTorch is installed with CUDA support if using GPU
   - Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Memory Issues**
   - Reduce `N_STEPS` or `BATCH_SIZE` in `config.py`
   - Use CPU instead of GPU for smaller memory footprint

4. **Slow Training**
   - Ensure GPU is being used if available
   - Consider reducing `TOTAL_TIMESTEPS` for initial experiments
   - Monitor TensorBoard to verify learning is happening

## File Descriptions

- **train.py**: Main training loop with environment setup, PPO initialization, and callbacks
- **evaluate.py**: Model evaluation with statistics and optional rendering
- **config.py**: Centralized hyperparameter configuration
- **requirements.txt**: Python package dependencies
- **.gitignore**: Excludes model checkpoints and logs from version control

## Next Steps

After baseline training:
1. Experiment with hyperparameters in `config.py`
2. Try longer training runs (2M+ timesteps)
3. Implement custom reward shaping
4. Add curriculum learning
5. Experiment with different network architectures
6. Compare with other algorithms (SAC, TD3, etc.)

## References

- [Gymnasium CarRacing](https://gymnasium.farama.org/environments/box2d/car_racing/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
