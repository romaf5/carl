"""
Configuration file for PPO training on CarRacing-v3 environment.
"""

# Environment settings
ENV_ID = "CarRacing-v3"
N_ENVS = 8  # Number of parallel environments for training
FRAME_STACK = 4  # Number of frames to stack for temporal information

# PPO Hyperparameters
LEARNING_RATE = 3e-4  # Initial learning rate for PPO
USE_LINEAR_LR_DECAY = True  # Linearly decay learning rate to 0
N_STEPS = 2048  # Number of steps to collect before each update
BATCH_SIZE = 64  # Minibatch size for optimization
N_EPOCHS = 10  # Number of epochs for optimization
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
CLIP_RANGE = 0.2  # Clipping parameter for PPO
ENT_COEF = 0.01  # Entropy coefficient for exploration
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5  # Gradient clipping

# Training settings
TOTAL_TIMESTEPS = 1_000_000  # Total training timesteps (1M for baseline)
SAVE_FREQ = 50_000  # Save model every 50k steps
LOG_DIR = "./logs"  # Directory for tensorboard logs
MODEL_DIR = "./models"  # Directory for saved models
MODEL_NAME = "ppo_carracing"  # Base name for saved models

# Evaluation settings
N_EVAL_EPISODES = 10  # Number of episodes for evaluation
EVAL_FREQ = 10_000  # Evaluate every N steps during training
RENDER_EVAL = False  # Whether to render during evaluation
