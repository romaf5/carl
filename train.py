"""
Training script for PPO on CarRacing-v3 environment.
"""
import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

import config


def linear_schedule(initial_value):
    """
    Linear learning rate schedule that decays to 0.

    Args:
        initial_value: Initial learning rate

    Returns:
        Schedule function that takes progress (0 to 1) and returns current LR
    """
    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0 (end).

        Args:
            progress_remaining: Remaining progress (1.0 at start, 0.0 at end)

        Returns:
            Current learning rate
        """
        return progress_remaining * initial_value

    return func


def make_env(rank):
    """Create and wrap the CarRacing environment."""
    def _init():
        env = gym.make(config.ENV_ID, continuous=True)
        env = Monitor(env)  # Monitor wrapper for logging
        return env
    return _init


def train():
    """Main training function."""
    # Create directories if they don't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, using CPU!")

    print(f"\nSetting up training for {config.ENV_ID}...")
    print(f"Number of parallel environments: {config.N_ENVS}")
    print(f"Total timesteps: {config.TOTAL_TIMESTEPS:,}")

    # Create multiple parallel environments with frame stacking
    print(f"Creating {config.N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
    env = VecFrameStack(env, n_stack=config.FRAME_STACK)

    print(f"Environments created with {config.FRAME_STACK} frame stack")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Total steps per rollout: {config.N_STEPS} steps × {config.N_ENVS} envs = {config.N_STEPS * config.N_ENVS} steps")

    # Create evaluation environment (single env for consistent evaluation)
    # Use SubprocVecEnv to match training env wrapper structure
    print(f"\nCreating evaluation environment...")
    eval_env = SubprocVecEnv([make_env(0)])
    eval_env = VecFrameStack(eval_env, n_stack=config.FRAME_STACK)
    # Apply VecTransposeImage to match what PPO does automatically for CnnPolicy
    eval_env = VecTransposeImage(eval_env)

    # Setup learning rate (with optional linear decay)
    if config.USE_LINEAR_LR_DECAY:
        learning_rate = linear_schedule(config.LEARNING_RATE)
        print(f"Using linear learning rate decay from {config.LEARNING_RATE} to 0")
    else:
        learning_rate = config.LEARNING_RATE
        print(f"Using constant learning rate: {config.LEARNING_RATE}")

    # Initialize PPO agent
    model = PPO(
        policy="CnnPolicy",  # CNN policy for image observations
        env=env,
        learning_rate=learning_rate,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
        device=device,  # Explicitly set device for GPU training
    )

    print("\nPPO agent initialized with hyperparameters:")
    print(f"  Device: {model.device}")
    print(f"  Initial learning rate: {config.LEARNING_RATE}")
    print(f"  LR schedule: {'Linear decay' if config.USE_LINEAR_LR_DECAY else 'Constant'}")
    print(f"  N steps: {config.N_STEPS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  N epochs: {config.N_EPOCHS}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.SAVE_FREQ,
        save_path=config.MODEL_DIR,
        name_prefix=config.MODEL_NAME,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Evaluation callback to save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_DIR,
        log_path=config.LOG_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"\nStarting training...")
    print(f"Checkpoints saved every {config.SAVE_FREQ:,} steps to {config.MODEL_DIR}")
    print(f"Evaluation every {config.EVAL_FREQ:,} steps ({config.N_EVAL_EPISODES} episodes)")
    print(f"Best model will be saved to {config.MODEL_DIR}/best_model.zip")
    print(f"Tensorboard logs: {config.LOG_DIR}")
    print(f"Monitor with: tensorboard --logdir={config.LOG_DIR}\n")

    # Train the agent
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        final_model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_final")
        model.save(final_model_path)
        print(f"\nTraining completed!")
        print(f"Final model saved to {final_model_path}.zip")
        print(f"Best model saved to {config.MODEL_DIR}/best_model.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        interrupted_model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_interrupted")
        model.save(interrupted_model_path)
        print(f"Model saved to {interrupted_model_path}.zip")
        print(f"Best model saved to {config.MODEL_DIR}/best_model.zip")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()
