"""
Evaluation script for trained PPO models on CarRacing-v3.
"""
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import config


def make_env(render_mode=None):
    """Create and wrap the CarRacing environment."""
    env = gym.make(config.ENV_ID, continuous=True, render_mode=render_mode)
    return env


def evaluate(model_path, n_episodes=None, render=None):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to the trained model (.zip file)
        n_episodes: Number of episodes to evaluate (defaults to config)
        render: Whether to render the environment (defaults to config)
    """
    if n_episodes is None:
        n_episodes = config.N_EVAL_EPISODES
    if render is None:
        render = config.RENDER_EVAL

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create environment
    render_mode = "human" if render else None
    env = DummyVecEnv([lambda: make_env(render_mode=render_mode)])
    env = VecFrameStack(env, n_stack=config.FRAME_STACK)

    print(f"Evaluating for {n_episodes} episodes...")
    print(f"Render: {render}")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    env.close()

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f} +/- {std_length:.2f}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")
    print("=" * 50)

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on CarRacing-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=config.N_EVAL_EPISODES,
        help=f"Number of episodes to evaluate (default: {config.N_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation",
    )

    args = parser.parse_args()

    evaluate(args.model_path, args.n_episodes, args.render)


if __name__ == "__main__":
    main()
