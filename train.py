"""
Training script for Pokemon Red RL agent
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.logger import configure

from pokemon_env import PokemonRedEnv, PokemonRedWrapper
from config import *


class TrainingCallback(CheckpointCallback):
    """Custom callback for training visualization and logging"""
    
    def __init__(self, save_freq, save_path, name_prefix="pokemon_rl_model", verbose=1):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if "r" in ep_info:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                
                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_length = np.mean(self.episode_lengths[-10:])
                    print(f"\n{'='*50}")
                    print(f"Episode: {len(self.episode_rewards)}")
                    print(f"Mean Reward (last 10): {mean_reward:.2f}")
                    print(f"Mean Length (last 10): {mean_length:.2f}")
                    print(f"Total Steps: {self.num_timesteps}")
                    print(f"{'='*50}\n")
        
        # Log action distribution every ~5000 steps
        if self.num_timesteps % 5000 == 0:
            try:
                # Access the underlying env to fetch action stats
                vec_env = self.model.get_env()
                # Our env is DummyVecEnv with one env
                base_env = vec_env.envs[0]
                # Unwrap Monitor -> PokemonRedWrapper -> PokemonRedEnv
                wrapped = getattr(base_env, "env", base_env)
                core_env = getattr(wrapped, "env", wrapped)
                if hasattr(core_env, "get_action_stats"):
                    stats = core_env.get_action_stats()
                    counts = stats["counts"]
                    total = max(stats["total"], 1)
                    # Log per-action probabilities to TensorBoard
                    for i, c in enumerate(counts):
                        self.model.logger.record(f"actions/count_{i}", c)
                        self.model.logger.record(f"actions/p_{i}", c / total)
                    self.model.logger.record("actions/total", total)
            except Exception:
                pass

        return super()._on_step()


def make_env(rom_path, headless=False):
    """Create and wrap the environment"""
    def _init():
        env = PokemonRedEnv(rom_path=rom_path, render_mode="human", headless=headless)
        env = PokemonRedWrapper(env, stack_frames=4)
        env = Monitor(env)
        return env
    return _init


def train(
    rom_path=ROM_PATH,
    total_timesteps=TOTAL_TIMESTEPS,
    headless=False,
    load_model=None
):
    """
    Train the RL agent
    
    Args:
        rom_path: Path to Pokemon Red ROM file
        total_timesteps: Total number of training steps
        headless: If True, run without rendering (faster training)
        load_model: Path to existing model to continue training
    """
    print("="*70)
    print("POKEMON RED RL TRAINING")
    print("="*70)
    print(f"ROM Path: {rom_path}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Headless Mode: {headless}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create run-specific directories
    run_model_dir = os.path.join(MODEL_DIR, f"run_{timestamp}")
    run_log_dir = os.path.join(LOG_DIR, f"run_{timestamp}")
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    env = DummyVecEnv([make_env(rom_path, headless=headless)])
    
    # Create or load model
    if load_model and os.path.exists(load_model):
        print(f"\n[2/4] Loading existing model from {load_model}...")
        model = PPO.load(load_model, env=env)
        print("Model loaded successfully!")
    else:
        print("\n[2/4] Creating new PPO model...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            verbose=1,
            tensorboard_log=run_log_dir,
            device="auto"  # Automatically use CUDA if available
        )
        print("Model created successfully!")
    
    # Configure logger
    logger = configure(run_log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # Create callbacks
    print("\n[3/4] Setting up callbacks...")
    checkpoint_callback = TrainingCallback(
        save_freq=SAVE_FREQ,
        save_path=run_model_dir,
        name_prefix=f"pokemon_rl_model_{timestamp}",
        verbose=1
    )
    
    # Start training
    print("\n[4/4] Starting training...")
    print("\nTIP: You can monitor training with TensorBoard:")
    print(f"    tensorboard --logdir={run_log_dir}")
    print("\nPress Ctrl+C to stop training gracefully.\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(run_model_dir, f"pokemon_final_{timestamp}.zip")
        model.save(final_model_path)
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Run directory: {run_model_dir}")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        final_model_path = os.path.join(run_model_dir, f"pokemon_interrupted_{timestamp}.zip")
        model.save(final_model_path)
        print(f"Model saved to: {final_model_path}")
        print(f"Run directory: {run_model_dir}")
    
    finally:
        env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL agent to play Pokemon Red")
    parser.add_argument(
        "--rom",
        type=str,
        default=ROM_PATH,
        help="Path to Pokemon Red ROM file"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering (faster training)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to existing model to continue training"
    )
    
    args = parser.parse_args()
    
    # Check if ROM exists
    if not os.path.exists(args.rom):
        print(f"ERROR: ROM file not found at {args.rom}")
        print("Please ensure PokemonRed.gb is in the roms/ directory")
        return
    
    train(
        rom_path=args.rom,
        total_timesteps=args.timesteps,
        headless=args.headless,
        load_model=args.load
    )


if __name__ == "__main__":
    main()

