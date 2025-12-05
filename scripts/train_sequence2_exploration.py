"""
Training script for Sequence 2: Exploration
Goal: Learn to explore Pallet Town and reach grass area
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.logger import configure

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokemon_env import PokemonRedEnv, PokemonRedWrapper
from config import config_sequence2_exploration as config

def make_env(env_id=0, headless=False, record_video=False):
    """Create a single environment"""
    def _init():
        env = PokemonRedEnv(
            rom_path=config.ROM_PATH,
            render_mode="rgb_array",
            headless=headless,
            record_video=record_video and (env_id == 0),  # Only first env records
            video_fps=60,
            env_id=f"seq2_env{env_id}",
            training_mode="exploration"
        )
        env = PokemonRedWrapper(env, stack_frames=4)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train Sequence 2: Exploration")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=config.NUM_ENVS, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (faster)")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--model-name", type=str, default="sequence2_exploration", help="Model name prefix")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--save-freq", type=int, default=config.SAVE_FREQ, help="Save frequency")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_name}_{timestamp}"
    model_path = os.path.join(config.MODEL_DIR, run_name)
    log_path = os.path.join(config.LOG_DIR, run_name)
    
    print(f"\n{'='*60}")
    print(f"Training Sequence 2: Exploration")
    print(f"{'='*60}")
    print(f"Model: {run_name}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Environments: {args.envs}")
    print(f"Headless: {args.headless}")
    print(f"Video Recording: {not args.no_video}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"{'='*60}\n")
    
    # Create vectorized environment
    envs = [make_env(i, headless=args.headless, record_video=not args.no_video) for i in range(args.envs)]
    vec_env = DummyVecEnv(envs)
    
    # Note: Frame stacking is handled by PokemonRedWrapper, not VecFrameStack
    # This ensures consistency with play.py
    
    # Transpose images for CNN (HWC -> CHW)
    vec_env = VecTransposeImage(vec_env)
    
    # Create PPO model
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        verbose=args.verbose,
        tensorboard_log=log_path,
        device=args.device
    )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=model_path,
        name_prefix=args.model_name,
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Train
    print("Starting training...\n")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(model_path, f"{args.model_name}_final.zip")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Close environments
    vec_env.close()

if __name__ == "__main__":
    main()

