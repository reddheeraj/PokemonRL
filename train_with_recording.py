"""
Training script with video recording and headless support
"""

import os
import time
import argparse
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from pokemon_env import PokemonRedEnv, PokemonRedWrapper
from config import *

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = np.zeros(9, dtype=np.int64)
        
    def _on_step(self) -> bool:
        # Log action distributions
        if hasattr(self.training_env, 'get_attr'):
            try:
                action_counts = self.training_env.get_attr('action_counts')[0]
                self.action_counts += action_counts
            except:
                pass
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log episode statistics
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            
            print(f"Rollout completed - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
            
            # Log action distribution
            total_actions = np.sum(self.action_counts)
            if total_actions > 0:
                action_probs = self.action_counts / total_actions
                print(f"Action distribution: {action_probs}")
                self.action_counts.fill(0)

def create_env(env_id, headless=True, record_video=False):
    """Create a single environment"""
    def _init():
        # FIXED: All environments use rgb_array mode for vectorized compatibility
        env = PokemonRedEnv(
            rom_path=ROM_PATH,
            render_mode="rgb_array",  # Always use rgb_array for vectorized envs
            headless=headless,
            record_video=record_video,
            video_fps=30,  # Lower FPS for recording to save space
            env_id=env_id  # Pass environment ID for unique video filenames
        )
        env = PokemonRedWrapper(env, stack_frames=4)
        env = Monitor(env)
        return env
    return _init

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Pokemon Red RL Agent")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                       help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})")
    parser.add_argument("--envs", type=int, default=NUM_ENVS,
                       help=f"Number of parallel environments (default: {NUM_ENVS})")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                       help=f"Learning rate (default: {LEARNING_RATE})")
    
    # Environment settings
    parser.add_argument("--headless", action="store_true",
                       help="Run all environments in headless mode (no windows)")
    parser.add_argument("--hybrid", action="store_true", default=True,
                       help="Use hybrid mode: 1 windowed env for video, rest headless (default)")
    parser.add_argument("--no-video", action="store_true",
                       help="Disable video recording")
    
    # Model settings
    parser.add_argument("--model-name", type=str, default=None,
                       help="Custom name for model directory")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device to use for training (default: auto)")
    
    # Training control
    parser.add_argument("--save-freq", type=int, default=SAVE_FREQ,
                       help=f"Save model every N steps (default: {SAVE_FREQ:,})")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                       help="Verbosity level (default: 1)")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Determine environment setup
    if args.headless:
        # All headless
        env_setup = "ALL HEADLESS"
        video_recording = False
    elif args.no_video:
        # All windowed, no video
        env_setup = "ALL WINDOWED (no video)"
        video_recording = False
    else:
        # Hybrid mode (default)
        env_setup = "HYBRID (1 windowed + video, rest headless)"
        video_recording = True
    
    print("=" * 80)
    print("POKEMON RED RL TRAINING WITH COMMAND LINE ARGUMENTS")
    print("=" * 80)
    print(f"ROM Path: {ROM_PATH}")
    print(f"Total Timesteps: {args.timesteps:,}")
    print(f"Environments: {args.envs}")
    print(f"Environment Setup: {env_setup}")
    print(f"Video Recording: {'Enabled' if video_recording else 'Disabled'}")
    print(f"Device: {device}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 80)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model_name:
        run_dir = f"models/{args.model_name}_{timestamp}"
    else:
        run_dir = f"models/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n[1/4] Creating {args.envs} parallel environments...")
    
    # Create environments based on arguments
    env_fns = []
    for i in range(args.envs):
        if args.headless:
            # All headless
            record_video = False
            headless = True
            print(f"Environment {i}: HEADLESS")
        elif args.no_video:
            # All windowed, no video
            record_video = False
            headless = False
            print(f"Environment {i}: WINDOWED (no video)")
        else:
            # Hybrid mode (default)
            if i == 0:
                # First environment: with window for video recording
                record_video = True
                headless = False
                print(f"Environment {i}: WINDOWED (with video recording)")
            else:
                # Other environments: headless for speed
                record_video = False
                headless = True
                print(f"Environment {i}: HEADLESS")
        
        env_fns.append(create_env(f"PokemonRed-v{i}", headless=headless, record_video=record_video))
    
    # Create vectorized environment
    env = DummyVecEnv(env_fns)
    
    # Transpose image for CNN (channels first)
    env = VecTransposeImage(env)
    
    print(f"[2/4] Creating new PPO model...")
    
    # Create PPO model
    print(f"Using {device} device")
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        verbose=args.verbose,
        device=device,
        tensorboard_log=f"logs/run_{timestamp}"
    )
    
    print("Model created successfully!")
    print(f"Logging to logs/run_{timestamp}")
    
    # Create callback
    callback = TrainingCallback()
    
    print(f"\n[3/4] Starting training...")
    if not args.headless and not args.no_video:
        print("Training will run with 1 windowed environment for video recording!")
        print("Check the 'recordings' folder for the training video.")
    print("ANTI-SPAM penalties are now MUCH STRONGER to prevent button spam!")
    
    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print(f"\n[4/4] Saving model...")
    
    # Save the final model
    model_path = os.path.join(run_dir, "pokemon_final.zip")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training info
    info_path = os.path.join(run_dir, "training_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Total timesteps: {args.timesteps:,}\n")
        f.write(f"Environments: {args.envs}\n")
        f.write(f"Environment setup: {env_setup}\n")
        f.write(f"Video recording: {'Enabled' if video_recording else 'Disabled'}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Anti-spam penalties: STRONG\n")
    
    print("Training completed successfully!")
    if video_recording and not args.headless:
        print(f"Check recordings/training_PokemonRed-v0_{timestamp}.mp4 for the training video!")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()