"""
Script to visualize trained Pokemon Red RL agent
"""

import os
import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from pokemon_env import PokemonRedEnv, PokemonRedWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import cv2
from config import ROM_PATH


class ObsAdapter(gym.ObservationWrapper):
    """Adapt per-frame observation to match a target (H, W, C).
    Used to load legacy models expecting 84x84x1 when current env outputs 72x80x2.
    """
    def __init__(self, env, target_h, target_w, target_c):
        super().__init__(env)
        self.target_h = target_h
        self.target_w = target_w
        self.target_c = target_c
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(target_h, target_w, target_c), dtype=np.uint8
        )

    def observation(self, obs):
        # obs is (H, W, C_in). For legacy, pick first channel and resize to (84,84)
        if obs.ndim == 3:
            if obs.shape[2] > 1 and self.target_c == 1:
                obs = obs[:, :, 0]
            if obs.shape[0] != self.target_h or obs.shape[1] != self.target_w:
                obs = cv2.resize(obs, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            if obs.ndim == 2:
                obs = np.expand_dims(obs, axis=-1)
        return obs


def play_pokemon(model_path, rom_path=ROM_PATH, num_episodes=5, fps=30):
    """
    Run the trained agent and visualize gameplay
    
    Args:
        model_path: Path to trained model
        rom_path: Path to Pokemon Red ROM
        num_episodes: Number of episodes to play
        fps: Frames per second for visualization
    """
    print("="*70)
    print("POKEMON RED RL - GAMEPLAY VISUALIZATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"ROM: {rom_path}")
    print(f"Episodes: {num_episodes}")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        return
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        print(f"\nERROR: ROM file not found at {rom_path}")
        return
    
    # Load model first to detect expected observation shape
    print("Loading model...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    expected_shape = None
    if hasattr(model, "observation_space") and model.observation_space is not None:
        expected_shape = tuple(model.observation_space.shape)
        print(f"Model expects observation shape: {expected_shape}")

    legacy_adapter = False
    stack_frames = 4
    target_h, target_w, target_c = 72, 80, 2  # defaults for current env
    if expected_shape is not None:
        # expected_shape is (C, H, W) after transpose
        C, H, W = expected_shape
        if (H, W) == (84, 84) and C == 4:
            # Legacy model: expects 4x84x84 (1 channel, 4 stacked)
            legacy_adapter = True
            stack_frames = 4
            target_h, target_w, target_c = 84, 84, 1
            print("Using legacy adapter for 84x84x1 model")
        elif (H, W) == (72, 80):
            # New models: C is 8 when visited mask enabled with 4 stack
            stack_frames = max(1, C // 2)
            print(f"Using new model with {C} channels, {stack_frames} stack frames")
        else:
            print(f"Unknown model shape: {expected_shape}")

    # Build env consistent with the model's expected per-frame shape
    def _init():
        e = PokemonRedEnv(rom_path=rom_path, render_mode="human", headless=False)
        if legacy_adapter:
            e = ObsAdapter(e, target_h, target_w, target_c)
        e = PokemonRedWrapper(e, stack_frames=stack_frames)
        e = Monitor(e)
        return e

    vec_env = DummyVecEnv([_init])
    vec_env = VecTransposeImage(vec_env)
    
    # Calculate frame delay for FPS
    frame_delay = 1.0 / fps if fps > 0 else 0
    
    for episode in range(1, num_episodes + 1):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode}/{num_episodes}")
        print(f"{'='*70}")
        
        obs = vec_env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Convert numpy array to scalar integer
            action = int(action)
            actions = ["No-op", "Up", "Down", "Left", "Right", "A", "B", "Start", "Select"]
            
            # DEBUG: Print action and state info
            print(f"Step {steps}: Action {action} ({actions[action]})")
            
            # Take action
            obs, rewards, dones, infos = vec_env.step([action])
            done = bool(dones[0])
            reward = float(rewards[0])
            info = infos[0]
            
            episode_reward += reward
            steps += 1
            
            # Print info periodically
            if steps % 100 == 0:
                print(f"Steps: {steps:4d} | Reward: {episode_reward:7.2f} | "
                      f"Position: ({info['player_x']}, {info['player_y']}) | "
                      f"Map: {info['map_id']} | Party: {info['party_count']}")
            
            # Respect FPS
            time.sleep(frame_delay)
        
        print(f"\nEpisode {episode} completed!")
        print(f"Total steps: {steps}")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Final position: ({info['player_x']}, {info['player_y']})")
        print(f"Final map: {info['map_id']}")
        print(f"Party count: {info['party_count']}")
    
    vec_env.close()
    print(f"\n{'='*70}")
    print("GAMEPLAY COMPLETE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Play Pokemon Red with trained RL agent")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--rom", default=ROM_PATH, help="Path to Pokemon Red ROM")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--fps", type=int, default=30, help="FPS for visualization")
    
    args = parser.parse_args()
    
    play_pokemon(
        model_path=args.model,
        rom_path=args.rom,
        num_episodes=args.episodes,
        fps=args.fps
    )


if __name__ == "__main__":
    main()