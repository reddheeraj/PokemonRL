"""
Play/inference script to watch trained agent play Pokemon Red
"""

import os
import argparse
import time
from stable_baselines3 import PPO
from pokemon_env import PokemonRedEnv, PokemonRedWrapper
from config import *


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
    
    # Create environment (with rendering)
    print("\nCreating environment...")
    env = PokemonRedEnv(rom_path=rom_path, render_mode="human", headless=False)
    env = PokemonRedWrapper(env, stack_frames=4)
    
    # Load model
    print(f"Loading model...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    frame_delay = 1.0 / fps
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Convert numpy array to scalar integer
            action = int(action)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print info periodically
            if steps % 100 == 0:
                print(f"Steps: {steps:4d} | Reward: {episode_reward:7.2f} | "
                      f"Position: ({info['player_x']}, {info['player_y']}) | "
                      f"Map: {info['map_id']} | Party: {info['party_count']}")
            
            # Respect FPS
            time.sleep(frame_delay)
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1} finished!")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Final Party Count: {info['party_count']}")
        print(f"{'='*70}")
        
        # Wait a bit between episodes
        time.sleep(2)
    
    env.close()
    print("\nPlayback complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Watch trained RL agent play Pokemon Red"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file (.zip)"
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=ROM_PATH,
        help="Path to Pokemon Red ROM file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for visualization"
    )
    
    args = parser.parse_args()
    
    play_pokemon(
        model_path=args.model,
        rom_path=args.rom,
        num_episodes=args.episodes,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

