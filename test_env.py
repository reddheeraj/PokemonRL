"""
Test script to verify the environment is working correctly
"""

import argparse
from pokemon_env import PokemonRedEnv, PokemonRedWrapper
import cv2
import numpy as np
from config import ROM_PATH


def test_environment(rom_path=ROM_PATH, num_steps=1000):
    """
    Test the Pokemon Red environment with random actions
    
    Args:
        rom_path: Path to Pokemon Red ROM
        num_steps: Number of random steps to take
    """
    print("="*70)
    print("POKEMON RED ENVIRONMENT TEST")
    print("="*70)
    print(f"ROM: {rom_path}")
    print(f"Test Steps: {num_steps}")
    print("="*70)
    
    # Create environment
    print("\n[1/3] Creating environment...")
    try:
        env = PokemonRedEnv(rom_path=rom_path, render_mode="human", headless=False)
        env = PokemonRedWrapper(env, stack_frames=4)
        print("[OK] Environment created successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        return
    
    # Reset environment
    print("\n[2/3] Resetting environment...")
    try:
        obs, info = env.reset()
        print("[OK] Environment reset successfully!")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial info: {info}")
        # ---- Live map & mask preview (testing only) ----
        # When visited mask is enabled, visualize side-by-side with the frame
        try:
            h, w = obs.shape[:2]
            if obs.shape[2] >= 2:
                frame = obs[:, :, 0]
                mask = obs[:, :, 1]
                vis = np.concatenate([
                    cv2.resize(frame, (w*3, h*3), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(mask, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
                ], axis=1)
                cv2.imshow("Frame (left) | Visited Mask (right)", vis)
                cv2.waitKey(1)
        except Exception:
            pass
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        env.close()
        return
    
    # Test random actions
    print(f"\n[3/3] Testing with {num_steps} random actions...")
    print("\nTaking random actions (watch the game window)...")
    print("-" * 70)
    
    try:
        total_reward = 0
        for step in range(num_steps):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # Live update: show frame and visited mask side-by-side
            try:
                h, w = obs.shape[:2]
                if obs.shape[2] >= 2:
                    frame = obs[:, :, 0]
                    mask = obs[:, :, 1]
                else:
                    frame = obs[:, :, 0]
                    mask = np.zeros_like(frame)
                vis = np.concatenate([
                    cv2.resize(frame, (w*3, h*3), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(mask, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
                ], axis=1)
                cv2.imshow("Frame (left) | Visited Mask (right)", vis)
                cv2.waitKey(1)
            except Exception:
                pass
            
            # Print progress every 100 steps with tile information
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1:4d}/{num_steps} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Pos: ({info['player_x']:3d}, {info['player_y']:3d}) | "
                      f"Map: {info['map_id']:3d} | "
                      f"Party: {info['party_count']} | "
                      f"Tile: {info['tile_name']}")
            
            # Reset if episode ends
            if terminated or truncated:
                print(f"\n  Episode ended at step {step + 1}!")
                print(f"  Total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
        
        print("-" * 70)
        print("\n[OK] Test completed successfully!")
        print("\nEnvironment is working correctly!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during testing: {e}")
    finally:
        env.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\nEnvironment closed.")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Test Pokemon Red RL environment"
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=ROM_PATH,
        help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of test steps"
    )
    
    args = parser.parse_args()
    test_environment(args.rom, args.steps)


if __name__ == "__main__":
    main()