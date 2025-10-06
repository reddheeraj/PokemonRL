"""
Helper script to create a save state at the start of the game
This allows faster training by skipping the intro sequence
"""

import os
import argparse
from pyboy import PyBoy
from config import ROM_PATH


def create_savestate(rom_path=ROM_PATH, save_name="start_state.state"):
    """
    Run the game and create a save state
    
    This script helps you manually play to a desired starting point
    and save the state for training.
    """
    print("="*70)
    print("POKEMON RED - SAVE STATE CREATOR")
    print("="*70)
    print("\nInstructions:")
    print("1. The game will start")
    print("2. Play until you reach the desired starting point")
    print("3. Press ESC to save the state and exit")
    print("\nFor this project, recommended starting point:")
    print("- Right outside Professor Oak's lab in Pallet Town")
    print("- With a starter Pokemon already chosen")
    print("- Ready to walk north to Route 1")
    print("="*70)
    print("\nStarting game...")
    
    # Check ROM exists
    if not os.path.exists(rom_path):
        print(f"\nERROR: ROM not found at {rom_path}")
        return
    
    # Create saves directory
    os.makedirs("saves", exist_ok=True)
    save_path = os.path.join("saves", save_name)
    
    # Start emulator
    pyboy = PyBoy(rom_path, window="SDL2")
    
    print(f"\nGame running! Play to desired position, then close window to save.")
    print("Controls:")
    print("  Arrow Keys: D-Pad")
    print("  Z: A button")
    print("  X: B button")
    print("  Enter: Start")
    print("  Backspace: Select")
    
    try:
        # Run until window is closed
        while pyboy.tick():
            pass
    except KeyboardInterrupt:
        print("\nInterrupted!")
    
    # Save state
    with open(save_path, "wb") as f:
        pyboy.save_state(f)
    
    print(f"\n{'='*70}")
    print(f"Save state created: {save_path}")
    print(f"You can now use this for training!")
    print(f"{'='*70}")
    
    pyboy.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Create a save state for Pokemon Red training"
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=ROM_PATH,
        help="Path to Pokemon Red ROM"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="start_state.state",
        help="Name for save state file"
    )
    
    args = parser.parse_args()
    create_savestate(args.rom, args.output)


if __name__ == "__main__":
    main()

