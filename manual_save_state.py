"""
Launch a fresh instance of Pokemon Red for manual save state creation
"""

import time
from pyboy import PyBoy
from config import ROM_PATH

def manual_save_state():
    """Launch fresh game for manual save state creation"""
    print("Launching fresh Pokemon Red game...")
    print("Please follow these steps:")
    print("1. Navigate through the intro (press A to skip dialogue)")
    print("2. Get to Pallet Town (the starting area)")
    print("3. Make sure you can move around freely")
    print("4. Press Ctrl+C when you're ready to save")
    
    # Start PyBoy with visual window
    pyboy = PyBoy(ROM_PATH, window="SDL2")
    
    # Set emulation speed to 60 FPS (1.0 = normal speed)
    pyboy.set_emulation_speed(1.5)
    
    try:
        # Let the user navigate manually at 60 FPS
        while True:
            pyboy.tick()
            time.sleep(1/60)  # 60 FPS
    except KeyboardInterrupt:
        print("\nSaving state...")
        
        # Save the current state
        save_state_path = ROM_PATH.replace('.gb', '.gb.state')
        with open(save_state_path, 'wb') as f:
            pyboy.save_state(f)
        
        print(f"Save state created at {save_state_path}")
        print("You can now use this save state for training!")
        
        # Get current game state for verification
        from memory_reader import PokemonRedMemory
        memory = PokemonRedMemory(pyboy)
        
        print(f"Current position: {memory.get_player_position()}")
        print(f"Current map ID: {memory.get_map_id()}")
        print(f"Party count: {memory.get_party_count()}")
        print(f"In battle: {memory.is_in_battle()}")
        
    finally:
        pyboy.stop()

if __name__ == "__main__":
    manual_save_state()
