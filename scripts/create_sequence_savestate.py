"""
Create save states for different training sequences
Usage: python create_sequence_savestate.py <sequence_number>
  sequence_number: 1 (house_exit), 2 (exploration), or 3 (battle)
"""

import sys
import time
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboy import PyBoy
from memory_reader import PokemonRedMemory

# Sequence configurations
SEQUENCES = {
    1: {
        "name": "house_exit",
        "rom_folder": "roms/sequence1_house_exit",
        "description": "Learn to exit Red's house",
        "instructions": [
            "1. Navigate through the intro (press A to skip dialogue)",
            "2. Choose your name",
            "3. Get to the point where you're inside Red's house (upstairs)",
            "4. Make sure you can move around freely",
            "5. Press Ctrl+C when ready to save"
        ]
    },
    2: {
        "name": "exploration",
        "rom_folder": "roms/sequence2_exploration",
        "description": "Learn to explore and reach grass",
        "instructions": [
            "1. Navigate through the intro (press A to skip dialogue)",
            "2. Choose your name",
            "3. Exit Red's house and get to Pallet Town (outside)",
            "4. Make sure you can move around freely in Pallet Town",
            "5. Press Ctrl+C when ready to save"
        ]
    },
    3: {
        "name": "battle",
        "rom_folder": "roms/sequence3_battle",
        "description": "Learn battle mechanics",
        "instructions": [
            "1. Navigate through the intro (press A to skip dialogue)",
            "2. Choose your name",
            "3. Exit Red's house",
            "4. Go to grass area and trigger a battle with Blue",
            "5. Make sure you're in battle (not in menu, in actual battle screen)",
            "6. Press Ctrl+C when ready to save"
        ]
    }
}

def setup_rom_folder(sequence_num):
    """Copy ROM to sequence folder if it doesn't exist"""
    seq_config = SEQUENCES[sequence_num]
    rom_folder = seq_config["rom_folder"]
    
    # Create folder if it doesn't exist
    os.makedirs(rom_folder, exist_ok=True)
    
    # Copy ROM if it doesn't exist in sequence folder
    source_rom = "roms/PokemonRed.gb"
    target_rom = os.path.join(rom_folder, "PokemonRed.gb")
    
    if not os.path.exists(target_rom):
        if os.path.exists(source_rom):
            print(f"Copying ROM to {rom_folder}...")
            shutil.copy2(source_rom, target_rom)
            print(f"ROM copied successfully!")
        else:
            print(f"ERROR: Source ROM not found at {source_rom}")
            print("Please make sure PokemonRed.gb exists in roms/ folder")
            sys.exit(1)
    else:
        print(f"ROM already exists in {rom_folder}")
    
    return target_rom

def create_save_state(sequence_num):
    """Create a save state for the specified sequence"""
    if sequence_num not in SEQUENCES:
        print(f"ERROR: Invalid sequence number {sequence_num}")
        print("Valid sequences: 1 (house_exit), 2 (exploration), 3 (battle)")
        sys.exit(1)
    
    seq_config = SEQUENCES[sequence_num]
    
    print(f"\n{'='*60}")
    print(f"Creating save state for Sequence {sequence_num}: {seq_config['description']}")
    print(f"{'='*60}\n")
    
    # Setup ROM folder
    rom_path = setup_rom_folder(sequence_num)
    
    # Print instructions
    print("Instructions:")
    for instruction in seq_config["instructions"]:
        print(f"  {instruction}")
    print()
    
    # Launch PyBoy
    print("Launching Pokemon Red...")
    pyboy = PyBoy(rom_path, window="SDL2")
    pyboy.set_emulation_speed(1.0)  # Normal speed
    
    memory = PokemonRedMemory(pyboy)
    
    try:
        # Let user navigate manually
        print("\nGame is running. Navigate to the desired state...")
        print("Press Ctrl+C when ready to save.\n")
        
        last_info = None
        while True:
            pyboy.tick()
            time.sleep(1/60)  # 60 FPS
            
            # Print game state info every second (only if changed)
            try:
                x, y = memory.get_player_position()
                map_id = memory.get_map_id()
                party_count = memory.get_party_count()
                in_battle = memory.is_in_battle()
                
                current_info = (x, y, map_id, party_count, in_battle)
                if current_info != last_info:
                    print(f"Position: ({x}, {y}) | Map: {map_id} | Party: {party_count} | Battle: {in_battle}")
                    last_info = current_info
            except:
                pass  # Ignore errors during loading
                
    except KeyboardInterrupt:
        print("\n\nSaving state...")
        
        # Save the current state
        save_state_path = rom_path.replace('.gb', '.gb.state')
        with open(save_state_path, 'wb') as f:
            pyboy.save_state(f)
        
        print(f"\n✓ Save state created at: {save_state_path}")
        
        # Verify game state
        try:
            x, y = memory.get_player_position()
            map_id = memory.get_map_id()
            party_count = memory.get_party_count()
            in_battle = memory.is_in_battle()
            
            print(f"\nGame State Verification:")
            print(f"  Position: ({x}, {y})")
            print(f"  Map ID: {map_id}")
            print(f"  Party Count: {party_count}")
            print(f"  In Battle: {in_battle}")
            print(f"\n✓ Save state is ready for training!")
        except Exception as e:
            print(f"Warning: Could not verify game state: {e}")
        
    finally:
        pyboy.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_sequence_savestate.py <sequence_number>")
        print("  sequence_number: 1 (house_exit), 2 (exploration), or 3 (battle)")
        sys.exit(1)
    
    try:
        sequence_num = int(sys.argv[1])
        create_save_state(sequence_num)
    except ValueError:
        print(f"ERROR: Invalid sequence number '{sys.argv[1]}'")
        print("Please provide a number: 1, 2, or 3")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

