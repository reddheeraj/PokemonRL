"""
ROM Tester - Test different ROM files to find one that works
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyboy import PyBoy
from config import ROM_PATH


def test_rom(rom_path):
    """Test if a ROM file works with PyBoy"""
    print(f"\n{'='*60}")
    print(f"Testing ROM: {rom_path}")
    print(f"{'='*60}")
    
    try:
        # Check if file exists
        if not os.path.exists(rom_path):
            print(f"✗ File does not exist: {rom_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(rom_path)
        print(f"File size: {file_size:,} bytes")
        
        if file_size < 100000:  # Less than 100KB
            print("⚠ File seems too small")
            return False
        
        # Try to initialize PyBoy
        print("Attempting to initialize PyBoy...")
        pyboy = PyBoy(rom_path, window="null")  # Use null window for testing
        
        print("✓ PyBoy initialized successfully!")
        
        # Try to advance a few frames
        print("Testing game advancement...")
        for i in range(10):
            pyboy.tick()
        
        print("✓ Game advancement works!")
        
        # Clean up
        pyboy.stop()
        print("✓ ROM test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ ROM test failed: {e}")
        return False


def main():
    """Test all available ROM files"""
    print("POKEMON RED ROM TESTER")
    print("="*60)
    
    # List of ROM files to test
    rom_files = [
        "roms/PokemonRed.gb",
        "roms/PokemonRed2.gb",
        # Add more if you have them
    ]
    
    working_roms = []
    
    for rom_path in rom_files:
        if test_rom(rom_path):
            working_roms.append(rom_path)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    if working_roms:
        print(f"✓ Found {len(working_roms)} working ROM(s):")
        for rom in working_roms:
            print(f"  - {rom}")
        
        print(f"\nRecommendation: Use {working_roms[0]} for training")
        
        # Update config if needed
        if working_roms[0] != ROM_PATH:
            print(f"\nTo use {working_roms[0]}, update your config.py:")
            print(f"ROM_PATH = \"{working_roms[0]}\"")
    else:
        print("✗ No working ROMs found!")
        print("\nTroubleshooting:")
        print("1. Make sure you have a valid Pokemon Red ROM")
        print("2. Try downloading a fresh ROM file")
        print("3. Ensure the ROM is in .gb format (not .gbc or .zip)")
        print("4. Check that the ROM is not corrupted")


if __name__ == "__main__":
    main()


