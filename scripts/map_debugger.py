"""
Manual Map Debugger for Pokemon Red
Allows you to manually control the character and explore the game world
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pyboy import PyBoy
from memory_reader import PokemonRedMemory
from config import ROM_PATH
import time


class MapDebugger:
    """Manual map debugger for exploring Pokemon Red"""
    
    def __init__(self, rom_path=ROM_PATH):
        self.rom_path = rom_path
        self.pyboy = None
        self.memory = None
        self.running = True
        
        # Track previous state to detect changes
        self.last_position = (0, 0)
        self.last_map_id = 0
        self.last_party_count = 0
        self.last_battle_status = False
        
        # Initialize PyBoy
        self._init_pyboy()
        
        # Key mappings for manual control
        self.key_mappings = {
            ord('w'): 'up',
            ord('s'): 'down', 
            ord('a'): 'left',
            ord('d'): 'right',
            ord('j'): 'a',  # A button
            ord('k'): 'b',  # B button
            ord('i'): 'start',  # Start button
            ord('o'): 'select',  # Select button
        }
        
        # Display settings
        self.show_position_info = True
        self.show_map_info = True
        self.show_party_info = True
        
    def _init_pyboy(self):
        """Initialize PyBoy emulator without save state loading"""
        try:
            print(f"Loading ROM: {self.rom_path}")
            
            # Check if ROM file exists
            if not os.path.exists(self.rom_path):
                print(f"✗ ROM file not found: {self.rom_path}")
                print("Please make sure the ROM file exists in the roms/ directory")
                raise FileNotFoundError(f"ROM file not found: {self.rom_path}")
            
            # Check ROM file size
            rom_size = os.path.getsize(self.rom_path)
            print(f"ROM file size: {rom_size:,} bytes")
            
            if rom_size < 100000:  # Less than 100KB is suspicious
                print("⚠ ROM file seems too small, might be corrupted")
            
            # Initialize PyBoy without save state
            self.pyboy = PyBoy(self.rom_path, window="SDL2")
            
            # Skip the save state loading to avoid "No data" error
            print("✓ PyBoy initialized successfully!")
            print("Note: Starting from beginning (no save state loaded)")
            
            self.memory = PokemonRedMemory(self.pyboy)
            
            # Wait a bit for the game to load
            print("Waiting for game to load...")
            for _ in range(200):  # Wait 200 frames
                self.pyboy.tick()
            
            print("✓ Game loaded!")
            
        except Exception as e:
            print(f"✗ Failed to initialize PyBoy: {e}")
            print("\nTroubleshooting:")
            print("1. Check if ROM file exists and is valid")
            print("2. Try a different Pokemon Red ROM")
            print("3. Make sure ROM is in .gb format")
            raise
    
    def _get_game_info(self):
        """Get current game state information"""
        try:
            x, y = self.memory.get_player_position()
            map_id = self.memory.get_map_id()
            party_count = self.memory.get_party_count()
            in_battle = self.memory.is_in_battle()
            in_grass = self.memory.in_grass_area()
            
            return {
                'position': (x, y),
                'map_id': map_id,
                'party_count': party_count,
                'in_battle': in_battle,
                'in_grass': in_grass
            }
        except Exception as e:
            print(f"Error getting game info: {e}")
            return None
    
    def _has_state_changed(self, game_info):
        """Check if game state has changed significantly"""
        if not game_info:
            return False
            
        # Check for significant changes
        position_changed = game_info['position'] != self.last_position
        map_changed = game_info['map_id'] != self.last_map_id
        party_changed = game_info['party_count'] != self.last_party_count
        battle_changed = game_info['in_battle'] != self.last_battle_status
        
        return any([position_changed, map_changed, party_changed, battle_changed])
    
    def _update_last_state(self, game_info):
        """Update the last known state"""
        if not game_info:
            return
            
        self.last_position = game_info['position']
        self.last_map_id = game_info['map_id']
        self.last_party_count = game_info['party_count']
        self.last_battle_status = game_info['in_battle']
    
    def _display_info(self, game_info):
        """Display game information on screen only when state changes"""
        if not game_info:
            return
            
        # Create info overlay
        info_text = []
        
        if self.show_position_info:
            info_text.append(f"Position: ({game_info['position'][0]}, {game_info['position'][1]})")
        
        if self.show_map_info:
            info_text.append(f"Map ID: {game_info['map_id']}")
        
        if self.show_party_info:
            info_text.append(f"Party: {game_info['party_count']} Pokemon")
            info_text.append(f"Battle: {'Yes' if game_info['in_battle'] else 'No'}")
            info_text.append(f"Grass: {'Yes' if game_info['in_grass'] else 'No'}")
        
        # Print to console only when state changes
        print(f"\r{'='*50}")
        for line in info_text:
            print(f"  {line}")
        print(f"{'='*50}")
    
    def _handle_input(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            self.running = False
            return True
        elif key == ord('p'):
            # Toggle position info
            self.show_position_info = not self.show_position_info
            print(f"Position info: {'ON' if self.show_position_info else 'OFF'}")
        elif key == ord('m'):
            # Toggle map info
            self.show_map_info = not self.show_map_info
            print(f"Map info: {'ON' if self.show_map_info else 'OFF'}")
        elif key == ord('b'):
            # Toggle party info
            self.show_party_info = not self.show_party_info
            print(f"Party info: {'ON' if self.show_party_info else 'OFF'}")
        elif key == ord('h'):
            # Show help
            self._show_help()
        elif key in self.key_mappings:
            # Handle game controls
            button = self.key_mappings[key]
            self._press_button(button)
            return True
        
        return False
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "="*70)
        print("POKEMON RED MAP DEBUGGER - HELP")
        print("="*70)
        print("Controls:")
        print("  WASD - Move (W=Up, S=Down, A=Left, D=Right)")
        print("  J - A button, K - B button")
        print("  I - Start, O - Select")
        print("  P - Toggle position info")
        print("  M - Toggle map info")
        print("  B - Toggle party info")
        print("  H - Show this help")
        print("  Q - Quit")
        print("="*70)
        print("Info is only displayed when game state changes!")
        print("="*70)
    
    def _press_button(self, button):
        """Press a button in the game"""
        try:
            if button in ["up", "down", "left", "right"]:
                # For movement, press twice to actually move
                self.pyboy.button_press(button)
                self.pyboy.tick()
                self.pyboy.button_release(button)
                self.pyboy.tick()
                
                # Second press to actually move
                self.pyboy.button_press(button)
                self.pyboy.tick()
                self.pyboy.button_release(button)
            else:
                # For other buttons, single press
                self.pyboy.button_press(button)
                self.pyboy.tick()
                self.pyboy.button_release(button)
            
            print(f"Pressed: {button}")
            
        except Exception as e:
            print(f"Error pressing button {button}: {e}")
    
    def _advance_frames(self, num_frames=4):
        """Advance the game by a few frames"""
        for _ in range(num_frames):
            self.pyboy.tick()
    
    def run(self):
        """Main debugger loop"""
        print("="*70)
        print("POKEMON RED MAP DEBUGGER")
        print("="*70)
        print("Controls:")
        print("  WASD - Move (W=Up, S=Down, A=Left, D=Right)")
        print("  J - A button, K - B button")
        print("  I - Start, O - Select")
        print("  P - Toggle position info")
        print("  M - Toggle map info")
        print("  B - Toggle party info")
        print("  H - Show help")
        print("  Q - Quit")
        print("="*70)
        print("Info is only displayed when game state changes!")
        print("="*70)
        
        # Initialize last state
        initial_info = self._get_game_info()
        if initial_info:
            self._update_last_state(initial_info)
            self._display_info(initial_info)
        
        try:
            while self.running:
                # Get current game state
                game_info = self._get_game_info()
                
                # Only display info if state has changed
                if game_info and self._has_state_changed(game_info):
                    self._display_info(game_info)
                    self._update_last_state(game_info)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    self._handle_input(key)
                
                # Advance game frames
                self._advance_frames(4)
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nDebugger interrupted by user")
        except Exception as e:
            print(f"\n\nError in debugger: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.pyboy:
            self.pyboy.stop()
        cv2.destroyAllWindows()
        print("\nDebugger closed.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pokemon Red Map Debugger")
    parser.add_argument("--rom", type=str, default=ROM_PATH, help="Path to Pokemon Red ROM")
    
    args = parser.parse_args()
    
    try:
        debugger = MapDebugger(args.rom)
        debugger.run()
    except Exception as e:
        print(f"Failed to start debugger: {e}")


if __name__ == "__main__":
    main()