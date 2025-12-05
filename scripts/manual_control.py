"""
Manual control script for Pokemon Red
Allows you to manually control the game and see frame/mask side-by-side
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokemon_env import PokemonRedEnv, PokemonRedWrapper
from config import ROM_PATH

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: 'keyboard' library not available. Install with: pip install keyboard")
    print("Falling back to cv2.waitKey (arrow keys may not work properly)")


class ManualController:
    """Manual game controller with visual feedback"""
    
    def __init__(self, rom_path, training_mode=None, zoom_mask=False):
        self.rom_path = rom_path
        self.training_mode = training_mode
        
        # Zoom settings
        self.zoom_mask = zoom_mask  # Start with zoom mode based on arg
        self.zoom_level = 6  # Higher zoom for mask (6x vs 3x default)
        
        # Key mappings
        self.keyboard_available = KEYBOARD_AVAILABLE
        self.current_keys = set()
        self.last_action = 0
        
        if self.keyboard_available:
            # Use keyboard library for reliable arrow key detection
            keyboard.on_press_key('up', lambda _: self.current_keys.add('up'))
            keyboard.on_release_key('up', lambda _: self.current_keys.discard('up'))
            keyboard.on_press_key('down', lambda _: self.current_keys.add('down'))
            keyboard.on_release_key('down', lambda _: self.current_keys.discard('down'))
            keyboard.on_press_key('left', lambda _: self.current_keys.add('left'))
            keyboard.on_release_key('left', lambda _: self.current_keys.discard('left'))
            keyboard.on_press_key('right', lambda _: self.current_keys.add('right'))
            keyboard.on_release_key('right', lambda _: self.current_keys.discard('right'))
            keyboard.on_press_key('enter', lambda _: self.current_keys.add('enter'))
            keyboard.on_release_key('enter', lambda _: self.current_keys.discard('enter'))
            keyboard.on_press_key('space', lambda _: self.current_keys.add('space'))
            keyboard.on_release_key('space', lambda _: self.current_keys.discard('space'))
            keyboard.on_press_key('z', lambda _: self._toggle_zoom())  # Z to toggle zoom
        
        # Fallback key mappings for cv2.waitKey
        self.cv2_key_to_action = {
            82: 1,  # Up arrow
            84: 2,  # Down arrow
            81: 3,  # Left arrow
            83: 4,  # Right arrow
            ord('\r'): 5,  # Enter
            13: 5,  # Enter (numeric)
            ord(' '): 6,  # Spacebar
        }
        
        # Action names for display
        self.action_names = {
            0: "No-op",
            1: "Up",
            2: "Down",
            3: "Left",
            4: "Right",
            5: "A",
            6: "B"
        }
        
        # Create environment
        self._create_env()
        
    def _toggle_zoom(self):
        """Toggle zoom mode for visited mask"""
        self.zoom_mask = not self.zoom_mask
        mode_str = "ZOOMED" if self.zoom_mask else "NORMAL"
        print(f"[Mask view: {mode_str}]")
    
    def _create_env(self):
        """Create the environment"""
        self.env = PokemonRedEnv(
            rom_path=self.rom_path,
            render_mode="rgb_array",
            headless=True,  # We'll render manually
            training_mode=self.training_mode
        )
        self.env = PokemonRedWrapper(self.env, stack_frames=4)
        
    def _get_keyboard_action(self):
        """Get action from keyboard input"""
        if self.keyboard_available:
            # Use keyboard library for reliable arrow key support
            if 'up' in self.current_keys:
                return 1
            elif 'down' in self.current_keys:
                return 2
            elif 'left' in self.current_keys:
                return 3
            elif 'right' in self.current_keys:
                return 4
            elif 'enter' in self.current_keys:
                return 5
            elif 'space' in self.current_keys:
                return 6
            
            # Check for quit
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                return -1
            
            return None
        else:
            # Fallback to cv2.waitKey (less reliable for arrow keys)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                return -1
            
            if key in self.cv2_key_to_action:
                return self.cv2_key_to_action[key]
            
            return None
    
    def _visualize_observation(self, obs, info, action=None, step=0):
        """Visualize observation with frame and mask side-by-side"""
        try:
            h, w = obs.shape[:2]
            
            # Extract frame and mask
            if obs.shape[2] >= 2:
                frame = obs[:, :, 0]  # First channel (frame)
                mask = obs[:, :, 1]   # Second channel (visited mask)
            else:
                frame = obs[:, :, 0]
                mask = np.zeros_like(frame)
            
            # Scale settings
            frame_scale = 3
            mask_scale = self.zoom_level if self.zoom_mask else 3
            
            # Resize frame
            frame_resized = cv2.resize(frame, (w * frame_scale, h * frame_scale), interpolation=cv2.INTER_NEAREST)
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)
            
            # Add text overlay with game info to frame
            info_text = [
                f"Step: {step}",
                f"Action: {self.action_names.get(action, 'Unknown')}",
                f"Position: ({info.get('player_x', 0)}, {info.get('player_y', 0)})",
                f"Map ID: {info.get('map_id', 0)}",
                f"Party: {info.get('party_count', 0)}",
                f"Battle: {'Yes' if info.get('in_battle', False) else 'No'}"
            ]
            y_offset = 20
            for i, text in enumerate(info_text):
                cv2.putText(frame_bgr, text, (10, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if self.zoom_mask:
                # ZOOMED MODE: Large mask with color coding and grid
                mask_resized = cv2.resize(mask, (w * mask_scale, h * mask_scale), interpolation=cv2.INTER_NEAREST)
                
                # Create colorized mask (heatmap style)
                mask_normalized = (mask_resized / 255.0 * 255).astype(np.uint8)
                mask_colored = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_HOT)
                
                # Add grid overlay for better position reading
                grid_color = (50, 50, 50)
                grid_spacing = mask_scale * 5  # Grid every 5 original pixels
                for x in range(0, mask_colored.shape[1], grid_spacing):
                    cv2.line(mask_colored, (x, 0), (x, mask_colored.shape[0]), grid_color, 1)
                for y in range(0, mask_colored.shape[0], grid_spacing):
                    cv2.line(mask_colored, (0, y), (mask_colored.shape[1], y), grid_color, 1)
                
                # Add zoom indicator and labels
                cv2.putText(mask_colored, f"ZOOMED MASK ({mask_scale}x)", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(mask_colored, "Press Z to toggle", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Add colorbar legend
                legend_x = mask_colored.shape[1] - 30
                legend_h = min(100, mask_colored.shape[0] - 80)
                for i in range(legend_h):
                    color_val = int(255 * i / legend_h)
                    color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_HOT)[0][0].tolist()
                    cv2.line(mask_colored, (legend_x, mask_colored.shape[0] - 60 - i), 
                            (legend_x + 20, mask_colored.shape[0] - 60 - i), color, 1)
                cv2.putText(mask_colored, "255", (legend_x - 5, mask_colored.shape[0] - 60 - legend_h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(mask_colored, "0", (legend_x + 5, mask_colored.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                mask_bgr = mask_colored
            else:
                # NORMAL MODE: Simple grayscale mask
                mask_resized = cv2.resize(mask, (w * mask_scale, h * mask_scale), interpolation=cv2.INTER_NEAREST)
                mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_bgr, "Visited Mask", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(mask_bgr, "Press Z to zoom", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Match heights for side-by-side display
            frame_h = frame_bgr.shape[0]
            mask_h = mask_bgr.shape[0]
            
            if frame_h != mask_h:
                # Resize mask to match frame height while maintaining aspect ratio
                aspect = mask_bgr.shape[1] / mask_bgr.shape[0]
                new_mask_w = int(frame_h * aspect)
                mask_bgr = cv2.resize(mask_bgr, (new_mask_w, frame_h), interpolation=cv2.INTER_NEAREST)
            
            # Concatenate side-by-side
            vis = np.concatenate([frame_bgr, mask_bgr], axis=1)
            
            # Add title bar at bottom
            zoom_status = "ZOOMED" if self.zoom_mask else "NORMAL"
            title = f"Frame (Left) | Visited Mask [{zoom_status}] (Right) | Z=Toggle Zoom"
            cv2.putText(vis, title, (10, vis.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return vis
            
        except Exception as e:
            print(f"Error visualizing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self, max_steps=10000):
        """Run manual control loop"""
        print("="*70)
        print("POKEMON RED - MANUAL CONTROL")
        print("="*70)
        print("Controls:")
        print("  Arrow Keys     - Move (Up/Down/Left/Right)")
        print("  ENTER          - A button")
        print("  SPACEBAR       - B button")
        print("  Z              - Toggle zoom on visited mask")
        print("  Q or ESC       - Quit")
        if not self.keyboard_available:
            print("  NOTE: Install 'keyboard' library for better arrow key support:")
            print("        pip install keyboard")
        zoom_status = "ON (heatmap)" if self.zoom_mask else "OFF (grayscale)"
        print(f"\nMask Zoom: {zoom_status}")
        print("="*70)
        
        # Reset environment
        obs, info = self.env.reset()
        print(f"\nEnvironment reset!")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Initial position: ({info['player_x']}, {info['player_y']})")
        print(f"  Map ID: {info['map_id']}")
        print(f"  Party count: {info['party_count']}")
        print("\nStarting manual control...\n")
        
        step = 0
        total_reward = 0
        done = False
        
        try:
            while not done and step < max_steps:
                # Get keyboard input
                action = self._get_keyboard_action()
                
                # Check for quit
                if action == -1:
                    print("\nQuitting...")
                    break
                
                # Use last action if no new key pressed (hold key behavior)
                if action is None:
                    action = self.last_action
                else:
                    self.last_action = action
                
                # Take action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                
                # Visualize
                vis = self._visualize_observation(obs, info, action, step)
                if vis is not None:
                    cv2.imshow("Pokemon Red - Manual Control", vis)
                    cv2.waitKey(1)
                
                # Print progress every 50 steps
                if step % 50 == 0:
                    print(f"Step {step:4d} | Action: {self.action_names[action]:6s} | "
                          f"Reward: {total_reward:7.2f} | "
                          f"Pos: ({info['player_x']:3d}, {info['player_y']:3d}) | "
                          f"Map: {info['map_id']:3d} | "
                          f"Party: {info['party_count']}")
                
                # Reset if episode ends
                if done:
                    print(f"\nEpisode ended at step {step}!")
                    print(f"Total reward: {total_reward:.2f}")
                    print("Resetting environment...")
                    obs, info = self.env.reset()
                    total_reward = 0
                    done = False
                    time.sleep(0.5)  # Brief pause before reset
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.close()
    
    def close(self):
        """Close environment and windows"""
        self.env.close()
        cv2.destroyAllWindows()
        print("\nEnvironment closed.")


def main():
    parser = argparse.ArgumentParser(description="Manually control Pokemon Red with visual feedback")
    parser.add_argument("--sequence", type=int, choices=[1, 2, 3], help="Sequence number (1=house_exit, 2=exploration, 3=battle)")
    parser.add_argument("--rom", help="Path to ROM (overrides --sequence)")
    parser.add_argument("--steps", type=int, default=10000, help="Max steps to run")
    parser.add_argument("--zoom", action="store_true", help="Start with zoomed mask view (heatmap mode)")
    parser.add_argument("--zoom-level", type=int, default=6, help="Zoom level for mask (default: 6)")
    
    args = parser.parse_args()
    
    # Determine ROM path and training mode
    rom_path = None
    training_mode = None
    
    if args.sequence:
        if args.sequence == 1:
            from config import config_sequence1_house_exit as seq_config
            rom_path = seq_config.ROM_PATH
            training_mode = "house_exit"
            print(f"Using Sequence 1 (House Exit) - ROM: {rom_path}")
        elif args.sequence == 2:
            from config import config_sequence2_exploration as seq_config
            rom_path = seq_config.ROM_PATH
            training_mode = "exploration"
            print(f"Using Sequence 2 (Exploration) - ROM: {rom_path}")
        elif args.sequence == 3:
            from config import config_sequence3_battle as seq_config
            rom_path = seq_config.ROM_PATH
            training_mode = "battle"
            print(f"Using Sequence 3 (Battle) - ROM: {rom_path}")
    
    if args.rom:
        rom_path = args.rom
        print(f"Using explicit ROM path: {rom_path}")
    
    if not rom_path:
        rom_path = ROM_PATH
        print(f"Using default ROM path: {rom_path}")
    
    # Create controller and run
    controller = ManualController(rom_path, training_mode=training_mode, zoom_mask=args.zoom)
    controller.zoom_level = args.zoom_level  # Apply custom zoom level
    controller.run(max_steps=args.steps)


if __name__ == "__main__":
    main()

