"""
OpenAI Gymnasium environment for Pokemon Red - ANTI-LOOP VERSION
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import cv2
import os
from memory_reader import PokemonRedMemory
from config import *
import time
from datetime import datetime


class PokemonRedEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokemon Red - ANTI-LOOP VERSION
    
    Goal: Navigate Pallet Town and catch a Pokemon in the grass
    
    Observation Space: 
        - Downscaled grayscale game screen (72x80)
        - Optional visited mask for exploration tracking
    
    Action Space:
        0: No-op (do nothing)
        1: Up
        2: Down
        3: Left
        4: Right
        5: A button (interact/confirm)
        6: B button (cancel/run)
        [7,8: Start/Select - DISABLED when DISABLE_MENU_BUTTONS=True]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
    
    def __init__(self, rom_path=ROM_PATH, render_mode="rgb_array", headless=False, record_video=False, video_fps=60, env_id="env"):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode  # Always use rgb_array for vectorized envs
        self.headless = headless
        self.record_video = record_video
        self.video_fps = video_fps
        self.env_id = env_id
        self.video_writer = None
        self.video_frames = []
        
        # Initialize PyBoy emulator
        # PyBoy 2.x API: use 'null' instead of 'headless' for headless mode
        if headless:
            window_type = "null"  # Updated for PyBoy 2.x
        else:
            window_type = "SDL2"
            
        self.pyboy = PyBoy(
            self.rom_path,
            window=window_type
        )
        
        # Try to load save state if it exists
        save_state_path = self.rom_path.replace('.gb', '.gb.state')
        if os.path.exists(save_state_path):
            try:
                self.pyboy.load_state(open(save_state_path, 'rb'))
                print(f"Loaded save state from {save_state_path}")
            except Exception as e:
                print(f"Failed to load save state: {e}")
        else:
            print(f"No save state found at {save_state_path}, starting from beginning")
        
        self.screen = self.pyboy.screen
        self.memory = PokemonRedMemory(self.pyboy)
        
        # FIXED: Define action space based on menu button setting
        if DISABLE_MENU_BUTTONS:
            self.action_space = spaces.Discrete(7)  # 0-6: No-op, Up, Down, Left, Right, A, B
        else:
            self.action_space = spaces.Discrete(9)  # 0-8: All actions including Start, Select
        
        # Define observation space (72x80 grayscale + optional visited mask)
        # Pokemon screen is 144x160 (H x W); downsampled by 2 → 72x80
        channels = 1 + (1 if USE_VISITED_MASK else 0)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(72, 80, channels), dtype=np.uint8
        )
        
        # FIXED: Action mapping - properly handle disabled menu buttons
        if DISABLE_MENU_BUTTONS:
            self.action_map = {
                0: None,  # No-op
                1: "up",
                2: "down", 
                3: "left",
                4: "right",
                5: "a",
                6: "b"
                # Actions 7,8 (Start/Select) are completely removed from action space
            }
        else:
            self.action_map = {
                0: None,  # No-op
                1: "up",
                2: "down",
                3: "left", 
                4: "right",
                5: "a",
                6: "b",
                7: "start",
                8: "select"
            }
        
        # Episode tracking
        self.steps = 0
        self.episode_reward = 0
        
        # Game state tracking
        self.previous_party_count = 0
        self.previous_map_id = 0
        self.previous_position = (0, 0)
        self.last_action = 0
        self.entered_grass = False
        self.battle_started = False
        self.initial_hp = 0
        self.visited_states = set()
        self.seen_maps_this_episode = set()  # track map_ids seen this episode
        
        # NEW: Enhanced exploration tracking
        self.visited_positions = set()  # Track all visited positions
        self.position_history = []  # Recent position history for pattern detection
        self.action_patterns = []  # Track action patterns for loop detection
        self.last_significant_position = (0, 0)  # Last position that gave significant reward
        self.exploration_distance = 0  # Total distance explored this episode
        
        # NEW: Anti-spam tracking
        self.a_button_streak = 0  # Count consecutive A button presses
        self.b_button_streak = 0  # Count consecutive B button presses
        self.same_position_streak = 0  # Count steps in same position
        self.last_position = (0, 0)  # Track last position for streak counting
        
        # NEW: Anti-loop tracking
        self.position_visits = {}  # Track how many times each position was visited
        self.loop_detection_window = 20  # Window for loop detection
        self.exploration_momentum = 0  # Track continuous exploration
        
        # Collision detection
        self.stuck_counter = 0
        self.last_movement_position = (0, 0)
        self.movement_actions = [1, 2, 3, 4]  # Up, Down, Left, Right
        self.recent_actions = []  # Track recent actions for pattern detection
        self.no_op_streak = 0  # Count consecutive no-op actions
        # Action logging (for training visualization)
        self.action_counts = np.zeros(self.action_space.n, dtype=np.int64)
        self.action_log = []  # recent actions (bounded)
        # Visited masks (downsampled to 72x80)
        # We track one mask per map_id within an episode so exploration is map-aware
        self.visited_masks = {}
        self.visited_mask = np.zeros((72, 80), dtype=np.uint8)
        # For each map, store an origin (anchor) in native coordinates so that
        # the first time we enter a map, the player's current position is placed
        # near the center of the mask instead of top-left.
        self.visited_origins = {}
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset the emulator
        self.pyboy.stop()
        
        # Reinitialize PyBoy with correct window type
        if self.headless:
            window_type = "null"
        else:
            window_type = "SDL2"
            
        self.pyboy = PyBoy(
            self.rom_path,
            window=window_type
        )
        
        # Try to load save state if it exists
        save_state_path = self.rom_path.replace('.gb', '.gb.state')
        if os.path.exists(save_state_path):
            try:
                self.pyboy.load_state(open(save_state_path, 'rb'))
                print(f"Loaded save state from {save_state_path}")
            except Exception as e:
                print(f"Failed to load save state: {e}")
        else:
            print(f"No save state found at {save_state_path}, starting from beginning")
        self.screen = self.pyboy.screen
        self.memory = PokemonRedMemory(self.pyboy)
        
        # Skip intro and get to gameplay
        # This advances the game to a playable state
        # In a real implementation, you'd load a save state here
        for _ in range(100):  # Skip some initial frames
            self.pyboy.tick()
            
        # Wait for map to load before starting reward calculation
        # Check if we're in a valid game state (not main menu)
        # max_wait = 1000  # Maximum frames to wait
        # wait_count = 0
        # while wait_count < max_wait:
        #     current_map_id = self.memory.get_map_id()
        #     # Check if we have a valid map ID (not 0 or invalid)
        #     if current_map_id > 0:
        #         break
        #     self.pyboy.tick()
        #     wait_count += 1
        
        # Reset tracking variables
        self.steps = 0
        self.episode_reward = 0
        self.previous_party_count = self.memory.get_party_count()
        self.previous_map_id = self.memory.get_map_id()
        self.previous_position = self.memory.get_player_position()
        self.last_action = 0
        self.entered_grass = False
        self.battle_started = False
        self.initial_hp = 0
        self.visited_states = set()
        self.seen_maps_this_episode = set([self.previous_map_id])
        
        # Reset enhanced exploration tracking
        self.visited_positions = set([self.previous_position])
        self.position_history = [self.previous_position]
        self.action_patterns = []
        self.last_significant_position = self.previous_position
        self.exploration_distance = 0
        
        # Reset anti-spam tracking
        self.a_button_streak = 0
        self.b_button_streak = 0
        self.same_position_streak = 0
        self.last_position = self.previous_position
        
        # Reset anti-loop tracking
        self.position_visits = {}
        self.exploration_momentum = 0
        
        # Reset collision detection
        self.stuck_counter = 0
        self.last_movement_position = self.memory.get_player_position()
        self.recent_actions = []
        self.no_op_streak = 0
        self.action_counts[:] = 0
        self.action_log = []
        # Reset visited masks for new episode
        if USE_VISITED_MASK:
            self.visited_masks.clear()
            current_map_id = self.previous_map_id
            self.visited_masks[current_map_id] = np.zeros_like(self.visited_mask)
            # Center the current position in the mask by choosing an origin
            px, py = self.memory.get_player_position()
            self.visited_origins[current_map_id] = (px - (SCREEN_WIDTH - 1) / 2.0,
                                                    py - (SCREEN_HEIGHT - 1) / 2.0)
            self.visited_mask = self.visited_masks[current_map_id]
        
        observation = self._get_observation()
        info = self._get_info()
        
        # Start video recording if enabled
        if self.record_video:
            self._start_video_recording()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        
        # FIXED: Validate action is within current action space
        if action >= self.action_space.n:
            print(f"WARNING: Invalid action {action} for action space size {self.action_space.n}")
            action = 0  # Default to no-op
        
        # Store action before execution
        self.last_action = action
        
        # Track no-op streak
        if action == 0:
            self.no_op_streak += 1
        else:
            self.no_op_streak = 0
            
        # Track button spam streaks
        if action == 5:  # A button
            self.a_button_streak += 1
            self.b_button_streak = 0  # Reset B streak
        elif action == 6:  # B button
            self.b_button_streak += 1
            self.a_button_streak = 0  # Reset A streak
        else:
            # Reset both streaks for movement actions
            self.a_button_streak = 0
            self.b_button_streak = 0
        
        # Track recent actions for pattern detection
        self.recent_actions.append(action)
        if len(self.recent_actions) > 20:  # Keep only last 20 actions
            self.recent_actions.pop(0)

        # Record action for logging
        self.action_counts[action] += 1
        self.action_log.append(int(action))
        if len(self.action_log) > 5000:
            self.action_log.pop(0)
        
        # Execute action
        self._take_action(action)
        
        # Advance emulator
        for _ in range(SKIP_FRAMES):
            self.pyboy.tick()
        
        # Record frame if video recording is enabled
        if self.record_video:
            self._record_frame()
        
        # Get new state
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.steps >= MAX_STEPS_PER_EPISODE
        info = self._get_info()
        
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def _take_action(self, action):
        """Execute the given action in the emulator"""
        if action == 0:  # No-op
            return
        
        button = self.action_map[action]
        
        # If disabled action or no-op, do nothing
        if button is None:
            return
        
        # PyBoy 2.x API: use button_press and button_release methods
        # For movement actions (up, down, left, right), we need to press twice to actually move
        # First press changes facing direction, second press moves
        if button in ["up", "down", "left", "right"]:
            # First press - change facing direction
            self.pyboy.button_press(button)
            self.pyboy.tick()
            self.pyboy.button_release(button)
            self.pyboy.tick()
            
            # Second press - actually move
            self.pyboy.button_press(button)
            self.pyboy.tick()
            self.pyboy.button_release(button)
        else:
            # For other buttons (A, B, Start, Select), single press is enough
            self.pyboy.button_press(button)
            self.pyboy.tick()
            self.pyboy.button_release(button)
    
    def _get_observation(self):
        """Get the current screen as observation"""
        # Get screen as numpy array (RGB)
        screen_array = self.screen.ndarray
        
        # Convert to grayscale
        gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 80x72 (W x H for cv2)
        resized = cv2.resize(gray, (80, 72), interpolation=cv2.INTER_AREA)
        
        # Add channel(s)
        if USE_VISITED_MASK:
            # Update visited mask at current player location region
            # Use coarse info: mark center 5x5 region as visited
            px, py = self.memory.get_player_position()
            # Scale from native coordinates to mask size (80x72)
            # Guard against out-of-range memory reads by clamping
            px = max(0, min(px, SCREEN_WIDTH - 1))
            py = max(0, min(py, SCREEN_HEIGHT - 1))
            # Optional map-relative origin to center initial position
            ox, oy = 0.0, 0.0
            if VISITED_MASK_PER_MAP and hasattr(self, "visited_origins"):
                ox, oy = self.visited_origins.get(self.memory.get_map_id(), (0.0, 0.0))
            rx = px - ox
            ry = py - oy
            # Normalize relative coords to [0,1] over a window of SCREEN_WIDTH x SCREEN_HEIGHT
            rx = max(0.0, min(rx, SCREEN_WIDTH - 1))
            ry = max(0.0, min(ry, SCREEN_HEIGHT - 1))
            vx = int((rx / (SCREEN_WIDTH - 1)) * (80 - 1))
            vy = int((ry / (SCREEN_HEIGHT - 1)) * (72 - 1))
            x0, x1 = max(0, vx - 2), min(80, vx + 3)
            y0, y1 = max(0, vy - 2), min(72, vy + 3)
            self.visited_mask[y0:y1, x0:x1] = 255
            mask = self.visited_mask
            observation = np.stack([resized, mask], axis=-1)
        else:
            observation = np.expand_dims(resized, axis=-1)
        
        return observation
    
    def _calculate_reward(self):
        """Calculate reward based on game state with STRONG anti-loop logic"""
        # Only start calculating rewards when we're in a valid game state
        current_map_id = self.memory.get_map_id()
        if current_map_id <= 0:
            return 0.0  # No rewards during loading/main menu
            
        # Get current game state
        current_position = self.memory.get_player_position()
        current_party_count = self.memory.get_party_count()
        in_battle = self.memory.is_in_battle()
        in_grass = self.memory.in_grass_area()
        
        # Check if anything meaningful changed
        position_changed = current_position != self.previous_position
        party_changed = current_party_count != self.previous_party_count
        map_changed = current_map_id != self.previous_map_id
        battle_changed = in_battle != self.battle_started
        grass_changed = in_grass != self.entered_grass
        
        # If nothing meaningful changed, return minimal reward
        if not any([position_changed, party_changed, map_changed, battle_changed, grass_changed]):
            return 0.0  # No change, no reward
            
        reward = REWARDS["step"]  # Base reward per step
        
        # === ANTI-LOOP LOGIC (STRONG) ===
        # Track position visits for loop detection
        if current_position not in self.position_visits:
            self.position_visits[current_position] = 0
        self.position_visits[current_position] += 1
        
        # Strong penalty for returning to same position too often
        if self.position_visits[current_position] > 3:  # Visited this position 3+ times
            reward += REWARDS["anti_loop_penalty"]
            if self.position_visits[current_position] > 5:  # Very strong penalty for 5+ visits
                reward += REWARDS["loop_detection_penalty"]
        
        
        # Curriculum exploration - large bonus for new areas
        if current_position not in self.visited_positions:
            reward += REWARDS["curriculum_exploration"]
            self.visited_positions.add(current_position)
            self.exploration_momentum += 1
        else:
            # Reduce exploration momentum for returning to old areas
            self.exploration_momentum = max(0, self.exploration_momentum - 1)
        
        # Reward for exploration momentum
        if self.exploration_momentum > 0:
            reward += REWARDS["exploration_momentum"]
        
        # === ANTI-SPAM PENALTIES (BALANCED) ===
        # Track position changes for same-position penalty
        if current_position == self.last_position:
            self.same_position_streak += 1
        else:
            self.same_position_streak = 0
            self.last_position = current_position
        
        # Small penalty for staying in same position too long
        if self.same_position_streak > 5:  # 5+ steps in same position
            reward += REWARDS["same_position_streak"]
        
        # Moderate penalty if no movement for too long
        if self.same_position_streak > 20:  # 20+ steps in same position
            reward += REWARDS["movement_required"]
        
        # Small penalties for button spam
        if self.a_button_streak > 3:  # 3+ consecutive A presses
            reward += REWARDS["a_button_spam"]
            if self.a_button_streak > 5:  # Extra penalty for long streaks
                reward += REWARDS["button_spam_streak"]
        
        if self.b_button_streak > 3:  # 3+ consecutive B presses
            reward += REWARDS["b_button_spam"]
            if self.b_button_streak > 5:  # Extra penalty for long streaks
                reward += REWARDS["button_spam_streak"]
        
        # === MOVEMENT AND EXPLORATION REWARDS ===
        if position_changed:
            # Calculate distance traveled
            distance = ((current_position[0] - self.previous_position[0])**2 + 
                       (current_position[1] - self.previous_position[1])**2)**0.5
            
            # Basic movement reward
            reward += REWARDS["movement"]
            
            # Distance-based reward
            reward += REWARDS["distance_traveled"] * distance
            self.exploration_distance += distance
            
            # New position reward
            if current_position not in self.visited_positions:
                reward += REWARDS["new_position"]
                self.visited_positions.add(current_position)
            
            # Update position tracking
            self.previous_position = current_position
            self.position_history.append(current_position)
            if len(self.position_history) > 20:  # Keep only recent history
                self.position_history.pop(0)
            
            self.stuck_counter = 0  # Reset stuck counter
            
        else:
            # Player didn't move - could be hitting a wall or stuck
            if self.last_action in self.movement_actions:  # Only penalize movement actions
                reward += REWARDS["wall_hit"]  # Penalty for hitting wall
                self.stuck_counter += 1
            else:
                # Non-movement action, don't penalize
                self.stuck_counter = 0
                # Global no-move penalty (applies even if A/B/no-op)
                reward += REWARDS["no_move"]

        # === LOOP DETECTION AND BREAKING ===
        # Track action patterns
        self.action_patterns.append(self.last_action)
        if len(self.action_patterns) > 20:  # Keep only recent actions
            self.action_patterns.pop(0)
        
        # Detect repetitive patterns
        if len(self.action_patterns) >= 10:
            # Check for simple alternating patterns (A-B-A-B-A-B)
            recent_actions = self.action_patterns[-10:]
            if self._detect_repetitive_pattern(recent_actions):
                reward += REWARDS["pattern_penalty"]
                # Give bonus for breaking the pattern
                if len(set(recent_actions[-3:])) >= 2:  # Recent actions are diverse
                    reward += REWARDS["loop_break_bonus"]
        
        # Detect position loops (returning to same area repeatedly)
        if len(self.position_history) >= 10:
            if self._detect_position_loop():
                reward += REWARDS["pattern_penalty"]
        
        # === ACTION VARIETY REWARDS ===
        # Reward for trying different actions
        if len(self.recent_actions) >= 8:
            unique_actions = len(set(self.recent_actions[-8:]))
            if unique_actions >= 4:  # Good variety
                reward += REWARDS["variety_bonus"]
        
        # === EXPLORATION BONUSES ===
        # Large exploration bonus for significant movement
        if self.exploration_distance > 50:  # Significant exploration
            reward += REWARDS["exploration_bonus"]
            self.exploration_distance = 0  # Reset
        
        # === EXISTING REWARDS ===
        # Penalize no-op
        if self.last_action == 0:
            reward += REWARDS["no_op"]
        
        # Penalty for being stuck in same position for too long
        if self.stuck_counter > 10:  # Stuck for 10+ steps
            reward += REWARDS["stuck"]
            self.stuck_counter = 0  # Reset to avoid continuous penalty
        
        # Map transitions and first-time map visit bonus
        if map_changed:
            reward += REWARDS["map_transition"]
            if current_map_id not in self.seen_maps_this_episode:
                reward += REWARDS["first_time_on_map"]
                self.seen_maps_this_episode.add(current_map_id)
            self.previous_map_id = current_map_id
            # Switch to the visited mask for this map
            if USE_VISITED_MASK and VISITED_MASK_PER_MAP:
                if current_map_id not in self.visited_masks:
                    self.visited_masks[current_map_id] = np.zeros_like(self.visited_mask)
                    # New map: compute an origin so that current position is near center
                    px_now, py_now = self.memory.get_player_position()
                    self.visited_origins[current_map_id] = (
                        px_now - (SCREEN_WIDTH - 1) / 2.0,
                        px_now - (SCREEN_HEIGHT - 1) / 2.0,
                    )
                self.visited_mask = self.visited_masks[current_map_id]

        # Check if entered grass area (dense progress cue)
        if grass_changed and in_grass:
            reward += REWARDS["move_to_grass"]
            self.entered_grass = True
        
        # Check if battle started
        if battle_changed and in_battle:
            reward += REWARDS["start_battle"]
            self.battle_started = True
            self.initial_hp = self.memory.get_first_pokemon_hp()
        
        # Check if caught a Pokemon or obtained a starter outside battle
        if party_changed:
            # Distinguish between first acquisition (starter) and later catches
            if self.previous_party_count == 0 and not in_battle:
                reward += REWARDS["starter_obtained"]
            else:
                reward += REWARDS["catch_pokemon"]
            self.previous_party_count = current_party_count
        
        # Reward for exploration (visiting new states)
        state_hash = self.memory.get_game_state_hash()
        if state_hash not in self.visited_states:
            self.visited_states.add(state_hash)
            reward += 0.1  # Small exploration bonus
        
        return reward
    
    def _start_video_recording(self):
        """Start video recording"""
        if not self.record_video:
            return
            
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        
        # Generate timestamped filename with environment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"recordings/training_{self.env_id}_{timestamp}.mp4"
        
        # Get screen dimensions
        screen_array = self.screen.ndarray
        height, width = screen_array.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
        
        print(f"Started video recording for {self.env_id}: {video_path}")
    
    def _record_frame(self):
        """Record current frame to video"""
        if not self.record_video or self.video_writer is None:
            return
            
        # Get current screen
        screen_array = self.screen.ndarray
        
        # Convert RGB to BGR for OpenCV
        if len(screen_array.shape) == 3:
            frame = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
        else:
            frame = screen_array
            
        # Write frame to video
        self.video_writer.write(frame)
    
    def _stop_video_recording(self):
        """Stop video recording and save file"""
        if not self.record_video or self.video_writer is None:
            return
            
        self.video_writer.release()
        self.video_writer = None
        print("Video recording stopped and saved")
    
    def _detect_repetitive_pattern(self, actions):
        """Detect if the action sequence shows repetitive patterns"""
        if len(actions) < 6:
            return False
        
        # Check for simple alternating patterns (A-B-A-B-A-B)
        if len(actions) >= 6:
            pattern1 = actions[-6:-4]  # Last 2 actions
            pattern2 = actions[-4:-2]  # Previous 2 actions  
            pattern3 = actions[-2:]   # Current 2 actions
            
            if pattern1 == pattern2 == pattern3:
                return True
        
        # Check for longer repetitive sequences
        if len(actions) >= 8:
            # Check if last 4 actions repeat
            if actions[-4:] == actions[-8:-4]:
                return True
        
        return False
    
    def _detect_position_loop(self):
        """Detect if the agent is stuck in a position loop"""
        if len(self.position_history) < 8:
            return False
        
        # Check if we're returning to positions we've been to recently
        recent_positions = self.position_history[-8:]
        current_pos = recent_positions[-1]
        
        # Count how many times we've been near this position recently
        nearby_count = 0
        for pos in recent_positions[:-1]:  # Exclude current position
            distance = ((pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2)**0.5
            if distance < 3:  # Within 3 tiles
                nearby_count += 1
        
        return nearby_count >= 3  # Been near this position 3+ times recently
    
    def _is_terminated(self):
        """Check if episode should end (goal achieved)"""
        # Episode ends if we caught a Pokemon
        current_party_count = self.memory.get_party_count()
        if current_party_count > self.previous_party_count:
            return True
        return False
    
    def _get_info(self):
        """Get additional information about the current state"""
        x, y = self.memory.get_player_position()
        return {
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "player_x": x,
            "player_y": y,
            "map_id": self.memory.get_map_id(),
            "party_count": self.memory.get_party_count(),
            "in_battle": self.memory.is_in_battle(),
            "in_grass": self.memory.in_grass_area(),
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # PyBoy handles rendering automatically in SDL2 mode
            pass
        elif self.render_mode == "rgb_array":
            return self.screen.ndarray
    
    def close(self):
        """Clean up resources"""
        # Stop video recording if active
        if self.record_video:
            self._stop_video_recording()
        
        self.pyboy.stop()

    # ---- Debug / logging helpers ----
    def get_action_stats(self):
        """Return action counts and recent actions window for logging.
        Returns a dict with counts (list) and total steps counted.
        """
        counts = self.action_counts.tolist()
        total = int(sum(counts))
        return {"counts": counts, "total": total}


# Wrapper to handle preprocessing and frame stacking
class PokemonRedWrapper(gym.Wrapper):
    """
    Additional wrapper for preprocessing
    """
    def __init__(self, env, stack_frames=4):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = None
        
        # Update observation space for frame stacking
        h, w, c = self.observation_space.shape
        stacked_c = c * stack_frames
        low = np.repeat(self.observation_space.low, stack_frames, axis=-1)
        high = np.repeat(self.observation_space.high, stack_frames, axis=-1)
        # Ensure shapes are exactly (h, w, c * stack_frames)
        low = low[:, :, :stacked_c]
        high = high[:, :, :stacked_c]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames = [observation] * self.stack_frames
        return self._get_stacked_frames(), info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(observation)
        return self._get_stacked_frames(), reward, terminated, truncated, info
    
    def _get_stacked_frames(self):
        """Stack frames along the channel dimension"""
        return np.concatenate(self.frames, axis=-1)