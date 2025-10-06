"""
OpenAI Gymnasium environment for Pokemon Red
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import cv2
from memory_reader import PokemonRedMemory
from config import *


class PokemonRedEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokemon Red
    
    Goal: Navigate Pallet Town and catch a Pokemon in the grass
    
    Observation Space: 
        - Downscaled grayscale game screen (84x84)
    
    Action Space:
        0: No-op (do nothing)
        1: Up
        2: Down
        3: Left
        4: Right
        5: A button (interact/confirm)
        6: B button (cancel/run)
        7: Start (menu)
        8: Select
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}
    
    def __init__(self, rom_path=ROM_PATH, render_mode="human", headless=False):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.headless = headless
        
        # Initialize PyBoy emulator
        # PyBoy 2.x API: use 'window' instead of 'window_type'
        self.pyboy = PyBoy(
            self.rom_path,
            window="headless" if headless else "SDL2"
        )
        
        self.screen = self.pyboy.screen
        self.memory = PokemonRedMemory(self.pyboy)
        
        # Define action space
        # 0: No-op, 1: Up, 2: Down, 3: Left, 4: Right, 5: A, 6: B, 7: Start, 8: Select
        self.action_space = spaces.Discrete(9)
        
        # Define observation space (84x84 grayscale image)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        
        # Action mapping
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
        
        # Collision detection
        self.stuck_counter = 0
        self.last_movement_position = (0, 0)
        self.movement_actions = [1, 2, 3, 4]  # Up, Down, Left, Right
        self.recent_actions = []  # Track recent actions for pattern detection
        self.no_op_streak = 0  # Count consecutive no-op actions
        # Action logging (for training visualization)
        self.action_counts = np.zeros(self.action_space.n, dtype=np.int64)
        self.action_log = []  # recent actions (bounded)
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset the emulator
        self.pyboy.stop()
        self.pyboy = PyBoy(
            self.rom_path,
            window="headless" if self.headless else "SDL2"
        )
        self.screen = self.pyboy.screen
        self.memory = PokemonRedMemory(self.pyboy)
        
        # Skip intro and get to gameplay
        # This advances the game to a playable state
        # In a real implementation, you'd load a save state here
        for _ in range(100):  # Skip some initial frames
            self.pyboy.tick()
        
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
        
        # Reset collision detection
        self.stuck_counter = 0
        self.last_movement_position = self.memory.get_player_position()
        self.recent_actions = []
        self.no_op_streak = 0
        self.action_counts[:] = 0
        self.action_log = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        
        # Store action before execution
        self.last_action = action
        
        # Track no-op streak
        if action == 0:
            self.no_op_streak += 1
        else:
            self.no_op_streak = 0
        
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
        
        # PyBoy 2.x API: use button_press and button_release methods
        # Press button
        self.pyboy.button_press(button)
        self.pyboy.tick()
        
        # Release button
        self.pyboy.button_release(button)
    
    def _get_observation(self):
        """Get the current screen as observation"""
        # Get screen as numpy array
        # PyBoy 2.x API: use .ndarray instead of .screen_ndarray()
        screen_array = self.screen.ndarray
        
        # Convert to grayscale
        gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension
        observation = np.expand_dims(resized, axis=-1)
        
        return observation
    
    def _calculate_reward(self):
        """Calculate reward based on game state"""
        reward = REWARDS["step"]  # Small negative reward per step
        
        # Penalize menu actions (Start=7, Select=8)
        if self.last_action in [7, 8]:
            reward += REWARDS["menu_action"]
        
        # Check for movement and collision detection
        current_position = self.memory.get_player_position()
        moved = current_position != self.previous_position
        
        if moved:
            # Player moved successfully
            reward += REWARDS["movement"]
            self.previous_position = current_position
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

        # Penalize no-op and long no-op streaks
        if self.last_action == 0:
            reward += REWARDS["no_op"]
            if self.no_op_streak >= 5:
                reward += REWARDS["no_op_streak"]
        
        # Penalty for being stuck in same position for too long
        if self.stuck_counter > 10:  # Stuck for 10+ steps
            reward += REWARDS["stuck"]
            self.stuck_counter = 0  # Reset to avoid continuous penalty
        
        # Detect repetitive movement patterns (e.g., up-down-up-down)
        if len(self.recent_actions) >= 8:
            # Check for alternating patterns that suggest being stuck
            recent_movement = [a for a in self.recent_actions[-8:] if a in self.movement_actions]
            if len(recent_movement) >= 6:
                # Check for simple alternating patterns (A-B-A-B)
                if (len(recent_movement) >= 4 and 
                    recent_movement[-4:] == recent_movement[-6:-2]):
                    reward += REWARDS["stuck"] * 2  # Extra penalty for repetitive behavior
        
        # Map transitions and first-time map visit bonus
        current_map_id = self.memory.get_map_id()
        if current_map_id != self.previous_map_id:
            reward += REWARDS["map_transition"]
            if current_map_id not in self.seen_maps_this_episode:
                reward += REWARDS["first_time_on_map"]
                self.seen_maps_this_episode.add(current_map_id)
            self.previous_map_id = current_map_id

        # Check if entered grass area (dense progress cue)
        in_grass = self.memory.in_grass_area()
        
        if in_grass and not self.entered_grass:
            reward += REWARDS["move_to_grass"]
            self.entered_grass = True
        
        # Check if battle started
        in_battle = self.memory.is_in_battle()
        if in_battle and not self.battle_started:
            reward += REWARDS["start_battle"]
            self.battle_started = True
            self.initial_hp = self.memory.get_first_pokemon_hp()
        
        # Check if caught a Pokemon or obtained a starter outside battle
        current_party_count = self.memory.get_party_count()
        if current_party_count > self.previous_party_count:
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
        
        # Bonus for trying different actions when stuck
        if self.stuck_counter > 5 and len(self.recent_actions) >= 5:
            unique_actions = len(set(self.recent_actions[-5:]))
            if unique_actions >= 3:  # Tried at least 3 different actions
                reward += REWARDS["diversity_bonus"]  # Bonus for trying different approaches
        
        # Extra bonus for trying all 4 movement directions when stuck
        if self.stuck_counter > 10 and len(self.recent_actions) >= 10:
            recent_movement = [a for a in self.recent_actions[-10:] if a in self.movement_actions]
            unique_directions = len(set(recent_movement))
            if unique_directions >= 4:  # Tried all 4 directions (up, down, left, right)
                reward += REWARDS["diversity_bonus"] * 2  # Double bonus for trying all directions
        
        return reward
    
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
        low = np.repeat(self.observation_space.low, stack_frames, axis=-1)
        high = np.repeat(self.observation_space.high, stack_frames, axis=-1)
        self.observation_space = spaces.Box(
            low=low[:, :, :stack_frames],
            high=high[:, :, :stack_frames],
            dtype=np.uint8
        )
    
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

