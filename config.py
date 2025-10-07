"""
Configuration file for Pokemon Red RL training - ANTI-LOOP VERSION
"""

# ROM Configuration
ROM_PATH = "roms/PokemonRed.gb"

# Training Configuration
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
LEARNING_RATE = 0.0003
N_STEPS = 2048  # Number of steps to run for each environment per update
BATCH_SIZE = 128
N_EPOCHS = 10
GAMMA = 0.999  # Discount factor
GAE_LAMBDA = 0.95

# Environment Configuration
MAX_STEPS_PER_EPISODE = 5000  # Max steps before episode ends
SKIP_FRAMES = 4  # Number of frames to skip (action repeat)
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144
USE_VISITED_MASK = True  # Add an exploration visited-mask channel to observations
# TESTING SWITCH: Disable menu buttons (Start/Select) during training/playing to avoid spam
DISABLE_MENU_BUTTONS = True
VISITED_MASK_PER_MAP = True  # Maintain a separate visited mask per map_id within an episode
NUM_ENVS = 4  # Number of parallel environments for training

# Visualization
RENDER_TRAINING = True  # Show game screen during training
RENDER_FPS = 30  # FPS for rendering

# Checkpoint Configuration
SAVE_FREQ = 50_000  # Save model every N steps
MODEL_DIR = "models"
LOG_DIR = "logs"

# Game-specific addresses (Pokemon Red memory locations)
# These are approximate and might need adjustment
MEMORY_ADDRESSES = {
    "player_x": 0xD362,  # Player X position
    "player_y": 0xD361,  # Player Y position
    "map_id": 0xD35E,    # Current map ID
    "battle_state": 0xD057,  # Battle state flag
    "party_count": 0xD163,  # Number of Pokemon in party
    "wild_battle": 0xD057,  # Wild battle indicator
    "in_battle": 0xD057,  # In battle flag
    "pokemon_hp": 0xD16C,  # First Pokemon HP
}

# Map IDs (approximate)
PALLET_TOWN_MAP_ID = 0  # Pallet Town map
ROUTE_1_MAP_ID = 12  # Route 1 (grass area)

# Reward Configuration - ANTI-LOOP for exploration learning
REWARDS = {
    "step": 0.0,  # No step penalty/reward
    "move_to_grass": 10.0,  # Large reward for entering grass area
    "start_battle": 10.0,  # Reward for starting a wild battle
    "catch_pokemon": 100.0,  # Large reward for catching a Pokemon
    "pokemon_gained": 50.0,  # Reward for increasing party count
    "movement": 1.0,  # Good reward for actual movement
    "wall_hit": -0.05,  # Very small penalty for hitting walls
    "stuck": -0.05,  # Very small penalty for being stuck
    "diversity_bonus": 0.3,  # Bonus for trying different directions
    # Dense, event-like rewards inspired by PokeRL
    "map_transition": 3.0,  # Reward for map changes
    "first_time_on_map": 5.0,  # Bonus for new map exploration
    "starter_obtained": 50.0,  # One-time bonus when party first becomes non-zero outside battle
    # BALANCED anti-spam penalties - MUCH SMALLER
    "no_move": -0.05,  # Very small penalty when position does not change (was -0.5)
    "no_op": -0.02,  # Tiny penalty for no-op action (was -0.2)
    "a_button_spam": -0.1,  # Small penalty for A button spam (was -1.0)
    "b_button_spam": -0.1,  # Small penalty for B button spam (was -1.0)
    "button_spam_streak": -0.2,  # Moderate penalty for consecutive button spam (was -2.0)
    # NEW: Distance-based exploration rewards
    "distance_traveled": 0.2,  # Reward for moving away from previous position
    "new_position": 0.5,  # Reward for visiting a new position
    "exploration_bonus": 2.0,  # Large bonus for significant exploration
    # NEW: Loop-breaking rewards - MUCH SMALLER
    "pattern_penalty": -0.1,  # Small penalty for repetitive patterns (was -1.0)
    "loop_break_bonus": 1.0,  # Bonus for breaking out of loops
    "variety_bonus": 0.3,  # Bonus for action variety
    # NEW: Position-based penalties - MUCH SMALLER
    "same_position_streak": -0.05,  # Tiny penalty for staying in same position too long (was -0.5)
    "movement_required": -0.1,  # Small penalty if no movement for too long (was -1.0)
    # ANTI-LOOP REWARDS - NEW!
    "curriculum_exploration": 5.0,  # Large bonus for exploring new areas
    "anti_loop_penalty": -1.0,  # Strong penalty for returning to same area
    "position_novelty": 2.0,  # Reward for visiting new positions
    "tile_novelty": 1.0,  # Reward for standing on new tile types
    "loop_detection_penalty": -2.0,  # Strong penalty for detected loops
    "exploration_momentum": 0.5,  # Reward for continuous exploration
}