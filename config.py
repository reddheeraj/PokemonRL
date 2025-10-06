"""
Configuration file for Pokemon Red RL training
"""

# ROM Configuration
ROM_PATH = "roms/PokemonRed.gb"

# Training Configuration
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
LEARNING_RATE = 0.0003
N_STEPS = 2048  # Number of steps to run for each environment per update
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95

# Environment Configuration
MAX_STEPS_PER_EPISODE = 5000  # Max steps before episode ends
SKIP_FRAMES = 4  # Number of frames to skip (action repeat)
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144

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

# Reward Configuration
REWARDS = {
    "step": -0.01,  # Small negative reward per step to encourage efficiency
    "move_to_grass": 1.0,  # Reward for entering grass area
    "start_battle": 5.0,  # Reward for starting a wild battle
    "catch_pokemon": 100.0,  # Large reward for catching a Pokemon
    "pokemon_gained": 50.0,  # Reward for increasing party count
    "hp_damage": -0.1,  # Small penalty for taking damage
    "menu_action": -0.05,  # Penalty for opening menus (Start/Select)
    "movement": 0.03,  # Slightly higher reward for actual movement
    "wall_hit": -0.2,  # Penalty for hitting walls/obstacles
    "stuck": -0.05,  # Penalty for being stuck in same position
    "diversity_bonus": 0.1,  # Bonus for trying different directions
    # Dense, event-like rewards inspired by PokeRL
    "map_transition": 0.2,  # Reward for any map change (warp/door)
    "first_time_on_map": 0.3,  # Bonus the first time a map_id is seen in an episode
    "starter_obtained": 20.0,  # One-time bonus when party first becomes non-zero outside battle
    # Anti-idle penalties
    "no_move": -0.03,  # Penalty when position does not change (any action)
    "no_op": -0.05,  # Penalty for taking explicit no-op action (action 0)
    "no_op_streak": -0.1,  # Extra penalty when repeatedly taking no-op
}

