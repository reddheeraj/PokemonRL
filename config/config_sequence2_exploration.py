"""
Configuration for Sequence 2: Exploration Training
Goal: Learn to explore Pallet Town and reach grass area to trigger Professor Oak
"""

from config.config import *  # Import base config

# Override ROM path for this sequence
ROM_PATH = "roms/sequence2_exploration/PokemonRed.gb"

# Training Configuration
MAX_STEPS_PER_EPISODE = 5000  # Longer episodes for exploration
TOTAL_TIMESTEPS = 1_000_000  # More timesteps for complex exploration

# Reward Configuration - Focused on exploration and reaching grass
REWARDS = {
    "step": 0.0,  # No step penalty
    # Movement rewards
    "movement": 1.0,  # Good reward for actual movement
    "distance_traveled": 0.2,  # Reward for moving away from previous position
    "new_position": 0.5,  # Reward for visiting a new position
    "exploration_bonus": 2.0,  # Large bonus for significant exploration
    
    # Exploration specific rewards
    "map_transition": 3.0,  # Reward for map changes
    "first_time_on_map": 5.0,  # Bonus for new map exploration
    "move_to_grass": 20.0,  # LARGE reward for entering grass area (main goal!)
    "start_battle": 10.0,  # Reward for starting a wild battle (triggers Oak)
    
    # Anti-spam penalties - BALANCED
    "no_move": -0.05,  # Very small penalty when position does not change
    "no_op": -0.02,  # Tiny penalty for no-op action
    "a_button_spam": -0.1,  # Small penalty for A button spam
    "b_button_spam": -0.1,  # Small penalty for B button spam
    "button_spam_streak": -0.2,  # Moderate penalty for consecutive button spam
    
    # Anti-loop penalties - BALANCED
    "same_position_streak": -0.05,  # Tiny penalty for staying in same position too long
    "movement_required": -0.1,  # Small penalty if no movement for too long
    "pattern_penalty": -0.1,  # Small penalty for repetitive patterns
    "anti_loop_penalty": -1.0,  # Penalty for returning to same position too often
    "loop_detection_penalty": -2.0,  # Strong penalty for excessive revisits
    
    # Exploration rewards
    "curriculum_exploration": 5.0,  # Large bonus for exploring new areas
    "position_novelty": 2.0,  # Reward for visiting new positions
    "loop_break_bonus": 1.0,  # Bonus for breaking out of loops
    "variety_bonus": 0.3,  # Bonus for action variety
    "exploration_momentum": 0.5,  # Reward for continuous exploration
    
    # Wall/stuck penalties
    "wall_hit": -0.05,  # Very small penalty for hitting walls
    "stuck": -0.05,  # Very small penalty for being stuck
}

# Training mode identifier
TRAINING_MODE = "exploration"

