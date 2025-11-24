"""
Configuration for Sequence 1: House Exit Training
Goal: Learn to exit Red's house (navigate from inside house to outside)
"""

from config.config import *  # Import base config

# Override ROM path for this sequence
ROM_PATH = "roms/sequence1_house_exit/PokemonRed.gb"

# Training Configuration - Focused on short episodes
MAX_STEPS_PER_EPISODE = 2000  # Shorter episodes for house navigation
TOTAL_TIMESTEPS = 500_000  # Less timesteps needed for simpler task

# Reward Configuration - Focused on house exit
REWARDS = {
    "step": 0.0,  # No step penalty
    # Movement rewards - encourage exploration within house
    "movement": 0.5,  # Reward for actual movement
    "distance_traveled": 0.1,  # Reward for moving away from previous position
    "new_position": 0.3,  # Reward for visiting a new position
    "exploration_bonus": 1.0,  # Bonus for significant exploration
    
    # House exit specific rewards
    "map_transition": 10.0,  # Large reward for exiting house (map change)
    "first_time_on_map": 5.0,  # Bonus for new map exploration
    
    # Anti-spam penalties - SMALL
    "no_move": -0.05,  # Small penalty when position does not change
    "no_op": -0.02,  # Tiny penalty for no-op action
    "a_button_spam": -0.1,  # Small penalty for A button spam
    "b_button_spam": -0.1,  # Small penalty for B button spam
    "button_spam_streak": -0.2,  # Moderate penalty for consecutive button spam
    
    # Anti-loop penalties - SMALL
    "same_position_streak": -0.05,  # Tiny penalty for staying in same position too long
    "movement_required": -0.1,  # Small penalty if no movement for too long
    "pattern_penalty": -0.1,  # Small penalty for repetitive patterns
    "anti_loop_penalty": -0.5,  # Penalty for returning to same position too often
    "loop_detection_penalty": -1.0,  # Strong penalty for excessive revisits
    
    # Exploration rewards
    "curriculum_exploration": 2.0,  # Bonus for exploring new areas
    "position_novelty": 1.0,  # Reward for visiting new positions
    "loop_break_bonus": 0.5,  # Bonus for breaking out of loops
    "variety_bonus": 0.2,  # Bonus for action variety
    "exploration_momentum": 0.3,  # Reward for continuous exploration
    
    # Wall/stuck penalties
    "wall_hit": -0.05,  # Very small penalty for hitting walls
    "stuck": -0.05,  # Very small penalty for being stuck
}

# Training mode identifier
TRAINING_MODE = "house_exit"

