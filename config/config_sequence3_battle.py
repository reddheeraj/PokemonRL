"""
Configuration for Sequence 3: Battle Training
Goal: Learn battle mechanics (attack, items, switch Pokemon, run)
Note: In battle, movement actions (Up/Down/Left/Right) navigate menus,
      A confirms selection, B cancels/goes back
"""

from config.config import *  # Import base config

# Override ROM path for this sequence
ROM_PATH = "roms/sequence3_battle/PokemonRed.gb"

# Training Configuration - Battle episodes are shorter
MAX_STEPS_PER_EPISODE = 3000  # Battle episodes
TOTAL_TIMESTEPS = 1_000_000  # More timesteps for complex battle mechanics

# Reward Configuration - Focused on battle mechanics
REWARDS = {
    "step": 0.0,  # No step penalty
    
    # Battle action rewards
    "battle_action_taken": 0.5,  # Reward for taking any battle action (not no-op)
    "attack_selected": 2.0,  # Reward for selecting an attack
    "attack_used": 5.0,  # Reward for successfully using an attack
    "enemy_damaged": 3.0,  # Reward for damaging enemy
    "enemy_fainted": 20.0,  # Large reward for defeating enemy
    "battle_won": 50.0,  # Very large reward for winning battle
    
    # Battle navigation rewards
    "menu_navigation": 0.2,  # Small reward for navigating menus
    "item_used": 3.0,  # Reward for using an item
    "pokemon_switched": 2.0,  # Reward for switching Pokemon
    
    # Penalties
    "no_battle_action": -0.1,  # Penalty for not taking action in battle
    "battle_timeout": -1.0,  # Penalty for taking too long in battle
    "battle_lost": -10.0,  # Penalty for losing battle
    "friendly_damaged": -1.0,  # Penalty for taking damage
    "friendly_fainted": -5.0,  # Penalty for Pokemon fainting
    
    # Anti-spam penalties
    "no_op": -0.05,  # Penalty for no-op in battle
    "button_spam": -0.2,  # Penalty for button spam in battle
    
    # Movement rewards (for menu navigation)
    "movement": 0.1,  # Small reward for menu navigation
}

# Training mode identifier
TRAINING_MODE = "battle"

