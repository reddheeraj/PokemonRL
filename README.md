# PokeRL: Playing Pokémon Red with Deep Reinforcement Learning

<div align="center">

**Training AI agents to play Pokémon Red using Curriculum Learning and PPO**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-orange.svg)
![PyBoy](https://img.shields.io/badge/PyBoy-2.0+-green.svg)

</div>

---

## Overview

This project trains reinforcement learning agents to play Pokémon Red (1996) using a **modular curriculum learning** approach. Instead of training on the entire game at once, we decompose the task into three learnable sequences:

| Sequence | Objective | Description |
|----------|-----------|-------------|
| **1. House Exit** | Exit Red's house | Navigate from bedroom to outside |
| **2. Exploration** | Reach the grass | Travel through Pallet Town to trigger Oak's event |
| **3. Battle** | Win against Blue | Learn battle mechanics and win the rival fight |

Each sequence has its own:
- Custom reward function
- Save state (starting point)
- Training configuration
- Trained model

---

## Quick Start

### Watch Pre-Trained Agents Play

> Note: The models are not perfect and the agents may not be able to complete the tasks perfectly. You may have to give a manual trigger like pressing up arrow key or completing a dialogue scene to continue the task.

```bash
# Sequence 1: House Exit
python scripts/play.py --model final_models/house_exit_20251204_112405/house_exit_20251204_112405_final.zip --sequence 1 --fps 60

# Sequence 2: Exploration  
python scripts/play.py --model final_models/exploration_20251204_114057/exploration_20251204_114057_final.zip --sequence 2 --fps 60

# Sequence 3: Battle
python scripts/play.py --model final_models/battle_20251204_120117/battle_20251204_120117_final.zip --sequence 3 --fps 60
```

### Options for `play.py`

```bash
--model PATH      # Path to trained model (.zip file)
--sequence N      # Sequence number (1, 2, or 3) - auto-loads correct ROM
--episodes N      # Number of episodes to play (default: 5)
--fps N           # Playback speed (default: 30)
--rom PATH        # Override ROM path (optional)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Pokemon Red ROM (legally obtained)
- Windows/Linux/Mac

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/reddheeraj/PokemonRL.git
cd PokemonRL

# 2. Create conda environment
conda create -n pokemonred python=3.10
conda activate pokemonred

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your ROM
# unzip the roms.zip folder and place it in the root directory
```

### Verify Installation

```bash
python scripts/test_env.py
```

You should see the game window with random actions being taken.

---

## Project Structure

```
PokeRL/
├── pokemon_env.py              # Core Gymnasium environment
├── memory_reader.py            # Game RAM reading utilities
│
├── config/                     # Configuration files
│   ├── config.py               # Base configuration
│   ├── config_sequence1_house_exit.py
│   ├── config_sequence2_exploration.py
│   └── config_sequence3_battle.py
│
├── scripts/                    # Executable scripts
│   ├── play.py                 # Watch trained agents play
│   ├── train_sequence1_house_exit.py
│   ├── train_sequence2_exploration.py
│   ├── train_sequence3_battle.py
│   ├── manual_control.py       # Play manually with visualization
│   ├── create_sequence_savestate.py
│   └── test_env.py
│
├── final_models/               # Pre-trained models
│   ├── house_exit_20251204_112405/
│   ├── exploration_20251204_114057/
│   └── battle_20251204_120117/
│
├── roms/                       # ROM files
│   ├── PokemonRed.gb           # Base ROM
│   ├── sequence1_house_exit/   # Sequence 1 ROM + save state
│   ├── sequence2_exploration/  # Sequence 2 ROM + save state
│   └── sequence3_battle/       # Sequence 3 ROM + save state
│
├── logs/                       # TensorBoard training logs
├── models/                     # Training checkpoints
├── recordings/                 # Training videos
└── tests/                      # Visualization scripts
```

---

## Training Your Own Agents

### Train Each Sequence

```bash
# Sequence 1: House Exit (fastest to train)
python scripts/train_sequence1_house_exit.py --timesteps 500000 --envs 4

# Sequence 2: Exploration
python scripts/train_sequence2_exploration.py --timesteps 1000000 --envs 4

# Sequence 3: Battle
python scripts/train_sequence3_battle.py --timesteps 1000000 --envs 4
```

### Training Options

```bash
--timesteps N     # Total training timesteps (default varies by sequence)
--envs N          # Number of parallel environments (default: 4)
--headless        # Run without display (faster training)
--no-video        # Disable video recording
--model-name NAME # Custom name for saved model
```

### Monitor Training

```bash
tensorboard --logdir=logs
# Open http://localhost:6006 in browser
```

---

## Manual Play & Debugging

### Play Manually with Visualization

```bash
# Play with visited mask visualization
python scripts/manual_control.py --sequence 1

# With zoomed mask view
python scripts/manual_control.py --sequence 2 --zoom
```

**Controls:**
- Arrow Keys: Move
- A: A button
- Space: B button
- Enter: Start button
- Z: Toggle mask zoom
- Q/ESC: Quit

### Create Custom Save States

```bash
python scripts/create_sequence_savestate.py 1  # For sequence 1
python scripts/create_sequence_savestate.py 2  # For sequence 2
python scripts/create_sequence_savestate.py 3  # For sequence 3
```

---

## Technical Details

### Environment

- **Observation**: 72×80×8 (4 stacked frames × 2 channels: grayscale + visited mask)
- **Action Space**: 7 discrete actions (No-op, Up, Down, Left, Right, A, B)
- **Frame Skip**: 4 frames per action

### Neural Network (CNN Policy)

```
Input (8, 72, 80)
    ↓
Conv2d(8→32, 8×8, stride=4)  →  ReLU
    ↓
Conv2d(32→64, 4×4, stride=2) →  ReLU
    ↓
Conv2d(64→64, 3×3, stride=1) →  ReLU
    ↓
Flatten → Linear(1920→512) → ReLU
    ↓
┌───────────┴───────────┐
Policy Head          Value Head
(7 actions)          (state value)
```

**Total Parameters**: ~1M

### Key Features

1. **Anti-Loop Detection**: Multi-layered system prevents repetitive behavior
2. **Dense Reward Shaping**: Hierarchical rewards for consistent learning signal
3. **Per-Map Visited Mask**: Spatial memory for exploration
4. **Double-Press Movement**: Handles Pokémon Red's unique movement mechanics

---

## Reward Structure

### Sequence 1: House Exit
```python
REWARDS = {
    "movement": 0.5,
    "map_transition": 10.0,    # Goal: exit house
    "new_position": 0.3,
    "anti_loop_penalty": -0.5,
}
```

### Sequence 2: Exploration
```python
REWARDS = {
    "movement": 1.0,
    "move_to_grass": 20.0,     # Goal: reach grass
    "curriculum_exploration": 5.0,
    "loop_detection_penalty": -2.0,
}
```

### Sequence 3: Battle
```python
REWARDS = {
    "attack_used": 5.0,
    "enemy_fainted": 20.0,
    "battle_won": 50.0,        # Goal: win battle
    "battle_lost": -10.0,
}
```

---

## Troubleshooting

### ROM Not Found
```
Ensure your ROM is at: roms/PokemonRed.gb
```

### PyBoy Window Not Appearing
```bash
pip install pysdl2 pysdl2-dll
```

### CUDA Out of Memory
Reduce parallel environments:
```bash
python scripts/train_sequence1_house_exit.py --envs 2
```

### Model Not Working During Play
Ensure you're using the correct sequence flag:
```bash
# Model trained on sequence 1 must be played with --sequence 1
python scripts/play.py --model <seq1_model.zip> --sequence 1
```

---

## Dependencies

```
gymnasium
stable-baselines3
torch
pyboy
opencv-python
numpy
tensorboard
```

---

## Results Summary

| Sequence | Training Time | Success Rate |
|----------|--------------|--------------|
| House Exit | ~1 hour | 70% |
| Exploration | ~2 hours | 60% |
| Battle | ~2 hours | 50% |

---

## Acknowledgments

- **PyBoy**: Python Game Boy emulator
- **Stable-Baselines3**: RL algorithm implementations
- **Dr. Rubinstein**: PokeRL framework inspiration
- **Pokemon Red Disassembly**: Memory address documentation

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Happy Training! 🎮🤖**

*Gotta train 'em all!*

</div>
