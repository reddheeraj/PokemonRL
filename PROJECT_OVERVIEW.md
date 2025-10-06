# Pokemon Red RL - Project Overview

## 🎯 Project Goal

Train a deep reinforcement learning agent to accomplish a specific task in Pokemon Red:

> **"After walking in Pallet Town, catch one Pokemon in the grass"**

---

## 📁 Project Structure

```
PokeRL/
│
├── 📄 Core Python Files
│   ├── config.py                 # All hyperparameters and settings
│   ├── memory_reader.py          # Read Pokemon Red game memory
│   ├── pokemon_env.py            # Custom Gym environment
│   ├── train.py                  # Main training script
│   └── play.py                   # Watch trained agent play
│
├── 🛠️ Utility Scripts
│   ├── test_env.py               # Test environment setup
│   ├── setup_savestate.py        # Create game save states
│   ├── visualize_training.py     # Plot training progress
│   └── run_demo.py               # Interactive demo/menu
│
├── 📚 Documentation
│   ├── README.md                 # Complete documentation
│   ├── QUICKSTART.md             # Fast setup guide
│   ├── PROJECT_OVERVIEW.md       # This file
│   └── LICENSE                   # MIT license
│
├── 📦 Dependencies
│   ├── requirements.txt          # Python packages
│   └── .gitignore               # Git ignore rules
│
└── 📂 Directories (created during use)
    ├── roms/                    # Pokemon Red ROM (PokemonRed.gb)
    ├── models/                  # Saved model checkpoints
    ├── logs/                    # TensorBoard training logs
    └── saves/                   # Game save states (optional)
```

---

## 🧠 Technical Architecture

### 1. Environment (`pokemon_env.py`)

**PokemonRedEnv**
- Custom OpenAI Gymnasium environment
- Wraps PyBoy Game Boy emulator
- Provides standard RL interface (reset, step, render)

**Key Features:**
- **Observation**: 84×84 grayscale image, 4 frames stacked
- **Actions**: 9 discrete actions (movement, buttons)
- **Rewards**: Shaped rewards for progress toward goal
- **Episode Management**: Handles resets and termination

### 2. Memory Reader (`memory_reader.py`)

**PokemonRedMemory**
- Reads game RAM directly
- Extracts game state information
- Tracks player position, battles, Pokemon caught, etc.

**Memory Addresses:**
- Player X/Y position
- Current map ID
- Party count (number of Pokemon)
- Battle state
- HP values

### 3. Configuration (`config.py`)

Centralized configuration for:
- **Training hyperparameters** (learning rate, batch size, etc.)
- **Environment settings** (frame skip, episode length, etc.)
- **Reward structure** (values for different achievements)
- **Memory addresses** (Pokemon Red RAM locations)

### 4. Training Pipeline (`train.py`)

**Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Network**: CNN-based (processes images)
- **Training Loop**: Collect experiences → Update policy → Repeat
- **Checkpointing**: Save model every 50k steps
- **Logging**: TensorBoard integration

**Features:**
- Resume training from checkpoints
- Headless mode for faster training
- Progress tracking and statistics
- Graceful interruption handling

### 5. Inference (`play.py`)

- Load trained models
- Run agent in game
- Visualize behavior
- Collect performance statistics

---

## 🎮 How It Works

### The Reinforcement Learning Loop

```
1. Agent observes game screen (image)
   ↓
2. Neural network predicts action
   ↓
3. Action executed in emulator
   ↓
4. Game state changes
   ↓
5. Reward calculated based on progress
   ↓
6. Agent learns from experience
   ↓
7. Repeat!
```

### Observation Processing

```
Raw Game Screen (160×144 RGB)
        ↓
Convert to Grayscale
        ↓
Resize to 84×84
        ↓
Stack 4 consecutive frames
        ↓
Feed to Neural Network (84×84×4)
```

**Why 4 frames?**
- Provides temporal information
- Agent can perceive motion
- Helps understand game dynamics

### Action Space

The agent can choose from 9 actions:
- **0**: No-op (do nothing)
- **1**: Up
- **2**: Down  
- **3**: Left
- **4**: Right
- **5**: A button (interact, confirm)
- **6**: B button (cancel, back)
- **7**: Start (menu)
- **8**: Select

### Reward Function

The agent learns through rewards:

| Event | Reward | Purpose |
|-------|--------|---------|
| Each step | -0.01 | Encourage efficiency |
| Enter grass area | +1.0 | Progress toward goal |
| Start wild battle | +5.0 | Engage with Pokemon |
| Catch Pokemon | +100.0 | **GOAL ACHIEVED!** |
| Visit new state | +0.1 | Encourage exploration |

**Reward Shaping:**
- Small negative reward per step prevents "do nothing" strategy
- Progressively larger rewards guide agent toward goal
- Large final reward for achieving objective

---

## 🔬 Machine Learning Details

### Neural Network Architecture

```
Input: 84×84×4 image (stacked frames)
    ↓
Conv2D (32 filters, 8×8, stride 4) + ReLU
    ↓
Conv2D (64 filters, 4×4, stride 2) + ReLU
    ↓
Conv2D (64 filters, 3×3, stride 1) + ReLU
    ↓
Flatten
    ↓
Fully Connected (512 units) + ReLU
    ↓
Actor Head (9 actions) + Value Head (1 value)
```

### PPO Algorithm

**Proximal Policy Optimization:**
- On-policy algorithm (learns from current policy)
- Uses clipped objective to prevent large policy updates
- Balances exploration and exploitation
- Sample efficient and stable

**Key Hyperparameters:**
- Learning rate: 0.0003
- Discount factor (γ): 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01

### Training Process

**Phase 1: Random Exploration (0-50k steps)**
- Agent has no prior knowledge
- Explores randomly
- Discovers basic controls

**Phase 2: Learning Navigation (50k-200k steps)**
- Learns to move purposefully
- Discovers town boundaries
- Starts to understand spatial layout

**Phase 3: Goal-Directed Behavior (200k-500k steps)**
- Finds paths to grass areas
- Learns to trigger battles
- Improves efficiency

**Phase 4: Task Mastery (500k-1M+ steps)**
- Consistently reaches grass
- Engages in battles regularly
- Learns catching mechanics
- Achieves goal successfully

---

## 📊 Expected Performance

### Success Metrics

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| Avg Episode Reward | < 0 | 10-50 | > 100 |
| % Episodes reaching grass | 0% | 50% | 90%+ |
| % Episodes starting battle | 0% | 20% | 50%+ |
| % Episodes catching Pokemon | 0% | 5% | 20%+ |

### Training Timeline

**100k steps (~1 hour):**
- Learns basic movement
- Random but controlled

**500k steps (~4 hours):**
- Navigates toward grass consistently
- Occasional battle engagement

**1M steps (~8 hours):**
- Regular grass encounters
- Some successful catches
- Beginning of goal achievement

**2M+ steps (16+ hours):**
- Consistent goal achievement
- Efficient behavior
- High success rate

---

## 🚀 Usage Workflows

### Workflow 1: First Time Setup

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Verify setup
python test_env.py

# 3. Start training
python train.py

# 4. Monitor progress
tensorboard --logdir=logs

# 5. Watch trained agent
python play.py --model models/pokemon_final_*.zip
```

### Workflow 2: Continued Training

```bash
# Continue from checkpoint
python train.py --model models/pokemon_rl_model_20250106_500000_steps.zip --timesteps 1000000
```

### Workflow 3: Experimentation

```bash
# 1. Modify config.py (adjust rewards, hyperparameters)
# 2. Train new model
python train.py --timesteps 500000
# 3. Compare results in TensorBoard
tensorboard --logdir=logs
```

### Workflow 4: Quick Demo

```bash
# Interactive demo with menu
python run_demo.py
```

---

## 🔧 Customization Guide

### Modify Reward Function

Edit `pokemon_env.py`, method `_calculate_reward()`:

```python
def _calculate_reward(self):
    reward = REWARDS["step"]
    
    # Add your custom rewards here!
    # Example: Reward for moving north
    if self.memory.get_player_position()[1] < self.previous_y:
        reward += 0.5
    
    # ... rest of reward logic
    return reward
```

### Adjust Hyperparameters

Edit `config.py`:

```python
# Make training faster or slower
LEARNING_RATE = 0.001  # Higher = faster but less stable

# Longer episodes
MAX_STEPS_PER_EPISODE = 10000

# Different reward values
REWARDS["catch_pokemon"] = 200.0  # Even bigger reward!
```

### Change RL Algorithm

Edit `train.py`:

```python
from stable_baselines3 import DQN  # Instead of PPO

model = DQN(
    "CnnPolicy",
    env,
    # ... parameters
)
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

**Problem**: ROM not found
```bash
# Solution: Check file exists
ls roms/PokemonRed.gb

# Place ROM in correct location
cp /path/to/PokemonRed.gb roms/
```

**Problem**: PyBoy window doesn't appear
```bash
# Solution: Install SDL dependencies
# Ubuntu/Debian:
sudo apt-get install libsdl2-dev

# Mac:
brew install sdl2

# Windows: Should work out of the box
```

**Problem**: Training is slow
```bash
# Solution 1: Use headless mode
python train.py --headless

# Solution 2: Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Solution 3: Reduce batch size in config.py
BATCH_SIZE = 32  # Instead of 64
```

**Problem**: Agent not learning
```bash
# Solution 1: Train longer
python train.py --timesteps 2000000

# Solution 2: Check rewards in TensorBoard
tensorboard --logdir=logs

# Solution 3: Adjust reward values in config.py
```

---

## 📈 Monitoring Training

### TensorBoard Metrics

**Key Metrics to Watch:**

1. **rollout/ep_rew_mean** 
   - Average episode reward
   - Should trend upward over time
   - Indicates learning progress

2. **rollout/ep_len_mean**
   - Average episode length
   - May decrease as agent gets more efficient

3. **train/loss**
   - Neural network loss
   - Should generally decrease
   - Some fluctuation is normal

4. **train/learning_rate**
   - Current learning rate
   - May decrease over time if using schedule

### Console Output

During training, watch for:
- **Mean Reward** increasing
- **Episode** count growing
- **Checkpoints** being saved regularly

---

## 🎓 Learning Resources

### Reinforcement Learning
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

### Pokemon Red
- [PyBoy Docs](https://docs.pyboy.dk/)
- [Pokemon Red RAM Map](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map)
- [Pokemon Red Disassembly](https://github.com/pret/pokered)

---

## 🎯 Future Improvements

### Possible Extensions

1. **Extended Goals**
   - Catch multiple Pokemon
   - Reach specific locations
   - Win battles
   - Complete gyms

2. **Better State Representation**
   - Include game variables as features
   - Add previous action history
   - Multi-modal observations

3. **Curriculum Learning**
   - Start with simpler tasks
   - Gradually increase difficulty
   - Use save states for different starting points

4. **Advanced RL Techniques**
   - Curiosity-driven exploration
   - Hierarchical RL
   - Multi-task learning
   - Imitation learning from human play

5. **Performance Optimization**
   - Parallel environments
   - GPU acceleration
   - Faster emulation
   - State caching

---

## 📊 Technical Specifications

### System Requirements

**Minimum:**
- Python 3.10+
- 4 GB RAM
- 2 GHz CPU
- 1 GB disk space

**Recommended:**
- Python 3.10+
- 8+ GB RAM
- 4+ core CPU or CUDA GPU
- 5 GB disk space

### Dependencies

- **numpy**: Array operations
- **torch**: Deep learning
- **gymnasium**: RL framework
- **stable-baselines3**: RL algorithms
- **pyboy**: Game Boy emulator
- **opencv-python**: Image processing
- **tensorboard**: Visualization
- **matplotlib**: Plotting

### Performance

**Training Speed:**
- CPU only: ~100-200 steps/sec
- With GPU: ~500-1000 steps/sec

**Memory Usage:**
- Training: ~2-4 GB RAM
- Inference: ~500 MB RAM

---

## 📄 License

MIT License - See LICENSE file

This is an educational project. You must own a legal copy of Pokemon Red.

---

## 🙏 Acknowledgments

- **Nintendo, Game Freak, The Pokemon Company** - Pokemon Red
- **PyBoy** - Excellent Python Game Boy emulator
- **Stable-Baselines3** - RL algorithms made easy
- **OpenAI Gym/Gymnasium** - Standard RL interface

---

<div align="center">

**Ready to train your Pokemon agent?**

Start with: `python run_demo.py`

Good luck! 🎮🤖

</div>

