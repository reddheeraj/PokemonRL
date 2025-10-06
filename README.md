# Pokemon Red Reinforcement Learning Project

<div align="center">

**Teaching an AI to play Pokemon Red using Deep Reinforcement Learning**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

## 📋 Project Goal

Train a reinforcement learning agent to accomplish a specific task in Pokemon Red:

> **"After walking in Pallet Town, catch one Pokemon in the grass"**

This project uses:
- **PyBoy**: A Python Game Boy emulator
- **Stable-Baselines3**: State-of-the-art RL algorithms
- **PPO**: Proximal Policy Optimization algorithm
- **Custom Gym Environment**: Tailored for Pokemon Red gameplay

---

## 🎯 Features

- ✅ Custom OpenAI Gymnasium environment for Pokemon Red
- ✅ Visual training process (watch the AI learn in real-time)
- ✅ Memory reading utilities to track game state
- ✅ Reward shaping for efficient learning
- ✅ Model checkpointing and TensorBoard logging
- ✅ Inference mode to watch trained agents play
- ✅ Frame stacking for temporal understanding

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.10 or higher** (3.10, 3.11, or 3.12 recommended)
- **Pokemon Red ROM** (legally obtained) at `PokeRL/roms/PokemonRed.gb`
- **Windows/Linux/Mac** with display support for visualization

### Step 1: Clone or Navigate to Project Directory

```bash
cd PokeRL
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical computing
- `torch` - Deep learning framework
- `gymnasium` - RL environment framework
- `stable-baselines3` - RL algorithms (PPO, etc.)
- `pyboy` - Game Boy emulator
- `opencv-python` - Image processing
- `tensorboard` - Training visualization
- `matplotlib` - Plotting
- `tqdm` - Progress bars
- `pandas` - Data handling

### Step 4: Verify ROM File

Ensure your Pokemon Red ROM is located at:
```
PokeRL/
├── roms/
│   └── PokemonRed.gb  ← Your ROM file here
├── train.py
├── play.py
└── ...
```

### Step 5: Test the Environment

Before training, verify everything is working:

```bash
python test_env.py
```

This will:
- Create the Pokemon Red environment
- Display the game window
- Execute random actions for 1000 steps
- Print game state information

**Expected output**: You should see a game window with Pokemon Red running and random movements.

---

## 🚀 Training the Agent

### Basic Training

Start training with default parameters:

```bash
python train.py
```

This will:
- Create the environment
- Initialize a PPO agent
- Start training for 1,000,000 timesteps
- Display the game window during training
- Save checkpoints every 50,000 steps
- Log metrics to TensorBoard

### Training Options

```bash
# Train for more steps
python train.py --timesteps 2000000

# Train without visualization (faster, for overnight training)
python train.py --headless

# Continue training from a checkpoint
python train.py --model models/pokemon_rl_model_20250106_120000_500000_steps.zip

# Custom ROM path
python train.py --rom path/to/your/rom.gb
```

### Monitoring Training with TensorBoard

Open a new terminal and run:

```bash
tensorboard --logdir=logs
```

Then open your browser to `http://localhost:6006` to see:
- Episode rewards over time
- Loss curves
- Learning rate schedules
- And more training metrics

### What to Expect During Training

**Initial Phase (0-100k steps):**
- Agent explores randomly
- Learns basic movement
- Discovers how to navigate Pallet Town

**Middle Phase (100k-500k steps):**
- Agent learns to move toward grass areas
- Begins to understand battle mechanics
- May enter battles occasionally

**Advanced Phase (500k-1M+ steps):**
- Agent consistently reaches grass
- Engages in wild battles
- Learns to catch Pokemon (goal achievement!)

**Note**: Training time varies:
- With GPU: ~4-8 hours for 1M steps
- CPU only: ~12-24 hours for 1M steps

---

## 🎮 Watching the Trained Agent Play

After training, watch your agent play:

```bash
python play.py --model models/pokemon_final_20250106_120000.zip
```

Options:

```bash
# Watch for more episodes
python play.py --model models/your_model.zip --episodes 10

# Change visualization speed
python play.py --model models/your_model.zip --fps 60

# Use custom ROM
python play.py --model models/your_model.zip --rom path/to/rom.gb
```

This will:
- Load your trained model
- Display the game window
- Show the agent playing
- Print statistics (position, rewards, party count)

---

## 📊 Project Structure

```
PokeRL/
├── roms/
│   └── PokemonRed.gb          # Your Pokemon Red ROM
├── models/                     # Saved model checkpoints (created during training)
├── logs/                       # TensorBoard logs (created during training)
├── saves/                      # Save states (optional, for faster training)
│
├── config.py                   # Configuration and hyperparameters
├── memory_reader.py            # Game memory reading utilities
├── pokemon_env.py              # Custom Gym environment
├── train.py                    # Training script
├── play.py                     # Inference/playback script
├── test_env.py                 # Environment testing script
├── setup_savestate.py          # Helper to create save states
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Training Parameters
```python
TOTAL_TIMESTEPS = 1_000_000    # Total training steps
LEARNING_RATE = 0.0003         # PPO learning rate
N_STEPS = 2048                 # Steps per environment update
BATCH_SIZE = 64                # Batch size for training
GAMMA = 0.99                   # Discount factor
```

### Environment Settings
```python
MAX_STEPS_PER_EPISODE = 5000   # Max steps before episode ends
SKIP_FRAMES = 4                # Frame skip (action repeat)
```

### Reward Values
```python
REWARDS = {
    "step": -0.01,              # Step penalty (encourages efficiency)
    "move_to_grass": 1.0,       # Entering grass area
    "start_battle": 5.0,        # Starting a wild battle
    "catch_pokemon": 100.0,     # Successfully catching a Pokemon!
}
```

---

## 🔧 Advanced Usage

### Creating a Save State

To skip the intro and start training from a specific point:

```bash
python setup_savestate.py
```

This opens the game. Play to your desired starting point (e.g., outside Oak's lab with a starter Pokemon), then close the window. The state will be saved.

Then modify `pokemon_env.py` to load this save state in the `reset()` method.

### Custom Reward Functions

The reward function is in `pokemon_env.py`, method `_calculate_reward()`. You can modify it to:
- Reward specific movements
- Penalize getting stuck
- Reward exploration
- Etc.

### Using Different RL Algorithms

The code uses PPO by default, but you can try others from Stable-Baselines3:

```python
from stable_baselines3 import DQN, A2C, SAC

# In train.py, replace:
model = PPO(...)
# with:
model = DQN(...)
```

---

## 📈 Expected Results

### Success Criteria

The agent has learned successfully when it:
1. ✅ Consistently navigates from starting position to grass area
2. ✅ Enters wild Pokemon battles
3. ✅ Successfully catches at least one Pokemon
4. ✅ Episode reward > 100 (indicates Pokemon was caught)

### Typical Learning Curve

- **Episodes 1-100**: Mostly random movement, low rewards
- **Episodes 100-500**: Learns navigation, reaches grass occasionally
- **Episodes 500-1000+**: Consistent grass navigation, increasing battle frequency
- **After 1000+ episodes**: Should achieve goal occasionally to frequently

---

## 🐛 Troubleshooting

### Issue: "ROM file not found"
**Solution**: Ensure `PokemonRed.gb` is in the `roms/` directory with exact filename.

### Issue: PyBoy window not appearing
**Solution**: 
- Check your display settings
- Try running without `--headless` flag
- Ensure SDL2 is installed: `pip install pysdl2`

### Issue: Training is very slow
**Solutions**:
- Use `--headless` flag to disable rendering
- Reduce `N_STEPS` in `config.py`
- Ensure PyTorch is using GPU: `torch.cuda.is_available()` should return `True`

### Issue: Agent not learning
**Solutions**:
- Check reward function is working (see TensorBoard)
- Increase training time (try 2-5M timesteps)
- Adjust reward values in `config.py`
- Verify memory addresses are correct for your ROM version

### Issue: CUDA out of memory
**Solutions**:
- Reduce `BATCH_SIZE` in `config.py`
- Use CPU instead: Set `device="cpu"` in `train.py`

### Issue: Import errors
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

---

## 🎓 How It Works

### The RL Loop

```
1. Agent observes game screen (84x84 grayscale, 4 frames stacked)
2. Agent selects action (up, down, left, right, A, B, start, select)
3. Action is executed in the game
4. Game state changes
5. Reward is calculated based on game state changes
6. Agent learns from this experience
7. Repeat!
```

### Observation Space

- **Input**: 84x84x4 grayscale image (4 stacked frames)
- **Processing**: Downsampled and grayscaled from original 160x144 RGB
- **Why stacking?**: Gives the agent temporal information (movement direction, animation states)

### Action Space

9 discrete actions:
- 0: No-op (do nothing)
- 1: Up
- 2: Down
- 3: Left
- 4: Right
- 5: A (interact, confirm)
- 6: B (cancel, run)
- 7: Start (menu)
- 8: Select

### Reward Structure

The agent receives rewards for:
- **Exploration**: Small bonus for visiting new game states
- **Progress**: Larger rewards for entering grass areas
- **Engagement**: Reward for starting battles
- **Goal Achievement**: Large reward for catching a Pokemon

Small negative reward per step encourages efficient behavior.

---

## 📚 Learning Resources

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Explained](https://openai.com/blog/openai-baselines-ppo/)

### Pokemon Red
- [Pokemon Red Disassembly](https://github.com/pret/pokered)
- [PyBoy Documentation](https://docs.pyboy.dk/)
- [Pokemon Red RAM Map](https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map)

---

## 🤝 Contributing

Improvements welcome! Some ideas:
- Better reward shaping
- Curriculum learning (progressive difficulty)
- Multi-objective rewards
- Different RL algorithms comparison
- Extended goals (catch specific Pokemon, reach certain locations)

---

## ⚖️ Legal Notice

This project is for educational purposes. You must own a legal copy of Pokemon Red to use this software. The ROM file is not included and must be obtained legally.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎉 Acknowledgments

- **PyBoy**: Excellent Python Game Boy emulator
- **Stable-Baselines3**: Making RL accessible
- **Pokemon Red**: The classic game that started it all
- **OpenAI Gym**: Standard RL environment interface

---

## 📧 Questions?

If you encounter issues:
1. Check the Troubleshooting section
2. Verify your setup with `test_env.py`
3. Review configuration in `config.py`
4. Check TensorBoard logs for training issues

---

<div align="center">

**Happy Training! 🎮🤖**

*May your agent catch them all!*

</div>

