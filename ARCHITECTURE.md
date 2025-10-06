# 🏗️ Pokemon Red RL - System Architecture

Visual guide to understanding how all components work together.

---

## 📊 High-Level System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERACTION                         │
├─────────────────────────────────────────────────────────────┤
│  train.py  │  play.py  │  test_env.py  │  run_demo.py      │
└──────┬──────────────┬─────────────────────────────────────┘
       │              │
       │              └──────────────┐
       │                             │
       ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     CONFIGURATION                            │
├─────────────────────────────────────────────────────────────┤
│  config.py - Hyperparameters, rewards, settings, paths      │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                  RL ENVIRONMENT (Gym)                        │
├─────────────────────────────────────────────────────────────┤
│                     pokemon_env.py                           │
│  • PokemonRedEnv (base environment)                          │
│  • PokemonRedWrapper (preprocessing)                         │
│  • Observation: 84x84x4 grayscale stacked frames             │
│  • Action: 9 discrete actions                                │
│  • Reward: Shaped rewards for progress                       │
└──────┬────────────────────────────┬─────────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│   MEMORY READER      │    │     PYBOY EMULATOR           │
├──────────────────────┤    ├──────────────────────────────┤
│  memory_reader.py    │    │  Game Boy Emulation          │
│  • Read RAM          │    │  • Load ROM                  │
│  • Extract state     │    │  • Execute actions           │
│  • Track progress    │    │  • Render screen             │
│  • Detect events     │    │  • Provide game state        │
└──────────────────────┘    └──────────────────────────────┘
                                     │
                                     ▼
                            ┌──────────────────────┐
                            │   POKEMON RED ROM    │
                            ├──────────────────────┤
                            │  roms/PokemonRed.gb  │
                            └──────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    RL ALGORITHM (PPO)                        │
├─────────────────────────────────────────────────────────────┤
│              Stable-Baselines3 PPO Agent                     │
│  • CNN Policy Network                                        │
│  • Actor-Critic Architecture                                 │
│  • Experience Collection                                     │
│  • Policy Optimization                                       │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUTS & LOGGING                          │
├─────────────────────────────────────────────────────────────┤
│  • models/ - Saved checkpoints                               │
│  • logs/ - TensorBoard training logs                         │
│  • Console output - Training statistics                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Training Loop Flow

```
START TRAINING
    ↓
┌───────────────────────────────────────────────────────┐
│ 1. INITIALIZE                                         │
│    • Load ROM                                         │
│    • Create environment                               │
│    • Initialize PPO agent                             │
│    • Setup logging                                    │
└───────────────┬───────────────────────────────────────┘
                ↓
        ┌───────────────┐
        │ 2. RESET ENV  │
        │   • Load game │
        │   • Get obs   │
        └───────┬───────┘
                ↓
        ┌────────────────────────────────────────┐
        │ 3. COLLECT EXPERIENCES                 │
        │    For N_STEPS (2048):                 │
        │    ┌──────────────────────────────┐    │
        │    │ a. Agent observes screen     │    │
        │    │ b. Neural net predicts action│    │
        │    │ c. Execute action in game    │    │
        │    │ d. Game state changes        │    │
        │    │ e. Calculate reward          │    │
        │    │ f. Store experience          │    │
        │    └──────────────────────────────┘    │
        └───────┬────────────────────────────────┘
                ↓
        ┌────────────────────────────────────────┐
        │ 4. UPDATE POLICY                       │
        │    • Compute advantages (GAE)          │
        │    • Run PPO optimization              │
        │    • Update neural network weights     │
        │    • Clip policy changes               │
        └───────┬────────────────────────────────┘
                ↓
        ┌────────────────────────────────────────┐
        │ 5. LOG & SAVE                          │
        │    • Log metrics to TensorBoard        │
        │    • Print statistics                  │
        │    • Save checkpoint (every 50k steps) │
        └───────┬────────────────────────────────┘
                ↓
                │
        ┌───────┴────────┐
        │  Done training?│
        │  (max steps)   │
        └───────┬────────┘
                │
        ┌───────┴────────┐
       YES              NO
        │                │
        ↓                └──────> Back to step 3
    FINISH
    Save final model
```

---

## 🧠 Neural Network Architecture

```
INPUT (Game Screen)
    84 x 84 x 4 (grayscale, 4 frames stacked)
        ↓
┌────────────────────────────────────────┐
│  CONVOLUTIONAL LAYER 1                 │
│  • 32 filters, 8x8 kernel, stride 4    │
│  • ReLU activation                     │
│  • Output: 20 x 20 x 32                │
└────────────────┬───────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│  CONVOLUTIONAL LAYER 2                 │
│  • 64 filters, 4x4 kernel, stride 2    │
│  • ReLU activation                     │
│  • Output: 9 x 9 x 64                  │
└────────────────┬───────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│  CONVOLUTIONAL LAYER 3                 │
│  • 64 filters, 3x3 kernel, stride 1    │
│  • ReLU activation                     │
│  • Output: 7 x 7 x 64                  │
└────────────────┬───────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│  FLATTEN                               │
│  • Output: 3136 features               │
└────────────────┬───────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│  FULLY CONNECTED LAYER                 │
│  • 512 units                           │
│  • ReLU activation                     │
└────────────────┬───────────────────────┘
                 ↓
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│  ACTOR HEAD  │  │   CRITIC HEAD    │
│  (Policy)    │  │   (Value)        │
│              │  │                  │
│  9 actions   │  │  1 scalar value  │
│  (softmax)   │  │  (state value)   │
└──────────────┘  └──────────────────┘
       ↓                  ↓
   ACTION            VALUE ESTIMATE
 (what to do)     (how good is state)
```

---

## 🎮 Environment Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    GAME STATE                                │
│                  (Pokemon Red ROM)                           │
│  • Player position (X, Y)                                    │
│  • Current map ID                                            │
│  • Party Pokemon count                                       │
│  • Battle state                                              │
│  • HP values                                                 │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├──────────────────┬────────────────────────────────────┐
       ↓                  ↓                                    ↓
┌─────────────┐   ┌──────────────┐                    ┌──────────────┐
│  SCREEN     │   │  MEMORY      │                    │   REWARD     │
│  CAPTURE    │   │  READING     │                    │  CALCULATION │
└──────┬──────┘   └──────┬───────┘                    └──────┬───────┘
       │                 │                                    │
       ↓                 ↓                                    ↓
┌──────────────┐  ┌──────────────────┐              ┌─────────────────┐
│  Processing  │  │  State Tracking  │              │ Reward Shaping  │
│  • RGB→Gray  │  │  • Position Δ    │              │ • Step penalty  │
│  • Resize    │  │  • Map changes   │              │ • Grass bonus   │
│  • 84x84     │  │  • Battle detect │              │ • Battle bonus  │
│  • Stack x4  │  │  • Pokemon count │              │ • Catch reward  │
└──────┬───────┘  └──────┬───────────┘              └─────┬───────────┘
       │                 │                                 │
       │                 │                                 │
       └────────┬────────┴─────────────┬───────────────────┘
                ↓                      ↓
        ┌────────────────┐     ┌──────────────┐
        │  OBSERVATION   │     │    REWARD    │
        │  (to agent)    │     │  (to agent)  │
        └────────────────┘     └──────────────┘
```

---

## 📦 Module Responsibilities

### **config.py**
```
┌──────────────────────────────────┐
│  CONFIGURATION MODULE            │
├──────────────────────────────────┤
│  Defines:                        │
│  • Training hyperparameters      │
│  • Environment settings          │
│  • Reward values                 │
│  • Memory addresses              │
│  • File paths                    │
└──────────────────────────────────┘
```

### **memory_reader.py**
```
┌──────────────────────────────────┐
│  MEMORY READER MODULE            │
├──────────────────────────────────┤
│  PokemonRedMemory class:         │
│  • read_byte()                   │
│  • read_word()                   │
│  • get_player_position()         │
│  • get_map_id()                  │
│  • get_party_count()             │
│  • is_in_battle()                │
│  • get_first_pokemon_hp()        │
│  • in_grass_area()               │
│  • get_game_state_hash()         │
└──────────────────────────────────┘
```

### **pokemon_env.py**
```
┌──────────────────────────────────┐
│  ENVIRONMENT MODULE              │
├──────────────────────────────────┤
│  PokemonRedEnv class:            │
│  • __init__()                    │
│  • reset()                       │
│  • step(action)                  │
│  • _get_observation()            │
│  • _calculate_reward()           │
│  • _is_terminated()              │
│  • render()                      │
│                                  │
│  PokemonRedWrapper class:        │
│  • Frame stacking                │
│  • Additional preprocessing      │
└──────────────────────────────────┘
```

### **train.py**
```
┌──────────────────────────────────┐
│  TRAINING MODULE                 │
├──────────────────────────────────┤
│  Functions:                      │
│  • train()                       │
│  • make_env()                    │
│  • main()                        │
│                                  │
│  TrainingCallback class:         │
│  • Log statistics                │
│  • Save checkpoints              │
│  • Print progress                │
└──────────────────────────────────┘
```

### **play.py**
```
┌──────────────────────────────────┐
│  INFERENCE MODULE                │
├──────────────────────────────────┤
│  Functions:                      │
│  • play_pokemon()                │
│  • main()                        │
│                                  │
│  Purpose:                        │
│  • Load trained model            │
│  • Run episodes                  │
│  • Display gameplay              │
│  • Collect statistics            │
└──────────────────────────────────┘
```

---

## 🔄 Action → Reward Cycle

```
Agent State: Looking at screen
       ↓
┌──────────────────┐
│  DECIDE ACTION   │
│  Neural Net:     │
│  Screen → Action │
│  (e.g., "right") │
└────────┬─────────┘
         ↓
┌─────────────────────┐
│  EXECUTE IN GAME    │
│  PyBoy:             │
│  Press right button │
│  Advance 4 frames   │
└────────┬────────────┘
         ↓
┌─────────────────────────────┐
│  READ NEW STATE             │
│  Memory Reader:             │
│  • Player moved right       │
│  • Position: (5, 10) → (6, 10)│
│  • Still in same map        │
│  • No battle started        │
└────────┬────────────────────┘
         ↓
┌────────────────────────────┐
│  CALCULATE REWARD          │
│  Reward Function:          │
│  • Step penalty: -0.01     │
│  • New position: +0.1      │
│  • Total: +0.09            │
└────────┬───────────────────┘
         ↓
┌────────────────────────────┐
│  LEARN                     │
│  PPO Algorithm:            │
│  • Store experience        │
│  • Update policy (later)   │
│  • Improve behavior        │
└────────────────────────────┘
         ↓
    REPEAT FOR NEXT STEP
```

---

## 📁 File System Interaction

```
Program Execution
    ↓
┌─────────────────────────────────────────┐
│  READ                                   │
│  • config.py (settings)                 │
│  • roms/PokemonRed.gb (game)            │
│  • models/*.zip (if resuming)           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  WRITE DURING TRAINING                  │
│  • models/*.zip (checkpoints)           │
│  • logs/* (TensorBoard data)            │
│  • Console output                       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  OPTIONAL                               │
│  • saves/*.state (save states)          │
│  • *.png (screenshots)                  │
│  • *.csv (exported data)                │
└─────────────────────────────────────────┘
```

---

## 🎯 Component Interaction Example

**Scenario: Agent catches first Pokemon**

```
1. Agent sees screen
   pokemon_env.py: _get_observation()
   → Returns: 84x84x4 array

2. Agent decides action (press A)
   Stable-Baselines3 PPO: predict()
   → Returns: action = 5 (A button)

3. Action executed
   pokemon_env.py: step(5)
   → Calls: _take_action(5)
   → PyBoy: Presses A button

4. Game advances
   PyBoy: tick() x 4 frames
   → Pokemon Red: Processes button press
   → Game: Pokeball thrown, Pokemon caught!

5. Memory read
   memory_reader.py: get_party_count()
   → Reads RAM address 0xD163
   → Returns: 1 (was 0 before)

6. Reward calculated
   pokemon_env.py: _calculate_reward()
   → Detects: party_count increased
   → Returns: +100.0 (CATCH_POKEMON reward)

7. Episode ends
   pokemon_env.py: _is_terminated()
   → Returns: True (goal achieved!)

8. Logged and saved
   train.py: TrainingCallback
   → Logs to TensorBoard
   → Prints success message
   → Saves checkpoint
```

---

## 💡 Design Patterns Used

### **1. Environment Pattern (OpenAI Gym)**
- Standardized RL interface
- reset(), step(), render()
- Observation/action spaces

### **2. Callback Pattern**
- TrainingCallback for checkpoints
- Hooks into training loop
- Logging and saving logic

### **3. Wrapper Pattern**
- PokemonRedWrapper
- Adds preprocessing to base environment
- Frame stacking

### **4. Configuration Pattern**
- Central config.py file
- All settings in one place
- Easy to modify and experiment

### **5. Facade Pattern**
- memory_reader.py
- Simplifies RAM access
- High-level game state API

---

## 🚀 Performance Considerations

### **Bottlenecks**
```
Slowest: Emulation (PyBoy)
  ↓
Medium: Neural network forward pass
  ↓
Fastest: Reward calculation
```

### **Optimization Strategies**
```
1. Frame skipping (4 frames)
   → Reduces computation by 75%

2. Headless mode
   → No rendering overhead
   → 2-3x faster training

3. GPU acceleration
   → PyTorch CUDA
   → 5-10x faster neural net

4. Parallel environments (not implemented)
   → Could use VecEnv
   → N environments in parallel
```

---

## 📚 Summary

This architecture provides:
- **Modularity**: Each component has clear responsibility
- **Extensibility**: Easy to modify rewards, observations, actions
- **Debuggability**: Clear data flow, logging at each step
- **Maintainability**: Well-documented, standard patterns

Key Innovation:
- **Reward Shaping**: Guides agent toward complex goal
- **Memory Reading**: Direct game state access
- **Frame Stacking**: Temporal information from images

---

<div align="center">

**Understanding leads to better customization!** 🔧

See **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** for implementation details.

</div>

