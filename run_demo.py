"""
Interactive demo script to help users get started
"""

import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_setup():
    """Check if setup is complete"""
    print_header("CHECKING SETUP")
    
    issues = []
    
    # Check Python version
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        issues.append("Python 3.10 or higher recommended")
    
    # Check if ROM exists
    if os.path.exists("roms/PokemonRed.gb"):
        print("✓ Pokemon Red ROM found")
    else:
        print("✗ Pokemon Red ROM not found")
        issues.append("Place PokemonRed.gb in roms/ directory")
    
    # Check if dependencies are installed
    try:
        import numpy
        import torch
        import gymnasium
        import stable_baselines3
        import pyboy
        import cv2
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e.name}")
        issues.append("Run: pip install -r requirements.txt")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ Setup looks good!")
        return True


def show_menu():
    """Show interactive menu"""
    print_header("POKEMON RED RL - INTERACTIVE DEMO")
    
    print("What would you like to do?\n")
    print("1. Test environment (verify everything works)")
    print("2. Start training (begin teaching the AI)")
    print("3. Watch trained agent play")
    print("4. View training progress (TensorBoard)")
    print("5. Show configuration")
    print("6. Quick start guide")
    print("0. Exit")
    print()


def run_test_env():
    """Run environment test"""
    print_header("TESTING ENVIRONMENT")
    print("This will:")
    print("  - Open the game window")
    print("  - Execute 1000 random actions")
    print("  - Display game state information")
    print("\nPress Enter to continue, or Ctrl+C to cancel...")
    try:
        input()
        subprocess.run([sys.executable, "test_env.py"])
    except KeyboardInterrupt:
        print("\nCancelled.")


def run_training():
    """Start training"""
    print_header("START TRAINING")
    print("This will train an AI agent to play Pokemon Red.")
    print("\nOptions:")
    print("  1. Quick training (100k steps, ~30-60 min)")
    print("  2. Standard training (1M steps, ~4-8 hours)")
    print("  3. Extended training (2M steps, ~8-16 hours)")
    print("  4. Custom")
    print("  0. Back")
    
    choice = input("\nChoice: ").strip()
    
    cmd = [sys.executable, "train.py"]
    
    if choice == "1":
        cmd.extend(["--timesteps", "100000"])
    elif choice == "2":
        cmd.extend(["--timesteps", "1000000"])
    elif choice == "3":
        cmd.extend(["--timesteps", "2000000"])
    elif choice == "4":
        steps = input("Enter number of timesteps: ").strip()
        cmd.extend(["--timesteps", steps])
    elif choice == "0":
        return
    else:
        print("Invalid choice.")
        return
    
    headless = input("Run without rendering (faster)? (y/n): ").strip().lower()
    if headless == 'y':
        cmd.append("--headless")
    
    print("\nStarting training...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTraining stopped.")


def run_playback():
    """Watch trained agent"""
    print_header("WATCH TRAINED AGENT")
    
    # Find models
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith(".zip")]
        if models:
            print("Available models:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            print()
            
            choice = input("Select model number (or path): ").strip()
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    model_path = os.path.join("models", models[idx])
                else:
                    print("Invalid selection.")
                    return
            except ValueError:
                model_path = choice
            
            episodes = input("Number of episodes to watch (default 5): ").strip()
            episodes = episodes if episodes else "5"
            
            cmd = [
                sys.executable, "play.py",
                "--model", model_path,
                "--episodes", episodes
            ]
            
            print("\nStarting playback...")
            subprocess.run(cmd)
        else:
            print("No trained models found.")
            print("Train a model first using option 2.")
    else:
        print("No models directory found.")
        print("Train a model first using option 2.")


def run_tensorboard():
    """Launch TensorBoard"""
    print_header("TENSORBOARD VISUALIZATION")
    print("This will launch TensorBoard to visualize training progress.")
    print("\nOnce launched:")
    print("  1. Open your browser")
    print("  2. Go to: http://localhost:6006")
    print("  3. Press Ctrl+C in terminal to stop TensorBoard")
    print("\nPress Enter to continue...")
    try:
        input()
        subprocess.run(["tensorboard", "--logdir=logs"])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("\nTensorBoard not found. Is it installed?")


def show_config():
    """Display configuration"""
    print_header("CURRENT CONFIGURATION")
    
    try:
        from config import *
        print(f"ROM Path:              {ROM_PATH}")
        print(f"Total Timesteps:       {TOTAL_TIMESTEPS:,}")
        print(f"Learning Rate:         {LEARNING_RATE}")
        print(f"Steps per Update:      {N_STEPS}")
        print(f"Batch Size:            {BATCH_SIZE}")
        print(f"Max Episode Steps:     {MAX_STEPS_PER_EPISODE}")
        print(f"\nRewards:")
        for key, value in REWARDS.items():
            print(f"  {key:20s}: {value}")
        print("\nTo modify, edit config.py")
    except Exception as e:
        print(f"Error loading config: {e}")


def show_quickstart():
    """Show quick start guide"""
    print_header("QUICK START GUIDE")
    print("""
1. SETUP
   - Ensure PokemonRed.gb is in roms/ directory
   - Install dependencies: pip install -r requirements.txt

2. TEST
   - Run: python test_env.py
   - Should see game window with random actions

3. TRAIN
   - Run: python train.py
   - Watch the agent learn in real-time
   - Checkpoints saved every 50k steps
   - Use Ctrl+C to stop gracefully

4. WATCH
   - Run: python play.py --model models/your_model.zip
   - Watch your trained agent play!

5. MONITOR
   - Run: tensorboard --logdir=logs
   - Open browser to http://localhost:6006
   - See training metrics and graphs

TIPS:
- Use --headless for faster training
- Training takes 4-8 hours for 1M steps
- Check README.md for detailed information
- Check QUICKSTART.md for more examples
    """)
    input("\nPress Enter to continue...")


def main():
    """Main demo loop"""
    # Check setup first
    if not check_setup():
        print("\n⚠️  Please fix the issues above before continuing.")
        sys.exit(1)
    
    while True:
        try:
            show_menu()
            choice = input("Choice: ").strip()
            
            if choice == "1":
                run_test_env()
            elif choice == "2":
                run_training()
            elif choice == "3":
                run_playback()
            elif choice == "4":
                run_tensorboard()
            elif choice == "5":
                show_config()
            elif choice == "6":
                show_quickstart()
            elif choice == "0":
                print("\nGoodbye! Happy training! 🎮")
                break
            else:
                print("\nInvalid choice. Please try again.")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()

