"""
Visualize training progress from TensorBoard logs
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob


def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files"""
    # Find event files
    event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Load most recent event file
    event_file = max(event_files, key=os.path.getctime)
    print(f"Loading: {event_file}")
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Get available tags
    print(f"Available tags: {ea.Tags()['scalars']}")
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': e.step, 'value': e.value} for e in events
        ])
    
    return data


def plot_training_progress(data, save_path=None):
    """Create visualization of training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pokemon Red RL Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Reward
    if 'rollout/ep_rew_mean' in data:
        ax = axes[0, 0]
        df = data['rollout/ep_rew_mean']
        ax.plot(df['step'], df['value'], linewidth=2, color='#3498db')
        ax.set_title('Mean Episode Reward', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    if 'rollout/ep_len_mean' in data:
        ax = axes[0, 1]
        df = data['rollout/ep_len_mean']
        ax.plot(df['step'], df['value'], linewidth=2, color='#e74c3c')
        ax.set_title('Mean Episode Length', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    if 'train/learning_rate' in data:
        ax = axes[1, 0]
        df = data['train/learning_rate']
        ax.plot(df['step'], df['value'], linewidth=2, color='#2ecc71')
        ax.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss
    if 'train/loss' in data:
        ax = axes[1, 1]
        df = data['train/loss']
        ax.plot(df['step'], df['value'], linewidth=2, color='#f39c12')
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def print_statistics(data):
    """Print training statistics"""
    print("\n" + "="*70)
    print("TRAINING STATISTICS")
    print("="*70)
    
    if 'rollout/ep_rew_mean' in data:
        df = data['rollout/ep_rew_mean']
        print(f"\nEpisode Reward:")
        print(f"  Initial: {df['value'].iloc[0]:.2f}")
        print(f"  Final:   {df['value'].iloc[-1]:.2f}")
        print(f"  Max:     {df['value'].max():.2f}")
        print(f"  Min:     {df['value'].min():.2f}")
    
    if 'rollout/ep_len_mean' in data:
        df = data['rollout/ep_len_mean']
        print(f"\nEpisode Length:")
        print(f"  Initial: {df['value'].iloc[0]:.2f}")
        print(f"  Final:   {df['value'].iloc[-1]:.2f}")
        print(f"  Max:     {df['value'].max():.2f}")
        print(f"  Min:     {df['value'].min():.2f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Pokemon Red RL training progress"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save plot image (if not specified, displays interactively)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading TensorBoard data...")
    data = load_tensorboard_data(args.logdir)
    
    if data is None:
        print("\nNo data found. Make sure you've started training and have log files.")
        print(f"Expected location: {args.logdir}")
        return
    
    # Print statistics
    print_statistics(data)
    
    # Create plots
    print("\nGenerating plots...")
    plot_training_progress(data, args.save)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

