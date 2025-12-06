import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import json
import glob
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=48, forecast=12):
        """
        Args:
            data: numpy array of shape [timesteps, num_variates]
            lookback: number of past timesteps to use
            forecast: number of future timesteps to predict
        """
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.forecast = forecast

    def __len__(self):
        return len(self.data) - self.lookback - self.forecast + 1

    def __getitem__(self, idx):
        # Input: [lookback, num_variates]
        x = self.data[idx:idx + self.lookback]
        # Target: [forecast, num_variates]
        y = self.data[idx + self.lookback:idx + self.lookback + self.forecast]
        return x, y
    

def visualize_predictions(
    model,
    data_loader,
    device,
    num_samples=5,
    num_variates=4,
    save_path=None
):
    """
    Visualize model predictions on test data.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        device: Device to run inference on
        num_samples: Number of random samples to plot
        num_variates: Number of variates to show per sample
        save_path: Optional path to save the figure
    """
    model.eval()
    
    # Get a batch of test data
    try:
        x_batch, y_batch = next(iter(data_loader))
    except StopIteration:
        print("Warning: Could not get test batch for visualization")
        return
    
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(x_batch)
    
    # Move to CPU for plotting
    x_batch = x_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Determine dimensions
    batch_size = x_batch.shape[0]
    lookback_len = x_batch.shape[1]  # T
    forecast_len = y_batch.shape[1]  # S
    total_variates = x_batch.shape[2]  # N
    
    # Randomly sample from batch
    num_samples = min(num_samples, batch_size)
    sample_indices = np.random.choice(batch_size, num_samples, replace=False)
    
    # Limit number of variates to plot
    num_variates = min(num_variates, total_variates)
    variate_indices = np.linspace(0, total_variates - 1, num_variates, dtype=int)
    
    # Create subplot grid
    fig, axes = plt.subplots(num_samples, num_variates, figsize=(4*num_variates, 3*num_samples))
    
    # Handle single row or column cases
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_variates == 1:
        axes = axes.reshape(-1, 1)
    
    for i, sample_idx in enumerate(sample_indices):
        for j, var_idx in enumerate(variate_indices):
            ax = axes[i, j]
            
            # Extract data for this sample and variate
            x_history = x_batch[sample_idx, :, var_idx]
            y_truth = y_batch[sample_idx, :, var_idx]
            y_pred = predictions[sample_idx, :, var_idx]
            
            # Time indices
            history_time = np.arange(lookback_len)
            # Extend forecast time to include the connection point
            forecast_time_extended = np.arange(lookback_len - 1, lookback_len + forecast_len)

            # Plot history
            ax.plot(history_time, x_history, 
                label='History', color='blue', alpha=0.7, linewidth=1.5)

            # Connect forecast to history by prepending the last history point
            last_history_point = x_history[-1]
            y_truth_connected = np.concatenate([[last_history_point], y_truth])
            y_pred_connected = np.concatenate([[last_history_point], y_pred])

            # Plot ground truth (connected)
            ax.plot(forecast_time_extended, y_truth_connected,
                label='Ground Truth', color='green', linewidth=2, 
                marker='o', markersize=3, alpha=0.8)

            # Plot predictions (connected)
            ax.plot(forecast_time_extended, y_pred_connected,
                label='Prediction', color='red', linestyle='--', linewidth=2,
                marker='x', markersize=4, alpha=0.8)            
            # Add vertical separator
            ax.axvline(x=lookback_len, color='black', linestyle=':', 
                      alpha=0.5, linewidth=1)
            
            # Labels and styling
            if i == 0:
                ax.set_title(f'Variate {var_idx}', fontweight='bold', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'Sample {sample_idx}\nValue', fontsize=9)
            if i == num_samples - 1:
                ax.set_xlabel('Timestep', fontsize=9)
            
            ax.grid(True, alpha=0.3)
            
            # Only show legend on first subplot
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='best')
    
    plt.suptitle(f'Model Predictions vs Ground Truth\n({num_samples} samples, {num_variates} variates)', 
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_history(file_name, experiment_name=None, experiment_dir='./experiments',
                          save_path=None, show=True):
    """
    Plot training history from a JSON file.
    
    Args:
        file_name: Name of JSON file (e.g., 'all_experiments.json' or 'metrics_baseline_20241204.json')
        experiment_name: Optional specific experiment name to plot (if file has multiple)
        experiment_dir: Directory with experiments (default: './experiments')
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    filepath = os.path.join(experiment_dir, file_name)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle list vs single experiment
    if isinstance(data, list):
        if experiment_name:
            # Find matching experiment
            metrics = next((m for m in data if m.get('experiment_name') == experiment_name), None)
            if metrics is None:
                raise ValueError(f"Experiment '{experiment_name}' not found in {file_name}")
        else:
            # Use most recent (last in list)
            metrics = data[-1]
            print(f"Plotting most recent experiment: {metrics.get('experiment_name', 'N/A')}")
    else:
        metrics = data
    
    history = metrics['training_history']
    exp_name = metrics.get('experiment_name', 'Experiment')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot 1: Train vs Test Loss
    axes[0].plot(epochs, history['train_losses'], 
                label='Train Loss', color='blue', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs, history['test_losses'], 
                label='Test Loss', color='red', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('MSE Loss', fontweight='bold')
    axes[0].set_title('Training vs Test Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Test MAE
    axes[1].plot(epochs, history['test_maes'], 
                color='green', linewidth=2, marker='d', markersize=4)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('MAE', fontweight='bold')
    axes[1].set_title('Test MAE over Time', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 
                color='purple', linewidth=2, marker='^', markersize=4)
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{exp_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


