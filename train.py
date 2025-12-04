import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import os
from datetime import datetime
from utils import visualize_predictions

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-3,
        weight_decay=1e-5
    ):
        """       
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/validation data
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: Weight decay for AdamW optimizer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Model size
        self.model_size = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model initialized on {device}")
        print(f"Total parameters: {self.model_size:,}")
        print(f"Trainable parameters: {self.trainable_params:,}")
    
    def _train_epoch(self, epoch, num_epochs):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
            leave=False
        )
        
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return epoch_loss / len(self.train_loader)
    
    def _evaluate(self, epoch, num_epochs):
        """Evaluate on test set."""
        self.model.eval()
        test_loss = 0.0
        mae_total = 0.0
        
        pbar = tqdm(
            self.test_loader,
            desc=f'Epoch {epoch+1}/{num_epochs} [Val]',
            leave=False
        )
        
        with torch.no_grad():
            for x_batch, y_batch in pbar:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(x_batch)
                
                # MSE loss
                loss = self.criterion(predictions, y_batch)
                test_loss += loss.item()
                
                # MAE
                mae = torch.abs(predictions - y_batch).mean()
                mae_total += mae.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_test_loss = test_loss / len(self.test_loader)
        avg_mae = mae_total / len(self.test_loader)
        
        return avg_test_loss, avg_mae
    
    def train(
        self,
        num_epochs=10,
        use_scheduler=True,
        scheduler_factor=0.5,
        scheduler_patience=3,
        save_dir='./experiments',
        save_metrics=True,
        experiment_name=None,
        results_file=None,
        append_results=True,
        plot_predictions=True,
        num_plot_samples=5,
        num_plot_variates=4
    ):
        """       
        Args:
            num_epochs: Number of epochs to train
            use_scheduler: Whether to use ReduceLROnPlateau scheduler
            scheduler_factor: Factor by which to reduce LR (default: 0.5)
            scheduler_patience: Number of epochs with no improvement before reducing LR (default: 3)
            save_dir: Directory to save model and metrics
            save_metrics: Whether to save training metrics as JSON
            experiment_name: Optional name for the experiment (used in filenames)
            results_file: Optional filename to save/append results (e.g., 'all_experiments.json')
                         If None, saves to individual files with timestamps
            append_results: If True and results_file exists, append to it. If False, overwrite.
            plot_predictions: Whether to plot prediction examples at the end
            num_plot_samples: Number of samples to plot (default: 5)
            num_plot_variates: Number of variates to plot per sample (default: 4)
        
        Returns:
            dict: Training history with losses and metrics
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize scheduler if requested
        scheduler = None
        if use_scheduler:
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience
            )
        
        # Training history
        history = {
            'train_losses': [],
            'test_losses': [],
            'test_maes': [],
            'learning_rates': []
        }
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n{'='*60}")
        print(f"Starting training: {num_epochs} epochs")
        print(f"Scheduler: {'Enabled' if use_scheduler else 'Disabled'}")
        print(f"{'='*60}\n")
        
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(epoch, num_epochs)
            history['train_losses'].append(train_loss)
            
            # Evaluate
            test_loss, test_mae = self._evaluate(epoch, num_epochs)
            history['test_losses'].append(test_loss)
            history['test_maes'].append(test_mae)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step(test_loss)
            
            
            # Print summary every 5 epochs or at the end
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {test_loss:.6f}")
                print(f"  Val MAE:    {test_mae:.6f}")
                print(f"  LR:         {current_lr:.2e}\n")
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}\n")
        
        # Save model
        model_filename = f"model_{timestamp}.pth"
        if experiment_name:
            model_filename = f"model_{experiment_name}_{timestamp}.pth"
        
        model_path = os.path.join(save_dir, model_filename)
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'final_train_loss': history['train_losses'][-1],
            'final_test_loss': history['test_losses'][-1],
            'final_test_mae': history['test_maes'][-1],
            'history': history
        }, model_path)
        
        print(f"Model saved: {model_path}")
        
        # Save metrics as JSON
        if save_metrics:
            metrics = {
                'experiment_name': experiment_name or 'default',
                'timestamp': timestamp,
                'model_file': model_filename,
                'training_config': {
                    'num_epochs': num_epochs,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                    'use_scheduler': use_scheduler,
                    'scheduler_factor': scheduler_factor if use_scheduler else None,
                    'scheduler_patience': scheduler_patience if use_scheduler else None,
                    'batch_size': self.train_loader.batch_size,
                },
                'model_info': {
                    'total_parameters': self.model_size,
                    'trainable_parameters': self.trainable_params,
                },
                'final_metrics': {
                    'train_loss_mse': round(history['train_losses'][-1], 6),
                    'test_loss_mse': round(history['test_losses'][-1], 6),
                    'test_mae': round(history['test_maes'][-1], 6),
                    'final_lr': history['learning_rates'][-1]
                },
                'best_metrics': {
                    'best_train_loss': round(min(history['train_losses']), 6),
                    'best_test_loss': round(min(history['test_losses']), 6),
                    'best_test_mae': round(min(history['test_maes']), 6),
                    'best_epoch_train': history['train_losses'].index(min(history['train_losses'])) + 1,
                    'best_epoch_test': history['test_losses'].index(min(history['test_losses'])) + 1,
                },
                'training_history': {
                    'train_losses': [round(x, 6) for x in history['train_losses']],
                    'test_losses': [round(x, 6) for x in history['test_losses']],
                    'test_maes': [round(x, 6) for x in history['test_maes']],
                    'learning_rates': history['learning_rates']
                }
            }
            
            # Handle results file (append or create new)
            if results_file:
                results_path = os.path.join(save_dir, results_file)
                
                # Load existing results if appending
                if append_results and os.path.exists(results_path):
                    try:
                        with open(results_path, 'r') as f:
                            all_results = json.load(f)
                        
                        # Ensure it's a list
                        if not isinstance(all_results, list):
                            all_results = [all_results]
                        
                        all_results.append(metrics)
                        print(f"Appending to existing results file ({len(all_results)} experiments total)")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {results_file}, creating new file")
                        all_results = [metrics]
                else:
                    all_results = [metrics]
                
                # Save all results
                with open(results_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"Results saved: {results_path}")
            
            # Also save individual metrics file with timestamp
            else:
                metrics_filename = f"metrics_{timestamp}.json"
                if experiment_name:
                    metrics_filename = f"metrics_{experiment_name}_{timestamp}.json"
                
                metrics_path = os.path.join(save_dir, metrics_filename)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"Metrics saved: {metrics_path}")
        
        # Print final summary
        print(f"\nFinal Results:")
        print(f"  Train Loss (MSE): {history['train_losses'][-1]:.6f}")
        print(f"  Test Loss (MSE):  {history['test_losses'][-1]:.6f}")
        print(f"  Test MAE:         {history['test_maes'][-1]:.6f}")
        print(f"  Best Test Loss:   {min(history['test_losses']):.6f} (epoch {history['test_losses'].index(min(history['test_losses'])) + 1})")
        print(f"  Best Test MAE:    {min(history['test_maes']):.6f} (epoch {history['test_maes'].index(min(history['test_maes'])) + 1})")
        
        # Plot predictions
        if plot_predictions:
            print(f"\nGenerating prediction plots...")
            visualize_predictions(
                self.model,
                self.test_loader,
                self.device,
                num_samples=num_plot_samples,
                num_variates=num_plot_variates,
                save_path=os.path.join(save_dir, f"predictions_{timestamp}.png")
            )
        
        return history
