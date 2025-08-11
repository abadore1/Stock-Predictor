"""
Training script for stock prediction transformer model
Includes early stopping, checkpointing, and comprehensive metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import config
from data_handler import StockDataHandler
from dataset import create_data_splits, create_data_loaders
from model import create_model


class EarlyStopping:
    """
    Early stopping to avoid overfitting
    """
    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE, 
                 min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        
    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class StockPredictionTrainer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_SAVE_PATH, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        """
        # Regression metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Directional accuracy (trading-specific metric)
        pred_direction = np.diff(predictions) > 0
        target_direction = np.diff(targets) > 0
        directional_accuracy = np.mean(pred_direction == target_direction)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'r2': r2
        }
    
    def evaluate_model(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation/test data
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(data_loader)
        metrics = self.calculate_metrics(np.array(all_predictions), np.array(all_targets))
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, "latest_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (sequences, targets) in enumerate(pbar):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.6f}'
            })
        
        return total_loss / len(train_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List]:
        """
        Complete training loop
        """
        print(f"Starting training for {config.NUM_EPOCHS} epochs...")
        print(f"Model has {self.model.count_parameters():,} parameters")
        
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        best_val_loss = np.inf
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.evaluate_model(val_loader)
            self.val_losses.append(val_loss)
            self.metrics_history.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"Val MAE: {val_metrics['mae']:.6f}")
            print(f"Val Directional Accuracy: {val_metrics['directional_accuracy']:.4f}")
            print(f"Val R²: {val_metrics['r2']:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("\nTraining completed!")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        history_path = os.path.join(config.RESULTS_SAVE_PATH, "training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {}
            for key, value in history.items():
                if key == 'metrics_history':
                    history_serializable[key] = [
                        {k: float(v) for k, v in metrics.items()} 
                        for metrics in value
                    ]
                else:
                    history_serializable[key] = [float(x) for x in value]
            
            json.dump(history_serializable, f, indent=2)
        
        print(f"Saved training history to {history_path}")
        
        return history


def main():
    """
    Main training pipeline
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    handler = StockDataHandler()
    
    # Try to load existing data, otherwise prepare new data
    try:
        data = handler.load_data()
        print("Loaded existing processed data")
    except FileNotFoundError:
        print("No existing data found, preparing new data...")
        data = handler.prepare_data()
    
    # Create data splits and loaders
    train_data, val_data, test_data = create_data_splits(data)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)
    
    # Get input dimension from dataset
    input_dim = train_loader.dataset.get_feature_dim()
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = create_model(input_dim)
    
    # Create trainer
    trainer = StockPredictionTrainer(model, device)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.evaluate_model(test_loader)
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }
    
    test_results_path = os.path.join(config.RESULTS_SAVE_PATH, "test_results.json")
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Saved test results to {test_results_path}")


if __name__ == "__main__":
    main()