"""
PyTorch Dataset implementation for time series stock prediction
Uses sliding window approach to create training samples
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List
import config


class StockTimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int = config.SEQUENCE_LENGTH,
                 prediction_horizon: int = config.PREDICTION_HORIZON):
        """
        Initialize dataset with sliding window approach
        
        Args:
            data: Processed stock data with both target and market features
            sequence_length: Number of time steps to use as input
            prediction_horizon: Number of steps ahead to predict
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Define feature columns
        target_features = config.FEATURE_COLUMNS
        market_features = [f'Market_{col}' for col in config.FEATURE_COLUMNS]
        self.feature_columns = target_features + market_features
        
        # Create sliding windows
        self.sequences, self.targets = self._create_sequences()
        
        print(f"Created {len(self.sequences)} sequences from {len(data)} data points")
        
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences and targets
        """
        sequences = []
        targets = []
        
        data_values = self.data[self.feature_columns].values
        target_values = self.data['Close'].values
        
        for i in range(len(data_values) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            sequence = data_values[i:i + self.sequence_length]
            sequences.append(sequence)
            
            # Target (next close price)
            target = target_values[i + self.sequence_length + self.prediction_horizon - 1]
            targets.append(target)
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        
        return sequence, target
    
    def get_feature_dim(self) -> int:
        """
        Get the number of input features
        """
        return len(self.feature_columns)


def create_data_splits(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets
    """
    # Sort data by timestamp
    data_sorted = data.sort_index()
    
    total_len = len(data_sorted)
    test_len = int(total_len * config.TEST_SPLIT)
    val_len = int(total_len * config.VALIDATION_SPLIT)
    train_len = total_len - test_len - val_len
    
    # Split chronologically (important for time series)
    train_data = data_sorted.iloc[:train_len]
    val_data = data_sorted.iloc[train_len:train_len + val_len]
    test_data = data_sorted.iloc[train_len + val_len:]
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def create_data_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                       test_data: pd.DataFrame, batch_size: int = config.BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training
    """
    # Create datasets
    train_dataset = StockTimeSeriesDataset(train_data)
    val_dataset = StockTimeSeriesDataset(val_data)
    test_dataset = StockTimeSeriesDataset(test_data)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle training data
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Created data loaders - Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def get_sequence_for_prediction(data: pd.DataFrame, sequence_length: int = config.SEQUENCE_LENGTH) -> torch.Tensor:
    """
    Get the most recent sequence for real-time prediction
    """
    if len(data) < sequence_length:
        raise ValueError(f"Not enough data. Need {sequence_length}, got {len(data)}")
    
    # Get feature columns
    target_features = config.FEATURE_COLUMNS
    market_features = [f'Market_{col}' for col in config.FEATURE_COLUMNS]
    feature_columns = target_features + market_features
    
    # Get the last sequence_length rows
    recent_data = data[feature_columns].iloc[-sequence_length:].values
    
    # Convert to tensor and add batch dimension
    sequence_tensor = torch.FloatTensor(recent_data).unsqueeze(0)
    
    return sequence_tensor


def main():
    """
    Test the dataset functionality
    """
    from data_handler import StockDataHandler
    
    # Load or prepare data
    handler = StockDataHandler()
    data = handler.prepare_data()
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(data)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)
    
    # Test a batch
    for batch_idx, (sequences, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
        if batch_idx == 0:  # Just show first batch
            break
    
    # Test prediction sequence
    pred_sequence = get_sequence_for_prediction(data)
    print(f"Prediction sequence shape: {pred_sequence.shape}")


if __name__ == "__main__":
    main()