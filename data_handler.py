"""
Data acquisition and preprocessing module for stock prediction AI
Handles yfinance data download, cleaning, and preparation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import config


class StockDataHandler:
    def __init__(self):
        self.target_scaler = MinMaxScaler()
        self.market_scaler = MinMaxScaler()
        self.target_data = None
        self.market_data = None
        self.combined_data = None
        
    def download_stock_data(self, symbol: str, period: str = config.DATA_PERIOD, 
                           interval: str = config.DATA_INTERVAL) -> pd.DataFrame:
        """
        Download stock data using yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Remove timezone information and ensure datetime index
            data.index = pd.to_datetime(data.index).tz_localize(None)
            
            # Sort by timestamp
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock data by handling missing values and outliers
        """
        # Remove rows with any NaN values
        data_clean = data.dropna()
        
        # Check for obvious data errors (negative prices, zero volume)
        data_clean = data_clean[data_clean['Close'] > 0]
        data_clean = data_clean[data_clean['Volume'] >= 0]
        
        # Remove extreme outliers (price changes > 20% in 5 minutes)
        price_change = data_clean['Close'].pct_change().abs()
        data_clean = data_clean[price_change <= 0.2]
        
        return data_clean
    
    def normalize_features(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler
        """
        features = config.FEATURE_COLUMNS
        normalized_data = data.copy()
        
        if fit_scaler:
            if config.NORMALIZATION_METHOD == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            normalized_data[features] = scaler.fit_transform(data[features])
            return normalized_data, scaler
        else:
            # Use existing scaler (for inference)
            normalized_data[features] = self.target_scaler.transform(data[features])
            return normalized_data
    
    def align_data(self, target_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align target and market data by timestamp
        """
        # Rename market data columns to avoid conflicts
        market_renamed = market_data.copy()
        market_renamed.columns = [f'Market_{col}' for col in market_renamed.columns]
        
        # Merge on index (timestamp)
        aligned_data = target_data.join(market_renamed, how='inner')
        
        # Remove rows where either dataset has missing values
        aligned_data = aligned_data.dropna()
        
        return aligned_data
    
    def prepare_data(self, save_to_file: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline
        """
        print("Starting data preparation...")
        
        # Download data for both symbols
        print(f"Downloading target data for {config.TARGET_SYMBOL}...")
        target_raw = self.download_stock_data(config.TARGET_SYMBOL)
        
        print(f"Downloading market data for {config.MARKET_SYMBOL}...")
        market_raw = self.download_stock_data(config.MARKET_SYMBOL)
        
        # Clean data
        print("Cleaning target data...")
        target_clean = self.clean_data(target_raw)
        
        print("Cleaning market data...")
        market_clean = self.clean_data(market_raw)
        
        # Normalize features
        print("Normalizing target data...")
        target_normalized, self.target_scaler = self.normalize_features(target_clean, fit_scaler=True)
        
        print("Normalizing market data...")
        market_normalized, self.market_scaler = self.normalize_features(market_clean, fit_scaler=True)
        
        # Align data by timestamp
        print("Aligning data...")
        self.combined_data = self.align_data(target_normalized, market_normalized)
        
        # Create directories if they don't exist
        os.makedirs(config.DATA_SAVE_PATH, exist_ok=True)
        
        if save_to_file:
            # Save processed data
            data_file = os.path.join(config.DATA_SAVE_PATH, "processed_data.csv")
            self.combined_data.to_csv(data_file)
            
            # Save scalers
            import pickle
            with open(os.path.join(config.DATA_SAVE_PATH, "target_scaler.pkl"), 'wb') as f:
                pickle.dump(self.target_scaler, f)
            with open(os.path.join(config.DATA_SAVE_PATH, "market_scaler.pkl"), 'wb') as f:
                pickle.dump(self.market_scaler, f)
        
        return self.combined_data
    
    def load_data(self, data_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load previously processed data
        """
        if data_file is None:
            data_file = os.path.join(config.DATA_SAVE_PATH, "processed_data.csv")
        
        self.combined_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # Load scalers
        import pickle
        with open(os.path.join(config.DATA_SAVE_PATH, "target_scaler.pkl"), 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(os.path.join(config.DATA_SAVE_PATH, "market_scaler.pkl"), 'rb') as f:
            self.market_scaler = pickle.load(f)
        
        print(f"Loaded data with shape: {self.combined_data.shape}")
        return self.combined_data
    
    def get_latest_data_raw(self) -> pd.DataFrame:
        """
        Get the most recent raw data (before normalization) for price reference
        """
        # Download latest data
        target_latest = self.download_stock_data(config.TARGET_SYMBOL, period="5d")
        market_latest = self.download_stock_data(config.MARKET_SYMBOL, period="5d")
        
        # Clean but don't normalize
        target_clean = self.clean_data(target_latest)
        market_clean = self.clean_data(market_latest)
        
        # Align data
        raw_data = self.align_data(target_clean, market_clean)
        
        return raw_data
    
    def get_latest_data(self) -> pd.DataFrame:
        """
        Get the most recent data for real-time inference
        """
        print("Fetching latest data for inference...")
        
        # Download latest data
        target_latest = self.download_stock_data(config.TARGET_SYMBOL, period="5d")
        market_latest = self.download_stock_data(config.MARKET_SYMBOL, period="5d")
        
        # Clean and normalize
        target_clean = self.clean_data(target_latest)
        market_clean = self.clean_data(market_latest)
        
        # Use existing scalers (don't refit)
        target_normalized = target_clean.copy()
        target_normalized[config.FEATURE_COLUMNS] = self.target_scaler.transform(target_clean[config.FEATURE_COLUMNS])
        
        market_normalized = market_clean.copy()
        market_normalized[config.FEATURE_COLUMNS] = self.market_scaler.transform(market_clean[config.FEATURE_COLUMNS])
        
        # Align data
        latest_data = self.align_data(target_normalized, market_normalized)
        
        return latest_data


def main():
    """
    Test the data handler functionality
    """
    handler = StockDataHandler()
    
    # Prepare data
    data = handler.prepare_data()
    
    print("\nData summary:")
    print(data.describe())
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")


if __name__ == "__main__":
    main()