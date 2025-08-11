"""
Technical Indicators Module for Stock Prediction AI
Implements RSI, MACD, Bollinger Bands, Stochastics, and Ichimoku Cloud indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Technical indicators calculation and signal generation class
    """
    
    def __init__(self):
        pass
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: Price series (typically Close price)
            period: RSI period (default: 14)
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series (typically Close price)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            
        Returns:
            Dict containing MACD line, signal line, and histogram
        """
        ema_fast = data.ewm(span=fast_period).mean()
        ema_slow = data.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series (typically Close price)
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            Dict containing upper band, middle band (SMA), and lower band
        """
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Dict containing %K and %D lines
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      tenkan_period: int = 9, kijun_period: int = 26, 
                      senkou_b_period: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud (一目均衡表)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            tenkan_period: Tenkan-sen period (default: 9)
            kijun_period: Kijun-sen period (default: 26)
            senkou_b_period: Senkou Span B period (default: 52)
            displacement: Cloud displacement (default: 26)
            
        Returns:
            Dict containing all Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                     low.rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=kijun_period).max() + 
                    low.rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=senkou_b_period).max() + 
                         low.rolling(window=senkou_b_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        result_df = df.copy()
        
        # RSI
        result_df['rsi'] = self.rsi(df['Close'])
        
        # MACD
        macd_data = self.macd(df['Close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.bollinger_bands(df['Close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        
        # Stochastic Oscillator
        stoch_data = self.stochastic_oscillator(df['High'], df['Low'], df['Close'])
        result_df['stoch_k'] = stoch_data['k_percent']
        result_df['stoch_d'] = stoch_data['d_percent']
        
        # Ichimoku Cloud
        ichimoku_data = self.ichimoku_cloud(df['High'], df['Low'], df['Close'])
        result_df['ichimoku_tenkan'] = ichimoku_data['tenkan_sen']
        result_df['ichimoku_kijun'] = ichimoku_data['kijun_sen']
        result_df['ichimoku_senkou_a'] = ichimoku_data['senkou_span_a']
        result_df['ichimoku_senkou_b'] = ichimoku_data['senkou_span_b']
        result_df['ichimoku_chikou'] = ichimoku_data['chikou_span']
        
        return result_df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on technical indicators
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            DataFrame with signal columns added
        """
        result_df = df.copy()
        
        # Initialize signal columns
        result_df['rsi_signal'] = 0
        result_df['macd_signal'] = 0
        result_df['bb_signal'] = 0
        result_df['stoch_signal'] = 0
        result_df['ichimoku_signal'] = 0
        result_df['combined_signal'] = 0
        
        # RSI signals (oversold/overbought)
        result_df.loc[result_df['rsi'] < 30, 'rsi_signal'] = 1  # Buy signal
        result_df.loc[result_df['rsi'] > 70, 'rsi_signal'] = -1  # Sell signal
        
        # MACD signals (crossover)
        macd_crossover = (result_df['macd'] > result_df['macd_signal']) & \
                        (result_df['macd'].shift(1) <= result_df['macd_signal'].shift(1))
        macd_crossunder = (result_df['macd'] < result_df['macd_signal']) & \
                         (result_df['macd'].shift(1) >= result_df['macd_signal'].shift(1))
        
        result_df.loc[macd_crossover, 'macd_signal'] = 1  # Buy signal
        result_df.loc[macd_crossunder, 'macd_signal'] = -1  # Sell signal
        
        # Bollinger Bands signals (price touching bands)
        result_df.loc[result_df['Close'] <= result_df['bb_lower'], 'bb_signal'] = 1  # Buy signal
        result_df.loc[result_df['Close'] >= result_df['bb_upper'], 'bb_signal'] = -1  # Sell signal
        
        # Stochastic signals (oversold/overbought)
        result_df.loc[(result_df['stoch_k'] < 20) & (result_df['stoch_d'] < 20), 'stoch_signal'] = 1
        result_df.loc[(result_df['stoch_k'] > 80) & (result_df['stoch_d'] > 80), 'stoch_signal'] = -1
        
        # Ichimoku signals (price vs cloud)
        cloud_top = np.maximum(result_df['ichimoku_senkou_a'], result_df['ichimoku_senkou_b'])
        cloud_bottom = np.minimum(result_df['ichimoku_senkou_a'], result_df['ichimoku_senkou_b'])
        
        result_df.loc[result_df['Close'] > cloud_top, 'ichimoku_signal'] = 1  # Above cloud (bullish)
        result_df.loc[result_df['Close'] < cloud_bottom, 'ichimoku_signal'] = -1  # Below cloud (bearish)
        
        # Combined signal (majority vote)
        signal_sum = (result_df['rsi_signal'] + result_df['macd_signal'] + 
                     result_df['bb_signal'] + result_df['stoch_signal'] + 
                     result_df['ichimoku_signal'])
        
        result_df.loc[signal_sum >= 2, 'combined_signal'] = 1  # Strong buy
        result_df.loc[signal_sum <= -2, 'combined_signal'] = -1  # Strong sell
        
        return result_df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get current signal summary for all indicators
        
        Args:
            df: DataFrame with calculated signals
            
        Returns:
            Dict with signal summaries
        """
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        def signal_to_text(signal_value: float) -> str:
            if signal_value > 0:
                return "BUY"
            elif signal_value < 0:
                return "SELL"
            else:
                return "HOLD"
        
        return {
            'RSI': signal_to_text(latest['rsi_signal']),
            'MACD': signal_to_text(latest['macd_signal']),
            'Bollinger Bands': signal_to_text(latest['bb_signal']),
            'Stochastic': signal_to_text(latest['stoch_signal']),
            'Ichimoku': signal_to_text(latest['ichimoku_signal']),
            'Combined': signal_to_text(latest['combined_signal'])
        }