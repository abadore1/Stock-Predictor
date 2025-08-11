# Technical Indicators Integration

## Overview

This update adds comprehensive technical indicators analysis to the Stock Predictor application, integrating traditional technical analysis tools with the existing AI-based prediction system.

## Features Added

### Technical Indicators Implemented

1. **RSI (Relative Strength Index)**
   - Period: 14 (configurable)
   - Signals: Overbought (>70), Oversold (<30)
   
2. **MACD (Moving Average Convergence Divergence)**
   - Fast EMA: 12, Slow EMA: 26, Signal: 9
   - Includes MACD line, signal line, and histogram
   
3. **Bollinger Bands**
   - Period: 20, Standard Deviation: 2.0
   - Upper band, middle band (SMA), lower band
   
4. **Stochastic Oscillator**
   - %K period: 14, %D period: 3
   - Signals: Overbought (>80), Oversold (<20)
   
5. **Ichimoku Cloud (一目均衡表)**
   - Tenkan-sen (9), Kijun-sen (26), Senkou B (52)
   - Full cloud visualization with displacement

### New Files

- **`technical_indicators.py`**: Core technical indicators calculation module
- **`test_technical_indicators.py`**: Testing script for validation

### Modified Files

- **`app.py`**: Added technical indicators section to Streamlit UI
- **`visualize.py`**: Enhanced with technical indicators plotting capabilities

## Usage

### In Streamlit App

The technical indicators are integrated into the main Streamlit interface with:

1. **Interactive Controls**:
   - Toggle individual indicators on/off
   - Select individual indicator for detailed view
   - Enable/disable buy/sell signals

2. **Comprehensive Dashboard**:
   - Combined view of all indicators
   - Price chart with Bollinger Bands and Ichimoku Cloud
   - Separate panels for RSI, MACD, and Stochastic

3. **Signal Summary**:
   - Real-time signal status for each indicator
   - Combined signal using majority voting
   - Color-coded signal display (BUY/SELL/HOLD)

4. **Latest Values Display**:
   - Current RSI, MACD, and Stochastic values
   - Bollinger Bands position percentage
   - Ichimoku Cloud status

### Command Line Interface

Enhanced the visualization menu with technical indicators option:

```bash
python visualize.py
# Select option 7 for Technical indicators
```

## Technical Implementation

### Signal Generation

The system generates signals using multiple strategies:

- **RSI**: Overbought/oversold levels
- **MACD**: Line crossovers
- **Bollinger Bands**: Price touching bands
- **Stochastic**: Extreme levels
- **Ichimoku**: Price vs cloud position

### Combined Signals

Uses majority voting from all indicators:
- **Strong Buy**: 2+ positive signals
- **Strong Sell**: 2+ negative signals
- **Hold**: Neutral or conflicting signals

### Visualization Features

- **Multi-panel Charts**: Separate subplots for each indicator
- **Interactive Plotly**: Zoom, pan, hover tooltips
- **Signal Markers**: Buy/sell points on price chart
- **Color Coding**: Intuitive green/red signal indication

## Configuration

All indicators use standard parameters but can be easily modified in the `TechnicalIndicators` class:

```python
# Example: Change RSI period
rsi = technical_indicators.rsi(data['Close'], period=21)

# Example: Adjust Bollinger Bands
bb = technical_indicators.bollinger_bands(data['Close'], period=20, std_dev=2.5)
```

## Integration with Existing System

The technical indicators complement the AI prediction system by:

1. **Providing Alternative Analysis**: Traditional technical analysis alongside ML predictions
2. **Signal Confirmation**: Cross-validation between AI and technical signals
3. **Risk Assessment**: Multiple perspectives on market conditions
4. **Educational Value**: Learning traditional trading strategies

## Future Enhancements

Potential areas for expansion:

1. **Additional Indicators**: Williams %R, CCI, ADX
2. **Custom Strategies**: User-defined indicator combinations
3. **Backtesting Integration**: Test technical strategies historically
4. **Alert System**: Notifications for signal changes
5. **Pattern Recognition**: Chart pattern detection

## Dependencies

The technical indicators module requires:
- pandas: Data manipulation
- numpy: Numerical calculations
- plotly: Interactive visualization

## Testing

Run the test script to validate functionality:

```bash
python test_technical_indicators.py
```

Note: Tests require pandas/numpy to be installed.