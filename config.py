"""
Configuration file for stock prediction AI project
Target: Toyota Motor Corporation (7203.T) day trading strategy
"""

# Stock symbols
TARGET_SYMBOL = "7203.T"  # Toyota Motor Corporation
MARKET_SYMBOL = "1321.T"  # Nikkei 225 ETF for market trend

# Data parameters
DATA_INTERVAL = "5m"  # 5-minute candlesticks
DATA_PERIOD = "60d"   # Last 60 days (yfinance limitation)
PREDICTION_HORIZON = 1  # Predict next 5-minute close price

# Model architecture parameters
SEQUENCE_LENGTH = 48  # 4 hours of 5-min data (48 * 5min = 4 hours)
MODEL_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
MAX_SEQUENCE_LENGTH = 1000

# GQA parameters
GQA_NUM_KEY_VALUE_HEADS = 2  # Reduced from num_heads for efficiency

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data preprocessing
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
NORMALIZATION_METHOD = 'minmax'  # 'minmax' or 'standard'

# File paths
MODEL_SAVE_PATH = "models/"
DATA_SAVE_PATH = "data/"
RESULTS_SAVE_PATH = "results/"
CHECKPOINT_PATH = "checkpoints/"

# Device configuration
DEVICE = "cpu"  # Will be updated to "cuda" if available

# Evaluation parameters
BACKTEST_INITIAL_CAPITAL = 1000000  # 100万円
TRANSACTION_COST = 0.001  # 0.1% transaction cost
SIGNAL_THRESHOLD = 0.005  # 0.5% price change threshold for buy/sell signals

# LLM Advisor parameters
LLM_MODEL = "gpt-3.5-turbo"  # OpenAI model to use
LLM_MAX_TOKENS = 500  # Maximum tokens for response
LLM_TEMPERATURE = 0.7  # Creativity level (0.0 to 1.0)
LLM_RATE_LIMIT_DELAY = 1.0  # Seconds between API calls to avoid rate limiting
LLM_MAX_RETRIES = 3  # Maximum retry attempts for API calls
LLM_HISTORY_LIMIT = 20  # Number of recent predictions to include in context
ADVISORY_HISTORY_LIMIT = 50  # Maximum stored advisory messages