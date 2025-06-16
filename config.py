# config.py - CENTRALIZED CONFIGURATION
# New file to consolidate settings and constants

import os
from pathlib import Path

# API Configuration
class APIConfig:
    """API keys and model configuration"""
    CLAUDE_API_KEY = os.environ.get("claude_api_key", "")
    GPT_API_KEY = os.environ.get("gpt_api_key", "")
    
    # Model names
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
    GPT_MODEL = "gpt-4o-2024-08-06"
    
    @classmethod
    def validate(cls):
        """Validate API keys are present"""
        missing = []
        if not cls.CLAUDE_API_KEY:
            missing.append("claude_api_key")
        if not cls.GPT_API_KEY:
            missing.append("gpt_api_key")
        
        if missing:
            raise ValueError(f"Missing API keys: {missing}")

# File Paths
class Paths:
    """File and directory paths"""
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    PLOTS_DIR = BASE_DIR / "plots"
    LOGS_DIR = BASE_DIR / "logs"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for path in [cls.MODELS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR]:
            path.mkdir(exist_ok=True)

# Model Configuration
class ModelConfig:
    """Machine learning model configuration"""
    
    # RandomForest parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 6
    MIN_SAMPLES_SPLIT = 50
    MIN_SAMPLES_LEAF = 20
    RANDOM_STATE = 42
    
    # Data parameters
    DEFAULT_PERIOD = "1y"
    MIN_DATA_POINTS = 100
    TEST_SIZE = 0.2
    
    # Feature columns
    FEATURE_COLUMNS = [
        'Returns', 'Log_Returns', 'Price_Change', 'High_Low_Pct', 'Close_Open_Pct',
        'MA_5', 'MA_10', 'MA_20', 'MA_50',
        'Price_MA5_Ratio', 'Price_MA20_Ratio', 'MA5_MA20_Ratio',
        'Volatility_5', 'Volatility_20',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'Volume_Ratio', 'Price_Position_14',
        'Trend_5', 'Trend_10'
    ]

# Visualization Configuration
class VizConfig:
    """Visualization settings"""
    
    # Plot settings
    DPI = 300
    FIGSIZE_SINGLE = (12, 8)
    FIGSIZE_MULTI = (15, 10)
    
    # Colors
    COLORS = {
        'actual': '#1f77b4',
        'predicted': '#ff7f0e',
        'signal': '#2ca02c',
        'error': '#d62728'
    }
    
    # Chart types
    CHART_TYPES = [
        'performance',
        'feature_importance', 
        'prediction_vs_actual',
        'backtest',
        'all'
    ]

# Logging Configuration
class LogConfig:
    """Logging settings"""
    
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # File naming
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    @classmethod
    def get_log_filename(cls, prefix="conversation"):
        """Get timestamped log filename"""
        from datetime import datetime
        timestamp = datetime.now().strftime(cls.TIMESTAMP_FORMAT)
        return f"{prefix}_{timestamp}.txt"

# Stock Analysis Configuration
class StockConfig:
    """Stock analysis settings"""
    
    # Default symbols for demos
    DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    # Analysis periods
    PERIODS = {
        'short': '3mo',
        'medium': '1y', 
        'long': '2y',
        'max': '5y'
    }
    
    # Technical indicator parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2

# Demo Configuration
class DemoConfig:
    """Demo and example queries"""
    
    DEMO_QUERIES = [
        "Create comprehensive RandomForest forecasting model for {symbol} - advanced data collection, ML features, model training, persistence, backtesting, and visualization",
        
        "Analyze {symbol} with machine learning forecasting - technical features, RandomForest training, price predictions, model saving, strategy backtesting, and performance charts",
        
        "Build professional predictive system for {symbol} - historical data, feature engineering, ML model training, price forecasting, confidence intervals, and investment recommendations",
        
        "Develop algorithmic trading system for {symbol} - comprehensive features, RandomForest training, model persistence, strategy backtesting, and executive reports",
        
        "Multi-stock comparative analysis - RandomForest models for {symbols}, price predictions, model persistence, comparative visualizations, and investment recommendations",
        
        "Portfolio optimization system - RandomForest models for major stocks, price movement predictions, model persistence, allocation recommendations, and comprehensive reports"
    ]
    
    @classmethod
    def get_demo_query(cls, index=0, symbol="AAPL", symbols="AAPL,GOOGL,TSLA"):
        """Get formatted demo query"""
        if index < len(cls.DEMO_QUERIES):
            return cls.DEMO_QUERIES[index].format(symbol=symbol, symbols=symbols)
        return cls.DEMO_QUERIES[0].format(symbol=symbol, symbols=symbols)

# System Configuration
class SystemConfig:
    """Overall system settings"""
    
    # Version info
    VERSION = "2.0.0"
    NAME = "Advanced Stock Forecasting System"
    
    # Performance settings
    MAX_WORKERS = 4
    TIMEOUT_SECONDS = 300
    
    # Error handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

# Initialize directories on import
Paths.create_directories()

# Validation helper
def validate_config():
    """Validate all configuration"""
    try:
        APIConfig.validate()
        print("✅ Configuration validated successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

# Export commonly used items
__all__ = [
    'APIConfig', 'Paths', 'ModelConfig', 'VizConfig', 
    'LogConfig', 'StockConfig', 'DemoConfig', 'SystemConfig',
    'validate_config'
]