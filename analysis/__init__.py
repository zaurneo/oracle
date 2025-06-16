# analysis/__init__.py - Package initialization
"""
Stock Analysis Package - Modular organization of forecasting tools
"""

# Import centralized state manager first
state_manager = None
ModelStateManager = None

try:
    from .shared_state import ModelStateManager, state_manager
    print("Centralized state manager imported successfully")
except ImportError as e:
    print(f"Warning: Could not import state manager: {e}")
    
    # Create a minimal fallback state manager
    class FallbackStateManager:
        """Fallback state manager when main one fails to import"""
        def __init__(self):
            self._data = {}
        
        def set_model_data(self, symbol, key, value):
            if symbol not in self._data:
                self._data[symbol] = {}
            self._data[symbol][key] = value
        
        def get_model_data(self, symbol, key, default=None):
            return self._data.get(symbol, {}).get(key, default)
        
        def has_model_data(self, symbol, key):
            return self.get_model_data(symbol, key) is not None
        
        def clear_model_data(self, symbol):
            if symbol in self._data:
                del self._data[symbol]
        
        def debug_state(self, symbol=None):
            print(f"Fallback state manager - data: {self._data}")
    
    state_manager = FallbackStateManager()
    ModelStateManager = FallbackStateManager

# Import all tools from submodules with better error handling
tools_imported = False

try:
    from .basic_tools import (
        get_stock_price,
        get_stock_news, 
        compare_stocks,
        get_technical_indicators
    )
    
    from .persistence import (
        save_trained_model,
        load_trained_model,
        list_saved_models,
        smart_predict_stock_price,
        delete_saved_model,
        model_performance_summary
    )
    
    from .ml_models import (
        fetch_historical_data,
        create_technical_features,
        train_decision_tree_model,
        predict_stock_price,
        backtest_model,
        quick_model_test
    )

    from .visualization import (
        create_model_visualization,
        model_summary_report
    )
    
    print("All analysis modules imported successfully")
    tools_imported = True
    
except ImportError as e:
    print(f"Warning: Some analysis modules failed to import: {e}")
    tools_imported = False

# Create stub functions only if imports failed
if not tools_imported:
    from langchain_core.tools import tool
    
    @tool
    def get_stock_price(symbol: str) -> str:
        """Get stock price (stub function)"""
        return f"Analysis package import error. Cannot get stock price for {symbol}."
    
    @tool 
    def save_trained_model(symbol: str, version: str = "latest", description: str = "") -> str:
        """Save trained model (stub function)"""
        return f"Analysis package import error. Cannot save model for {symbol}."
    
    @tool
    def create_technical_features(symbol: str, period: str = "1y") -> str:
        """Create technical features (stub function)"""
        return f"Analysis package import error. Cannot create features for {symbol}."
    
    @tool
    def train_decision_tree_model(symbol: str, test_size: float = 0.2, max_depth: int = 6) -> str:
        """Train decision tree model (stub function)"""
        return f"Analysis package import error. Cannot train model for {symbol}."
    
    @tool
    def predict_stock_price(symbol: str, days_ahead: int = 1) -> str:
        """Predict stock price (stub function)"""
        return f"Analysis package import error. Cannot predict price for {symbol}."
    
    @tool
    def create_model_visualization(symbol: str, chart_type: str = "performance") -> str:
        """Create model visualization (stub function)"""
        return f"Analysis package import error. Cannot create visualization for {symbol}."
    
    @tool
    def model_summary_report(symbol: str) -> str:
        """Generate model summary report (stub function)"""
        return f"Analysis package import error. Cannot generate report for {symbol}."
    
    @tool
    def get_stock_news(symbol: str) -> str:
        """Get stock news (stub function)"""
        return f"Analysis package import error. Cannot get news for {symbol}."
    
    @tool
    def compare_stocks(symbols: str) -> str:
        """Compare stocks (stub function)"""
        return f"Analysis package import error. Cannot compare stocks {symbols}."
    
    @tool
    def get_technical_indicators(symbol: str) -> str:
        """Get technical indicators (stub function)"""
        return f"Analysis package import error. Cannot get indicators for {symbol}."
    
    @tool
    def fetch_historical_data(symbol: str, period: str = "2y") -> str:
        """Fetch historical data (stub function)"""
        return f"Analysis package import error. Cannot fetch data for {symbol}."
    
    @tool
    def backtest_model(symbol: str, start_date: str = "2023-01-01") -> str:
        """Backtest model (stub function)"""
        return f"Analysis package import error. Cannot backtest model for {symbol}."
    
    @tool
    def quick_model_test(symbol: str) -> str:
        """Quick model test (stub function)"""
        return f"Analysis package import error. Cannot test model for {symbol}."
    
    @tool
    def load_trained_model(symbol: str, version: str = "latest") -> str:
        """Load trained model (stub function)"""
        return f"Analysis package import error. Cannot load model for {symbol}."
    
    @tool
    def list_saved_models() -> str:
        """List saved models (stub function)"""
        return "Analysis package import error. Cannot list models."
    
    @tool
    def smart_predict_stock_price(symbol: str, days_ahead: int = 1, auto_load: bool = True) -> str:
        """Smart predict stock price (stub function)"""
        return f"Analysis package import error. Cannot predict price for {symbol}."
    
    @tool
    def delete_saved_model(symbol: str, version: str = None) -> str:
        """Delete saved model (stub function)"""
        return f"Analysis package import error. Cannot delete model for {symbol}."
    
    @tool
    def model_performance_summary(symbol: str) -> str:
        """Model performance summary (stub function)"""
        return f"Analysis package import error. Cannot get performance summary for {symbol}."
    
    print("Using stub functions. Please check your analysis package installation.")

# Export all tools for easy importing
__all__ = [
    # State management
    'ModelStateManager', 'state_manager',
    
    # Basic tools
    'get_stock_price',
    'get_stock_news', 
    'compare_stocks',
    'get_technical_indicators',
    
    # ML models
    'fetch_historical_data',
    'create_technical_features',
    'train_decision_tree_model',
    'predict_stock_price',
    'backtest_model',
    'quick_model_test',
    
    # Persistence
    'save_trained_model',
    'load_trained_model',
    'list_saved_models',
    'smart_predict_stock_price',
    'delete_saved_model',
    'model_performance_summary',
    
    # Visualization
    'create_model_visualization',
    'model_summary_report'
]

# Package info
__version__ = "2.0.1"
__author__ = "Stock Forecasting System"
__description__ = "Modular stock analysis and ML forecasting tools with centralized state management"
