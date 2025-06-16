# fix_encoding.py - Fix file encoding issues
"""
This script creates clean versions of the files with proper encoding.
Run this to fix the encoding error.
"""

import os
from pathlib import Path

def create_clean_shared_state():
    """Create a clean shared_state.py file with ASCII-only content"""
    print("🔧 Creating clean shared_state.py file...")
    
    # Clean content without any special characters
    content = """# analysis/shared_state.py - Centralized State Manager
\"\"\"
Centralized state management for the entire analysis package.
This ensures all modules share the same state manager instance.
\"\"\"

import sys
from typing import Any, Optional, Dict

class ModelStateManager:
    \"\"\"Centralized state management for models across all modules\"\"\"
    _instance = None
    _models: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("Created new ModelStateManager instance")
        return cls._instance
    
    def set_model_data(self, symbol: str, key: str, value: Any) -> None:
        \"\"\"Set model data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            key: Data key (e.g., 'model', 'scaler')
            value: Data value to store
        \"\"\"
        if symbol not in self._models:
            self._models[symbol] = {}
        self._models[symbol][key] = value
        
        # Also set in main module's globals for maximum compatibility
        try:
            import __main__
            setattr(__main__, f"{symbol}_{key}", value)
        except Exception:
            pass
        
        # Also set in caller's globals for legacy compatibility
        try:
            frame = sys._getframe(1)
            frame.f_globals[f"{symbol}_{key}"] = value
        except Exception:
            pass
        
        print(f"Set {symbol}_{key} in state manager")
    
    def get_model_data(self, symbol: str, key: str, default: Any = None) -> Any:
        \"\"\"Get model data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            key: Data key (e.g., 'model', 'scaler')
            default: Default value if not found
            
        Returns:
            Stored data or default value
        \"\"\"
        # First try our internal storage
        if symbol in self._models and key in self._models[symbol]:
            print(f"Found {symbol}_{key} in state manager")
            return self._models[symbol][key]
        
        # Try main module globals
        try:
            import __main__
            value = getattr(__main__, f"{symbol}_{key}", None)
            if value is not None:
                print(f"Found {symbol}_{key} in __main__ globals")
                return value
        except Exception:
            pass
        
        # Try caller's globals
        try:
            frame = sys._getframe(1)
            value = frame.f_globals.get(f"{symbol}_{key}")
            if value is not None:
                print(f"Found {symbol}_{key} in caller globals")
                return value
        except Exception:
            pass
        
        print(f"Could not find {symbol}_{key} anywhere")
        return default
    
    def has_model_data(self, symbol: str, key: str) -> bool:
        \"\"\"Check if model data exists
        
        Args:
            symbol: Stock symbol
            key: Data key
            
        Returns:
            True if data exists, False otherwise
        \"\"\"
        return self.get_model_data(symbol, key) is not None
    
    def clear_model_data(self, symbol: str) -> None:
        \"\"\"Clear all data for a symbol
        
        Args:
            symbol: Stock symbol to clear
        \"\"\"
        if symbol in self._models:
            del self._models[symbol]
        print(f"Cleared all data for {symbol}")
    
    def debug_state(self, symbol: Optional[str] = None) -> None:
        \"\"\"Debug current state
        
        Args:
            symbol: Optional symbol to debug, or None for all
        \"\"\"
        if symbol:
            print(f"Debug state for {symbol}:")
            if symbol in self._models:
                for key, value in self._models[symbol].items():
                    print(f"  - {key}: {type(value)}")
            else:
                print(f"  - No data found for {symbol}")
        else:
            print(f"Debug all state:")
            print(f"  - Total symbols: {len(self._models)}")
            for sym, data in self._models.items():
                print(f"  - {sym}: {list(data.keys())}")

# Create the global state manager instance
state_manager = ModelStateManager()

# Export for easy importing
__all__ = ['ModelStateManager', 'state_manager']
"""
    
    # Ensure analysis directory exists
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Write the file with explicit UTF-8 encoding
    shared_state_path = analysis_dir / "shared_state.py"
    
    # Remove existing file if it exists
    if shared_state_path.exists():
        shared_state_path.unlink()
        print("Removed existing shared_state.py")
    
    # Write new clean file
    with open(shared_state_path, "w", encoding="utf-8", newline='\n') as f:
        f.write(content)
    
    print(f"✅ Created clean {shared_state_path} ({len(content)} bytes)")
    return True

def create_clean_init():
    """Create a clean __init__.py file"""
    print("🔧 Creating clean __init__.py file...")
    
    content = """# analysis/__init__.py - Package initialization
\"\"\"
Stock Analysis Package - Modular organization of forecasting tools
\"\"\"

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
        \"\"\"Fallback state manager when main one fails to import\"\"\"
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
        \"\"\"Get stock price (stub function)\"\"\"
        return f"Analysis package import error. Cannot get stock price for {symbol}."
    
    @tool 
    def save_trained_model(symbol: str, version: str = "latest", description: str = "") -> str:
        \"\"\"Save trained model (stub function)\"\"\"
        return f"Analysis package import error. Cannot save model for {symbol}."
    
    @tool
    def create_technical_features(symbol: str, period: str = "1y") -> str:
        \"\"\"Create technical features (stub function)\"\"\"
        return f"Analysis package import error. Cannot create features for {symbol}."
    
    @tool
    def train_decision_tree_model(symbol: str, test_size: float = 0.2, max_depth: int = 6) -> str:
        \"\"\"Train decision tree model (stub function)\"\"\"
        return f"Analysis package import error. Cannot train model for {symbol}."
    
    @tool
    def predict_stock_price(symbol: str, days_ahead: int = 1) -> str:
        \"\"\"Predict stock price (stub function)\"\"\"
        return f"Analysis package import error. Cannot predict price for {symbol}."
    
    @tool
    def create_model_visualization(symbol: str, chart_type: str = "performance") -> str:
        \"\"\"Create model visualization (stub function)\"\"\"
        return f"Analysis package import error. Cannot create visualization for {symbol}."
    
    @tool
    def model_summary_report(symbol: str) -> str:
        \"\"\"Generate model summary report (stub function)\"\"\"
        return f"Analysis package import error. Cannot generate report for {symbol}."
    
    @tool
    def get_stock_news(symbol: str) -> str:
        \"\"\"Get stock news (stub function)\"\"\"
        return f"Analysis package import error. Cannot get news for {symbol}."
    
    @tool
    def compare_stocks(symbols: str) -> str:
        \"\"\"Compare stocks (stub function)\"\"\"
        return f"Analysis package import error. Cannot compare stocks {symbols}."
    
    @tool
    def get_technical_indicators(symbol: str) -> str:
        \"\"\"Get technical indicators (stub function)\"\"\"
        return f"Analysis package import error. Cannot get indicators for {symbol}."
    
    @tool
    def fetch_historical_data(symbol: str, period: str = "2y") -> str:
        \"\"\"Fetch historical data (stub function)\"\"\"
        return f"Analysis package import error. Cannot fetch data for {symbol}."
    
    @tool
    def backtest_model(symbol: str, start_date: str = "2023-01-01") -> str:
        \"\"\"Backtest model (stub function)\"\"\"
        return f"Analysis package import error. Cannot backtest model for {symbol}."
    
    @tool
    def quick_model_test(symbol: str) -> str:
        \"\"\"Quick model test (stub function)\"\"\"
        return f"Analysis package import error. Cannot test model for {symbol}."
    
    @tool
    def load_trained_model(symbol: str, version: str = "latest") -> str:
        \"\"\"Load trained model (stub function)\"\"\"
        return f"Analysis package import error. Cannot load model for {symbol}."
    
    @tool
    def list_saved_models() -> str:
        \"\"\"List saved models (stub function)\"\"\"
        return "Analysis package import error. Cannot list models."
    
    @tool
    def smart_predict_stock_price(symbol: str, days_ahead: int = 1, auto_load: bool = True) -> str:
        \"\"\"Smart predict stock price (stub function)\"\"\"
        return f"Analysis package import error. Cannot predict price for {symbol}."
    
    @tool
    def delete_saved_model(symbol: str, version: str = None) -> str:
        \"\"\"Delete saved model (stub function)\"\"\"
        return f"Analysis package import error. Cannot delete model for {symbol}."
    
    @tool
    def model_performance_summary(symbol: str) -> str:
        \"\"\"Model performance summary (stub function)\"\"\"
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
"""
    
    # Write the file with explicit UTF-8 encoding
    init_path = Path("analysis") / "__init__.py"
    
    # Remove existing file if it exists
    if init_path.exists():
        init_path.unlink()
        print("Removed existing __init__.py")
    
    # Write new clean file
    with open(init_path, "w", encoding="utf-8", newline='\n') as f:
        f.write(content)
    
    print(f"✅ Created clean {init_path} ({len(content)} bytes)")
    return True

def fix_file_encoding():
    """Fix encoding issues by recreating files cleanly"""
    print("🔧 FIXING FILE ENCODING ISSUES")
    print("=" * 50)
    
    # Ensure analysis directory exists
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    try:
        # Create clean files
        create_clean_shared_state()
        create_clean_init()
        
        print("\n✅ All files recreated with clean encoding!")
        print("\n🧪 Test the fix:")
        print("python test_simple_import.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing files: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_file_encoding()