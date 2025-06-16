# analysis/shared_state.py - Centralized State Manager
"""
Centralized state management for the entire analysis package.
This ensures all modules share the same state manager instance.
"""

import sys
from typing import Any, Optional, Dict

class ModelStateManager:
    """Centralized state management for models across all modules"""
    _instance = None
    _models: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("Created new ModelStateManager instance")
        return cls._instance
    
    def set_model_data(self, symbol: str, key: str, value: Any) -> None:
        """Set model data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            key: Data key (e.g., 'model', 'scaler')
            value: Data value to store
        """
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
        """Get model data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            key: Data key (e.g., 'model', 'scaler')
            default: Default value if not found
            
        Returns:
            Stored data or default value
        """
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
        """Check if model data exists
        
        Args:
            symbol: Stock symbol
            key: Data key
            
        Returns:
            True if data exists, False otherwise
        """
        return self.get_model_data(symbol, key) is not None
    
    def clear_model_data(self, symbol: str) -> None:
        """Clear all data for a symbol - FIXED VERSION
        
        Args:
            symbol: Stock symbol to clear
        """
        # Clear from internal storage
        keys_to_clear = []
        if symbol in self._models:
            keys_to_clear = list(self._models[symbol].keys())
            del self._models[symbol]
        
        # Clear from main module globals
        try:
            import __main__
            for key in keys_to_clear:
                if hasattr(__main__, f"{symbol}_{key}"):
                    delattr(__main__, f"{symbol}_{key}")
                    print(f"Cleared {symbol}_{key} from __main__ globals")
        except Exception:
            pass
        
        # Clear from caller's globals
        try:
            frame = sys._getframe(1)
            for key in keys_to_clear:
                if f"{symbol}_{key}" in frame.f_globals:
                    del frame.f_globals[f"{symbol}_{key}"]
                    print(f"Cleared {symbol}_{key} from caller globals")
        except Exception:
            pass
        
        print(f"Cleared all data for {symbol}")
    
    def debug_state(self, symbol: Optional[str] = None) -> None:
        """Debug current state
        
        Args:
            symbol: Optional symbol to debug, or None for all
        """
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
