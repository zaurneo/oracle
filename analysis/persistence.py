# analysis/persistence.py - UPDATED to use centralized state manager
"""
Model persistence and management tools for production deployment.
FIXED: Uses centralized state manager from shared_state module.
"""

import os
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from langchain_core.tools import tool

# FIXED: Import centralized state manager
from .shared_state import state_manager

def ensure_models_directory() -> Path:
    """Create models directory structure if it doesn't exist"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    return models_dir

def get_model_paths(symbol: str, version: str = "latest") -> Dict[str, Path]:
    """Get all file paths for a model"""
    models_dir = ensure_models_directory()
    symbol_dir = models_dir / symbol
    symbol_dir.mkdir(exist_ok=True)
    
    return {
        'model': symbol_dir / f"model_{version}.joblib",
        'scaler': symbol_dir / f"scaler_{version}.joblib", 
        'features': symbol_dir / f"features_{version}.json",
        'metadata': symbol_dir / f"metadata_{version}.json"
    }

@tool
def save_trained_model(symbol: str, version: str = "latest", description: str = "") -> str:
    """Save trained model and all components to disk permanently
    
    Args:
        symbol: Stock ticker symbol
        version: Model version identifier (default: "latest")
        description: Optional description of the model
        
    Returns:
        Success message with file locations and model info
    """
    try:
        print(f"🔍 Attempting to save model for {symbol}...")
        state_manager.debug_state(symbol)
        
        # FIXED: Use centralized state manager
        if not state_manager.has_model_data(symbol, 'model'):
            return f"❌ No trained model found for {symbol} in memory. Train the model first using train_decision_tree_model.\n\nDebug info: {state_manager.debug_state(symbol)}"
        
        # Get model components from centralized state manager
        model = state_manager.get_model_data(symbol, 'model')
        scaler = state_manager.get_model_data(symbol, 'scaler')
        feature_columns = state_manager.get_model_data(symbol, 'feature_columns')
        
        if model is None or scaler is None or feature_columns is None:
            missing = []
            if model is None: missing.append('model')
            if scaler is None: missing.append('scaler')
            if feature_columns is None: missing.append('feature_columns')
            return f"❌ Incomplete model data for {symbol}. Missing: {missing}. Please retrain the model."
        
        print(f"✅ Found all model components for {symbol}")
        
        # Get model paths
        paths = get_model_paths(symbol, version)
        
        # Save model components using joblib
        joblib.dump(model, paths['model'])
        joblib.dump(scaler, paths['scaler'])
        
        # Save feature columns as JSON
        with open(paths['features'], 'w') as f:
            json.dump(feature_columns, f, indent=2)
        
        # Create comprehensive metadata
        metadata = {
            "symbol": symbol,
            "version": version,
            "description": description,
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, 'n_estimators', 'N/A'),
            "max_depth": getattr(model, 'max_depth', 'N/A'),
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "saved_at": datetime.now().isoformat(),
            "created_by": "Stock Forecasting System v2.0",
            "target_variable": "Price Changes (not absolute prices)"
        }
        
        # Add performance metrics if available
        test_data = state_manager.get_model_data(symbol, 'test_data')
        if test_data is not None:
            X_test, y_test, y_pred_test = test_data
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            pred_variance = np.var(y_pred_test)
            actual_variance = np.var(y_test)
            variance_ratio = pred_variance / actual_variance if actual_variance > 0 else 0
            
            metadata["performance"] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2),
                "variance_ratio": float(variance_ratio),
                "test_samples": len(y_test),
                "prediction_range": f"{np.min(y_pred_test):.2f} to {np.max(y_pred_test):.2f}"
            }
        
        # Save metadata
        with open(paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest versions
        latest_paths = get_model_paths(symbol, "latest")
        if version != "latest":
            import shutil
            for key in ['model', 'scaler', 'features', 'metadata']:
                if latest_paths[key].exists():
                    latest_paths[key].unlink()
                shutil.copy2(paths[key], latest_paths[key])
        
        # Calculate file sizes
        file_sizes = {}
        total_size = 0
        for key, path in paths.items():
            if path.exists():
                size_kb = path.stat().st_size / 1024
                file_sizes[key] = f"{size_kb:.1f} KB"
                total_size += size_kb
        
        result = f"""
✅ Model Successfully Saved for {symbol}!
{'=' * 50}

📁 Saved Files:
- Model: {paths['model']} ({file_sizes.get('model', 'N/A')})
- Scaler: {paths['scaler']} ({file_sizes.get('scaler', 'N/A')}) 
- Features: {paths['features']} ({file_sizes.get('features', 'N/A')})
- Metadata: {paths['metadata']} ({file_sizes.get('metadata', 'N/A')})

💾 Model Information:
- Version: {version}
- Type: {metadata['model_type']}
- Trees: {metadata['n_estimators']}
- Max Depth: {metadata['max_depth']}
- Features: {metadata['n_features']}
- Total Size: {total_size:.1f} KB

🚀 Production Ready:
- Auto-loading: smart_predict_stock_price('{symbol}')
- Manual loading: load_trained_model('{symbol}')
- Model management: list_saved_models()

💡 Model is now permanently saved and ready for production deployment!
        """
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error saving model for {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

@tool
def load_trained_model(symbol: str, version: str = "latest") -> str:
    """Load a previously saved model from disk into memory"""
    try:
        paths = get_model_paths(symbol, version)
        
        # Check if all required files exist
        missing_files = []
        for file_type, path in paths.items():
            if not path.exists():
                missing_files.append(file_type)
        
        if missing_files:
            return f"❌ Missing files for {symbol} version {version}: {missing_files}"
        
        # Load model components
        model = joblib.load(paths['model'])
        scaler = joblib.load(paths['scaler'])
        
        with open(paths['features'], 'r') as f:
            feature_columns = json.load(f)
        
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)
        
        # FIXED: Store in centralized state manager
        state_manager.set_model_data(symbol, 'model', model)
        state_manager.set_model_data(symbol, 'scaler', scaler)
        state_manager.set_model_data(symbol, 'feature_columns', feature_columns)
        
        result = f"""
✅ Model Successfully Loaded for {symbol}!
{'=' * 50}

🤖 Loaded Model Information:
- Version: {metadata['version']}
- Type: {metadata['model_type']}
- Trees: {metadata.get('n_estimators', 'N/A')}
- Max Depth: {metadata.get('max_depth', 'N/A')}
- Features: {metadata['n_features']}
- Saved: {metadata['saved_at'][:19]}

🚀 Model Ready for Use!
        """
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error loading model for {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

@tool
def list_saved_models() -> str:
    """List all saved models with their versions and metadata"""
    try:
        models_dir = ensure_models_directory()
        
        if not any(models_dir.iterdir()):
            return "📁 No models have been saved yet.\n\nTrain and save your first model with:\n1. create_technical_features('AAPL')\n2. train_decision_tree_model('AAPL')\n3. save_trained_model('AAPL')"
        
        result = "📁 SAVED MODELS INVENTORY\n"
        result += "=" * 60 + "\n"
        
        total_models = 0
        for symbol_dir in sorted(models_dir.iterdir()):
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                result += f"\n📊 {symbol}:\n"
                
                model_files = list(symbol_dir.glob("model_*.joblib"))
                if not model_files:
                    result += "   No models found\n"
                    continue
                
                for model_file in sorted(model_files):
                    version = model_file.stem.replace("model_", "")
                    metadata_file = symbol_dir / f"metadata_{version}.json"
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            saved_date = metadata.get('saved_at', 'Unknown')[:10]
                            model_type = metadata.get('model_type', 'Unknown')
                            n_trees = metadata.get('n_estimators', 'N/A')
                            
                            result += f"   📈 {version}: {model_type}({n_trees}) | {saved_date}\n"
                            
                        except json.JSONDecodeError:
                            result += f"   ⚠️  {version}: Metadata corrupted\n"
                    else:
                        result += f"   ❓ {version}: No metadata\n"
                    
                    total_models += 1
        
        result += f"\n📈 Total Models: {total_models}\n"
        return result
        
    except Exception as e:
        return f"❌ Error listing models: {str(e)}"

@tool
def smart_predict_stock_price(symbol: str, days_ahead: int = 1, auto_load: bool = True) -> str:
    """Smart prediction that auto-loads model if not in memory"""
    try:
        # Check centralized state manager
        if not state_manager.has_model_data(symbol, 'model'):
            if not auto_load:
                return f"❌ No model in memory for {symbol} and auto_load=False"
            
            load_result = load_trained_model.invoke({"symbol": symbol})
            if "❌" in load_result:
                return f"❌ No trained model found for {symbol} in memory or on disk."
        
        # Import and use prediction function
        try:
            from .ml_models import predict_stock_price
            return predict_stock_price.invoke({"symbol": symbol, "days_ahead": days_ahead})
        except ImportError:
            return f"❌ Cannot import prediction function"
        
    except Exception as e:
        return f"❌ Error in smart prediction for {symbol}: {str(e)}"

@tool
def delete_saved_model(symbol: str, version: Optional[str] = None) -> str:
    """Delete saved model(s) for cleanup and management"""
    try:
        models_dir = ensure_models_directory()
        symbol_dir = models_dir / symbol
        
        if not symbol_dir.exists():
            return f"❌ No saved models found for {symbol}"
        
        if version is None:
            import shutil
            file_count = len(list(symbol_dir.glob("*")))
            shutil.rmtree(symbol_dir)
            state_manager.clear_model_data(symbol)
            return f"✅ Deleted all {file_count} files for {symbol}"
        else:
            paths = get_model_paths(symbol, version)
            deleted_files = []
            for file_type, path in paths.items():
                if path.exists():
                    path.unlink()
                    deleted_files.append(file_type)
            
            if deleted_files:
                return f"✅ Deleted {symbol} version {version}: {', '.join(deleted_files)}"
            else:
                return f"❌ No files found for {symbol} version {version}"
        
    except Exception as e:
        return f"❌ Error deleting model: {str(e)}"

@tool
def model_performance_summary(symbol: str) -> str:
    """Get comprehensive performance summary of saved model"""
    try:
        models_dir = ensure_models_directory()
        symbol_dir = models_dir / symbol
        
        if not symbol_dir.exists():
            return f"❌ No saved models found for {symbol}"
        
        metadata_path = symbol_dir / "metadata_latest.json"
        if not metadata_path.exists():
            return f"❌ No metadata found for {symbol}"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return f"""
🤖 MODEL PERFORMANCE SUMMARY - {symbol}
- Type: {metadata.get('model_type', 'Unknown')}
- Trees: {metadata.get('n_estimators', 'N/A')}
- Features: {metadata.get('n_features', 'N/A')}
- Saved: {metadata.get('saved_at', 'Unknown')[:19]}
        """
        
    except Exception as e:
        return f"❌ Error getting performance summary: {str(e)}"

# Export all tools
__all__ = [
    'save_trained_model',
    'load_trained_model',
    'list_saved_models',
    'smart_predict_stock_price',
    'delete_saved_model', 
    'model_performance_summary'
]