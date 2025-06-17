# analysis/__init__.py - FIXED VERSION - No stub function fallback
"""
Stock Analysis Package - FIXED to prevent stub function fallback
"""

print("🔄 Loading analysis package...")

# Import centralized state manager first
state_manager = None
ModelStateManager = None

try:
    from .shared_state import ModelStateManager, state_manager
    print("✅ Centralized state manager imported successfully")
except ImportError as e:
    print(f"❌ Could not import state manager: {e}")
    raise  # Don't continue with fallback

# Import all tools - FIXED: No fallback to stubs
print("🔄 Importing analysis tools...")

try:
    from .basic_tools import (
        get_stock_price,
        get_stock_news, 
        compare_stocks,
        get_technical_indicators
    )
    print("✅ Basic tools imported")
    
    from .persistence import (
        save_trained_model,
        load_trained_model,
        list_saved_models,
        smart_predict_stock_price,
        delete_saved_model,
        model_performance_summary
    )
    print("✅ Persistence tools imported")
    
    from .ml_models import (
        fetch_historical_data,
        create_technical_features,
        train_decision_tree_model,
        predict_stock_price,
        backtest_model,
        quick_model_test
    )
    print("✅ ML tools imported")

    from .visualization import (
        create_model_visualization,
        model_summary_report
    )
    print("✅ Visualization tools imported")
    
    from .html_generator import (
        collect_analysis_data,
        gather_visualization_files,
        create_html_report,
        collect_multi_stock_data,
        gather_multi_stock_visualizations,
        create_comparative_html_report
    )
    print("✅ HTML generator tools imported")
    
    tools_imported = True
    print("✅ All analysis modules imported successfully (no fallback to stubs)")
    
except ImportError as e:
    print(f"❌ CRITICAL: Analysis tools import failed: {e}")
    print("❌ Cannot continue without real tools - check module syntax")
    import traceback
    traceback.print_exc()
    raise  # Don't use stub functions
except Exception as e:
    print(f"❌ CRITICAL: Analysis tools error: {e}")
    import traceback
    traceback.print_exc()
    raise

# Export all tools (REMOVED stub functions completely)
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
    'model_summary_report',
    
    # HTML Generator
    'collect_analysis_data',
    'gather_visualization_files',
    'create_html_report',
    'collect_multi_stock_data',
    'gather_multi_stock_visualizations',
    'create_comparative_html_report'
]

print(f"✅ Analysis package ready with {len(__all__)} tools")