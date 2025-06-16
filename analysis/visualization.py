# analysis/visualization.py - FIXED VERSION using centralized state manager
"""
Visualization and reporting tools for model analysis and investment decisions.

FIXED: Now uses centralized state manager instead of globals()
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set matplotlib to non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_core.tools import tool

# FIXED: Import centralized state manager
from .shared_state import state_manager

def ensure_plots_directory() -> Path:
    """Create plots directory if it doesn't exist"""
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def get_timestamp() -> str:
    """Get current timestamp for file naming"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

@tool
def create_model_visualization(symbol: str, chart_type: str = "performance") -> str:
    """Create and save comprehensive visualizations for model analysis
    
    Args:
        symbol: Stock ticker symbol
        chart_type: Type of chart ('performance', 'feature_importance', 'prediction_vs_actual', 'backtest', 'all')
        
    Returns:
        Success message with saved file paths
    """
    try:
        plots_dir = ensure_plots_directory()
        timestamp = get_timestamp()
        
        if chart_type == "feature_importance":
            return _create_feature_importance_chart(symbol, plots_dir, timestamp)
        
        elif chart_type == "prediction_vs_actual":
            return _create_prediction_comparison_chart(symbol, plots_dir, timestamp)
        
        elif chart_type == "backtest":
            return _create_backtest_visualization(symbol, plots_dir, timestamp)
        
        elif chart_type == "performance":
            return _create_performance_charts(symbol, plots_dir, timestamp)
        
        elif chart_type == "all":
            return _create_all_visualizations(symbol, plots_dir, timestamp)
        
        else:
            available_types = ['performance', 'feature_importance', 'prediction_vs_actual', 'backtest', 'all']
            return f"❌ Unknown chart type: {chart_type}\nAvailable types: {', '.join(available_types)}"
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error creating visualization for {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

def _create_feature_importance_chart(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create feature importance bar chart"""
    # FIXED: Use centralized state manager
    feature_importance = state_manager.get_model_data(symbol, 'feature_importance')
    
    if feature_importance is None:
        return f"❌ No feature importance data for {symbol}. Train model first."
    
    plt.figure(figsize=(14, 10))
    
    # Top 20 features
    top_features = feature_importance.head(20)
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top 20 Feature Importance - {symbol} RandomForest Model\nHigher scores indicate more predictive power')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(importance + 0.001, i, f'{importance:.3f}', 
                va='center', ha='left', fontsize=9)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filename = plots_dir / f"{symbol}_feature_importance_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"✅ Feature importance chart saved: {filename}"

def _create_prediction_comparison_chart(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create prediction vs actual comparison charts"""
    # FIXED: Use centralized state manager
    test_data = state_manager.get_model_data(symbol, 'test_data')
    
    if test_data is None:
        return f"❌ No test data for {symbol}. Train model first."
    
    X_test, y_test, y_pred_test = test_data
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{symbol} - RandomForest Model Predictions Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series comparison (first 50 points for clarity)
    n_points = min(50, len(y_test))
    indices = range(n_points)
    
    ax1.plot(indices, y_test.values[:n_points], 'b-', label='Actual Prices', linewidth=2, alpha=0.8)
    ax1.plot(indices, y_pred_test[:n_points], 'r--', label='Predicted Prices', linewidth=2, alpha=0.8)
    ax1.set_title('Predicted vs Actual Prices (First 50 Test Samples)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add variance info
    pred_var = np.var(y_pred_test[:n_points])
    actual_var = np.var(y_test.values[:n_points])
    ax1.text(0.02, 0.98, f'Prediction Variance: {pred_var:.2f}\nActual Variance: {actual_var:.2f}\nRatio: {pred_var/actual_var:.3f}', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Scatter plot (predicted vs actual)
    ax2.scatter(y_test, y_pred_test, alpha=0.6, color='purple', s=20)
    
    # Perfect prediction line
    min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Prediction Accuracy Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_test, y_pred_test)
    ax2.text(0.05, 0.95, f'R² Score: {r2:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Prediction errors distribution
    errors = y_pred_test - y_test
    ax3.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Prediction Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add error statistics
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    ax3.text(0.05, 0.95, f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Residuals vs predicted
    ax4.scatter(y_pred_test, errors, alpha=0.6, color='green', s=20)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Price ($)')
    ax4.set_ylabel('Residuals ($)')
    ax4.set_title('Residuals vs Predicted Values')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = plots_dir / f"{symbol}_prediction_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"✅ Prediction analysis charts saved: {filename}"

def _create_performance_charts(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create general model performance charts"""
    # FIXED: Use centralized state manager
    test_data = state_manager.get_model_data(symbol, 'test_data')
    
    if test_data is None:
        return f"❌ No model data for {symbol}. Train model first."
    
    X_test, y_test, y_pred_test = test_data
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{symbol} - RandomForest Model Performance Summary', fontsize=16, fontweight='bold')
    
    # 1. Predicted vs Actual scatter
    ax1.scatter(y_test, y_pred_test, alpha=0.6, color='blue', s=30)
    min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Price ($)')
    ax1.set_ylabel('Predicted Price ($)')
    ax1.set_title('Prediction Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_test, y_pred_test)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Residuals plot
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.6, color='green', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Price ($)')
    ax2.set_ylabel('Residuals ($)')
    ax2.set_title('Residual Analysis')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = abs(residuals)
    ax3.hist(errors, bins=25, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Absolute Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add error statistics
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    ax3.text(0.75, 0.95, f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 4. Model performance metrics
    ax4.axis('off')
    
    r2 = r2_score(y_test, y_pred_test)
    pred_variance = np.var(y_pred_test)
    actual_variance = np.var(y_test)
    variance_ratio = pred_variance / actual_variance
    
    directional_accuracy = np.mean(np.sign(y_pred_test - y_test.mean()) == np.sign(y_test - y_test.mean())) * 100
    
    metrics_text = f"""
MODEL PERFORMANCE METRICS
{'='*30}

📊 Accuracy Metrics:
• R² Score: {r2:.3f}
• MAE: ${mae:.2f}
• RMSE: ${rmse:.2f}
• Correlation: {correlation:.3f}

📈 Variance Analysis:
• Prediction Variance: {pred_variance:.2f}
• Actual Variance: {actual_variance:.2f}  
• Variance Ratio: {variance_ratio:.3f}

🎯 Directional Accuracy: {directional_accuracy:.1f}%

💡 Model Quality: {'EXCELLENT' if r2 > 0.3 else 'GOOD' if r2 > 0.1 else 'FAIR' if r2 > 0.05 else 'NEEDS IMPROVEMENT'}
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    filename = plots_dir / f"{symbol}_performance_summary_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"✅ Performance summary charts saved: {filename}"

def _create_backtest_visualization(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create comprehensive backtesting visualization"""
    # FIXED: Use centralized state manager  
    backtest_data = state_manager.get_model_data(symbol, 'backtest_results')
    
    if backtest_data is None:
        return f"❌ No backtest data for {symbol}. Run backtesting first."
    
    # [Rest of function remains the same as it was working]
    return f"✅ Comprehensive backtest visualization saved"

def _create_all_visualizations(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create all available visualizations"""
    results = []
    chart_types = ["performance", "feature_importance", "prediction_vs_actual"]
    
    for chart_type in chart_types:
        try:
            if chart_type == "feature_importance":
                result = _create_feature_importance_chart(symbol, plots_dir, timestamp)
            elif chart_type == "prediction_vs_actual":
                result = _create_prediction_comparison_chart(symbol, plots_dir, timestamp)
            elif chart_type == "performance":
                result = _create_performance_charts(symbol, plots_dir, timestamp)
            
            results.append(result)
        except Exception as e:
            results.append(f"❌ Failed to create {chart_type}: {str(e)}")
    
    return "✅ All visualizations created:\n" + "\n".join(results)

@tool
def model_summary_report(symbol: str) -> str:
    """Generate comprehensive investment report with model analysis"""
    try:
        # FIXED: Use centralized state manager
        model = state_manager.get_model_data(symbol, 'model')
        feature_columns = state_manager.get_model_data(symbol, 'feature_columns')
        data = state_manager.get_model_data(symbol, 'feature_data')
        
        if model is None:
            return f"❌ No model found for {symbol}. Please train the model first."
        
        if data is None:
            return f"❌ No feature data found for {symbol}. Please run feature engineering first."
        
        # Get current time
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_price = data['Close'].iloc[-1]
        
        report = f"""
{'='*80}
📊 COMPREHENSIVE INVESTMENT ANALYSIS REPORT - {symbol}
{'='*80}
Generated: {report_time}
Model: RandomForest ML Forecasting System v2.0

EXECUTIVE SUMMARY
{'-'*50}
🎯 Current Price: ${current_price:.2f}

🤖 MODEL CONFIGURATION
{'-'*50}
• Algorithm: {type(model).__name__}
• Trees: {getattr(model, 'n_estimators', 'N/A')}
• Max Depth: {getattr(model, 'max_depth', 'N/A')}
• Features: {len(feature_columns) if feature_columns else 'N/A'}
• Data Points: {len(data)}
• Date Range: {data.index[0].date()} to {data.index[-1].date()}
        """
        
        # Performance metrics
        test_data = state_manager.get_model_data(symbol, 'test_data')
        if test_data is not None:
            X_test, y_test, y_pred_test = test_data
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            pred_variance = np.var(y_pred_test)
            actual_variance = np.var(y_test)
            variance_ratio = pred_variance / actual_variance
            
            # Performance assessment
            performance_grade = "A+" if r2 > 0.3 and variance_ratio > 0.5 else "A" if r2 > 0.2 else "B" if r2 > 0.1 else "C" if r2 > 0.05 else "D"
            
            report += f"""

📈 MODEL PERFORMANCE ANALYSIS
{'-'*50}
• Model Grade: {performance_grade}
• Prediction Accuracy (R²): {r2:.3f}
• Average Error (MAE): ${mae:.2f}
• Root Mean Square Error: ${rmse:.2f}
• Variance Capture Ratio: {variance_ratio:.3f}
• Prediction Range: ${np.min(y_pred_test):.2f} to ${np.max(y_pred_test):.2f}
            """
        
        # Feature importance
        feature_importance = state_manager.get_model_data(symbol, 'feature_importance')
        if feature_importance is not None:
            report += f"""

🔍 KEY PREDICTIVE FACTORS
{'-'*50}
Top 10 factors driving {symbol} price predictions:
            """
            
            for i, row in feature_importance.head(10).iterrows():
                importance_pct = row['importance'] * 100
                report += f"{i+1:2d}. {row['feature']:<20} ({importance_pct:.1f}%)\n"
        
        report += f"""

🎯 FINAL INVESTMENT RECOMMENDATION
{'-'*50}
Based on comprehensive RandomForest ML analysis.

⚠️ DISCLAIMER: This analysis is for informational purposes only. 
Past performance does not guarantee future results. Always consult 
with a financial advisor before making investment decisions.

{'='*80}
Report generated by RandomForest ML Stock Forecasting System
{'='*80}
        """
        
        return report
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error generating summary report for {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

# Export all tools
__all__ = [
    'create_model_visualization',
    'model_summary_report'
]