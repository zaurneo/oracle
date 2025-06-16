# analysis/visualization.py - Visualization and Reporting Tools
"""
Visualization and reporting tools for model analysis and investment decisions.

This module provides:
- Model performance visualizations
- Feature importance charts
- Backtesting result plots
- Investment summary reports
- Executive-level analysis documents
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
        return f"❌ Error creating visualization for {symbol}: {str(e)}"

def _create_feature_importance_chart(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create feature importance bar chart"""
    if f'{symbol}_feature_importance' not in globals():
        return f"❌ No feature importance data for {symbol}. Train model first."
    
    feature_importance = globals()[f'{symbol}_feature_importance']
    
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
    if f'{symbol}_test_data' not in globals():
        return f"❌ No test data for {symbol}. Train model first."
    
    X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
    
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

def _create_backtest_visualization(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create comprehensive backtesting visualization"""
    if f'{symbol}_backtest_results' not in globals():
        return f"❌ No backtest data for {symbol}. Run backtesting first."
    
    backtest_data = globals()[f'{symbol}_backtest_results']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f'{symbol} - Comprehensive Backtesting Analysis', fontsize=18, fontweight='bold')
    
    # 1. Price chart with predictions (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(backtest_data.index, backtest_data['Close'], 'b-', label='Actual Price', linewidth=2, alpha=0.8)
    ax1.plot(backtest_data.index, backtest_data['Predicted_Price'], 'r--', label='Predicted Price', linewidth=2, alpha=0.8)
    ax1.set_title('Actual vs Predicted Prices Over Time')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction errors over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(backtest_data.index, backtest_data['Prediction_Error_Pct'], 'orange', alpha=0.7)
    ax2.set_title('Prediction Error %')
    ax2.set_ylabel('Error (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Cumulative returns comparison (spans 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(backtest_data.index, (backtest_data['Cumulative_Strategy_Return'] - 1) * 100, 
             'g-', label='ML Strategy', linewidth=3)
    ax3.plot(backtest_data.index, (backtest_data['Cumulative_Market_Return'] - 1) * 100, 
             'b-', label='Buy & Hold', linewidth=3, alpha=0.7)
    ax3.set_title('Cumulative Returns Comparison')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add performance metrics
    strategy_return = (backtest_data['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
    market_return = (backtest_data['Cumulative_Market_Return'].iloc[-1] - 1) * 100
    excess_return = strategy_return - market_return
    
    ax3.text(0.02, 0.98, f'Strategy: {strategy_return:+.1f}%\nMarket: {market_return:+.1f}%\nExcess: {excess_return:+.1f}%', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4. Trading signals
    ax4 = fig.add_subplot(gs[1, 2])
    signals = backtest_data[backtest_data['Signal'] != 0]
    buy_signals = signals[signals['Signal'] == 1]
    sell_signals = signals[signals['Signal'] == -1]
    
    ax4.plot(backtest_data.index, backtest_data['Close'], 'b-', alpha=0.5, linewidth=1)
    if len(buy_signals) > 0:
        ax4.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', 
                   s=60, alpha=0.8, label=f'Buy ({len(buy_signals)})')
    if len(sell_signals) > 0:
        ax4.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', 
                   s=60, alpha=0.8, label=f'Sell ({len(sell_signals)})')
    
    ax4.set_title('Trading Signals')
    ax4.set_ylabel('Price ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance metrics summary (spans 3 columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate additional metrics
    total_trades = (backtest_data['Signal'] != 0).sum()
    profitable_trades = (backtest_data['Strategy_Return'] > 0).sum()
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_error = backtest_data['Prediction_Error_Pct'].mean()
    max_error = backtest_data['Prediction_Error_Pct'].max()
    
    volatility = backtest_data['Strategy_Return'].std() * np.sqrt(252)
    sharpe = (backtest_data['Strategy_Return'].mean() / backtest_data['Strategy_Return'].std() * np.sqrt(252)) if backtest_data['Strategy_Return'].std() > 0 else 0
    
    metrics_text = f"""
BACKTESTING PERFORMANCE SUMMARY
{'='*50}

📊 MODEL PERFORMANCE                    💰 TRADING PERFORMANCE                 📈 RISK METRICS
• Average Error: {avg_error:.2f}%              • Total Trades: {total_trades}                    • Volatility: {volatility:.1%}
• Max Error: {max_error:.2f}%                  • Win Rate: {win_rate:.1f}%                      • Sharpe Ratio: {sharpe:.2f}
• Prediction Accuracy: {100-avg_error:.1f}%     • Strategy Return: {strategy_return:+.2f}%           • Max Drawdown: {(backtest_data['Cumulative_Strategy_Return'].cummax() / backtest_data['Cumulative_Strategy_Return'] - 1).max():.1%}

🎯 RECOMMENDATION: {'STRONG BUY' if excess_return > 10 else 'BUY' if excess_return > 5 else 'HOLD' if excess_return > 0 else 'AVOID'} 
   Strategy {'outperformed' if excess_return > 0 else 'underperformed'} market by {abs(excess_return):.1f} percentage points
    """
    
    ax5.text(0.02, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    filename = plots_dir / f"{symbol}_backtest_comprehensive_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return f"✅ Comprehensive backtest visualization saved: {filename}"

def _create_performance_charts(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create general model performance charts"""
    if f'{symbol}_test_data' not in globals():
        return f"❌ No model data for {symbol}. Train model first."
    
    X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
    
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

def _create_all_visualizations(symbol: str, plots_dir: Path, timestamp: str) -> str:
    """Create all available visualizations"""
    results = []
    chart_types = ["performance", "feature_importance", "prediction_vs_actual", "backtest"]
    
    for chart_type in chart_types:
        try:
            if chart_type == "feature_importance":
                result = _create_feature_importance_chart(symbol, plots_dir, timestamp)
            elif chart_type == "prediction_vs_actual":
                result = _create_prediction_comparison_chart(symbol, plots_dir, timestamp)
            elif chart_type == "backtest":
                result = _create_backtest_visualization(symbol, plots_dir, timestamp)
            elif chart_type == "performance":
                result = _create_performance_charts(symbol, plots_dir, timestamp)
            
            results.append(result)
        except Exception as e:
            results.append(f"❌ Failed to create {chart_type}: {str(e)}")
    
    return "✅ All visualizations created:\n" + "\n".join(results)

@tool
def model_summary_report(symbol: str) -> str:
    """Generate comprehensive investment report with model analysis
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Detailed investment report with model insights
    """
    try:
        if f'{symbol}_model' not in globals():
            return f"❌ No model found for {symbol}. Please train the model first."
        
        # Get current time
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Basic model info
        model = globals()[f'{symbol}_model']
        feature_columns = globals()[f'{symbol}_feature_columns']
        data = globals()[f'{symbol}_feature_data']
        
        current_price = data['Close'].iloc[-1]
        
        report = f"""
{'='*80}
📊 COMPREHENSIVE INVESTMENT ANALYSIS REPORT - {symbol}
{'='*80}
Generated: {report_time}
Model: RandomForest ML Forecasting System v2.0

EXECUTIVE SUMMARY
{'-'*50}
        """
        
        # Get prediction for executive summary
        try:
            from .ml_models import predict_stock_price
            prediction_result = predict_stock_price.invoke({"symbol": symbol})
            
            # Extract key prediction info
            if "Predicted Price" in prediction_result:
                lines = prediction_result.split('\n')
                for line in lines:
                    if "Predicted Price" in line:
                        pred_price_line = line
                    elif "Direction:" in line:
                        direction_line = line
                
                report += f"""
🎯 INVESTMENT RECOMMENDATION: Based on RandomForest ML Analysis
Current Price: ${current_price:.2f}
{pred_price_line.strip()}
{direction_line.strip() if 'direction_line' in locals() else ''}
                """
        except:
            report += f"\n🎯 Current Price: ${current_price:.2f}"
        
        # Model information
        report += f"""

🤖 MODEL CONFIGURATION
{'-'*50}
• Algorithm: {type(model).__name__}
• Trees: {getattr(model, 'n_estimators', 'N/A')}
• Max Depth: {getattr(model, 'max_depth', 'N/A')}
• Features: {len(feature_columns)}
• Data Points: {len(data)}
• Date Range: {data.index[0].date()} to {data.index[-1].date()}
        """
        
        # Performance metrics
        if f'{symbol}_test_data' in globals():
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            
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
        if f'{symbol}_feature_importance' in globals():
            feature_importance = globals()[f'{symbol}_feature_importance']
            
            report += f"""

🔍 KEY PREDICTIVE FACTORS
{'-'*50}
Top 10 factors driving {symbol} price predictions:
            """
            
            for i, row in feature_importance.head(10).iterrows():
                importance_pct = row['importance'] * 100
                report += f"{i+1:2d}. {row['feature']:<20} ({importance_pct:.1f}%)\n"
        
        # Backtesting results
        if f'{symbol}_backtest_results' in globals():
            backtest_data = globals()[f'{symbol}_backtest_results']
            
            strategy_return = (backtest_data['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
            market_return = (backtest_data['Cumulative_Market_Return'].iloc[-1] - 1) * 100
            excess_return = strategy_return - market_return
            
            total_trades = (backtest_data['Signal'] != 0).sum()
            profitable_trades = (backtest_data['Strategy_Return'] > 0).sum()
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Strategy assessment
            strategy_grade = "EXCELLENT" if excess_return > 10 else "GOOD" if excess_return > 5 else "FAIR" if excess_return > 0 else "POOR"
            
            report += f"""

💰 TRADING STRATEGY PERFORMANCE
{'-'*50}
• Strategy Grade: {strategy_grade}
• ML Strategy Return: {strategy_return:+.2f}%
• Buy & Hold Return: {market_return:+.2f}%
• Excess Return: {excess_return:+.2f}%
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}%
• Recommendation: {'STRONG BUY' if excess_return > 10 else 'BUY' if excess_return > 5 else 'HOLD' if excess_return > 0 else 'AVOID'}
            """
        
        # Risk assessment
        volatility = data['Returns'].std() * np.sqrt(252)  # Annualized volatility
        recent_volatility = data['Returns'].tail(30).std() * np.sqrt(252)
        
        risk_level = "HIGH" if volatility > 0.4 else "MEDIUM" if volatility > 0.2 else "LOW"
        
        report += f"""

⚠️ RISK ANALYSIS
{'-'*50}
• Annual Volatility: {volatility:.1%} ({risk_level} RISK)
• Recent Volatility (30d): {recent_volatility:.1%}
• Price Range (1Y): ${data['Low'].min():.2f} - ${data['High'].max():.2f}
• Current vs 1Y High: {(current_price / data['High'].max() - 1) * 100:+.1f}%
• Current vs 1Y Low: {(current_price / data['Low'].min() - 1) * 100:+.1f}%
        """
        
        # Technical signals
        latest_data = data.iloc[-1]
        rsi = latest_data.get('RSI', 'N/A')
        ma_20 = latest_data.get('MA_20', current_price)
        
        report += f"""

📊 CURRENT TECHNICAL SIGNALS
{'-'*50}
• Price vs 20-day MA: {'BULLISH' if current_price > ma_20 else 'BEARISH'} ({(current_price/ma_20-1)*100:+.1f}%)
• RSI Indicator: {rsi:.1f if isinstance(rsi, (int, float)) else rsi} {'(Overbought)' if isinstance(rsi, (int, float)) and rsi > 70 else '(Oversold)' if isinstance(rsi, (int, float)) and rsi < 30 else '(Neutral)' if isinstance(rsi, (int, float)) else ''}
• Recent Momentum: {((current_price / data['Close'].iloc[-6]) - 1) * 100:+.1f}% (5-day)
        """
        
        # Investment recommendation
        report += f"""

🎯 FINAL INVESTMENT RECOMMENDATION
{'-'*50}
Based on comprehensive RandomForest ML analysis combining:
✓ Technical pattern recognition across {len(feature_columns)} features
✓ Historical performance validation
✓ Risk-adjusted return analysis
✓ Current market conditions

RECOMMENDATION: {'STRONG BUY' if excess_return > 10 else 'BUY' if excess_return > 5 else 'HOLD' if excess_return > 0 else 'REVIEW' if f'{symbol}_backtest_results' in globals() else 'ANALYZE'}

⚠️ DISCLAIMER: This analysis is for informational purposes only. 
Past performance does not guarantee future results. Always consult 
with a financial advisor before making investment decisions.

{'='*80}
Report generated by RandomForest ML Stock Forecasting System
{'='*80}
        """
        
        return report
        
    except Exception as e:
        return f"❌ Error generating summary report for {symbol}: {str(e)}"

# Export all tools
__all__ = [
    'create_model_visualization',
    'model_summary_report'
]