# func.py - FIXED VERSION - Save visualizations to files instead of displaying
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # FIXED: Changed from DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler     # FIXED: Added for feature scaling
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import pickle
import json
from pathlib import Path

# Set matplotlib to non-interactive backend to avoid runtime errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from langchain_core.tools import tool

# Create plots directory if it doesn't exist
def ensure_plots_directory():
    """Create plots directory if it doesn't exist"""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    return 'plots'

# Original Stock Analysis Tools
@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic info"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else "N/A"
        
        return f"""
        {symbol}: ${current_price:.2f}
        Market Cap: {info.get('marketCap', 'N/A')}
        P/E Ratio: {info.get('trailingPE', 'N/A')}
        Volume: {info.get('volume', 'N/A')}
        52W High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
        52W Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
        """
    except Exception as e:
        return f"Error getting {symbol}: {str(e)}"

@tool
def get_technical_indicators(symbol: str) -> str:
    """Get technical analysis indicators"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo")
        
        current_price = hist['Close'].iloc[-1]
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        
        # RSI calculation
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        return f"""
        {symbol} Technical Analysis:
        Current: ${current_price:.2f}
        20-day MA: ${sma_20:.2f}
        50-day MA: ${sma_50:.2f}
        RSI: {rsi:.1f}
        Trend: {'Bullish' if current_price > sma_20 else 'Bearish'}
        """
    except Exception as e:
        return f"Error analyzing {symbol}: {str(e)}"

@tool
def get_stock_news(symbol: str) -> str:
    """Get recent news for stock"""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news[:3]
        
        if not news:
            return f"No recent news for {symbol}"
        
        result = f"Recent news for {symbol}:\n"
        for item in news:
            result += f"• {item.get('title', 'No title')}\n"
        
        return result
    except Exception as e:
        return f"Error getting news for {symbol}: {str(e)}"

@tool
def compare_stocks(symbols: str) -> str:
    """Compare multiple stocks (comma separated: AAPL,GOOGL,MSFT)"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        result = "Stock Comparison:\n" + "-" * 30 + "\n"
        
        for symbol in symbol_list[:5]:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")
            price = hist['Close'].iloc[-1] if not hist.empty else 0
            
            result += f"{symbol}: ${price:.2f} | P/E: {info.get('trailingPE', 'N/A')} | Cap: {info.get('marketCap', 'N/A')}\n"
        
        return result
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"

# FORECASTING TOOLS

@tool
def fetch_historical_data(symbol: str, period: str = "2y") -> str:
    """Fetch comprehensive historical stock data for feature engineering"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"No data found for {symbol}"
        
        info = f"""
Historical data for {symbol} ({period}):
- Date range: {hist.index[0].date()} to {hist.index[-1].date()}
- Total trading days: {len(hist)}
- Current price: ${hist['Close'].iloc[-1]:.2f}
- Price range: ${hist['Low'].min():.2f} - ${hist['High'].max():.2f}
- Average volume: {hist['Volume'].mean():,.0f}
        """
        return info
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"

# Replace these 3 functions COMPLETELY in your func.py file

@tool
def create_technical_features(symbol: str, period: str = "1y") -> str:
    """Create comprehensive technical analysis features for ML model - FINAL FIXED VERSION"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if len(data) < 50:
            return f"Insufficient data for {symbol}. Need at least 50 days."
        
        # Price-based features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Price_Change'] = data['Close'] - data['Open']
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open']
        
        # Moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Moving average ratios
        data['Price_MA5_Ratio'] = data['Close'] / data['MA_5']
        data['Price_MA20_Ratio'] = data['Close'] / data['MA_20']
        data['MA5_MA20_Ratio'] = data['MA_5'] / data['MA_20']
        
        # Volatility features
        data['Volatility_5'] = data['Returns'].rolling(window=5).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Volume features
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Price position features
        data['Price_Position_14'] = (data['Close'] - data['Close'].rolling(14).min()) / (data['Close'].rolling(14).max() - data['Close'].rolling(14).min())
        
        # Trend features
        data['Trend_5'] = data['Close'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
        data['Trend_10'] = data['Close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
        
        # FINAL FIX: Target variables - predict CHANGES not absolute prices
        data['Next_Price'] = data['Close'].shift(-1)
        data['Target_Change'] = data['Next_Price'] - data['Close']  # FIXED: Price change in dollars
        data['Target_Return'] = data['Returns'].shift(-1)
        data['Target_Price'] = data['Next_Price']  # Keep for compatibility
        
        # Save data globally
        globals()[f'{symbol}_feature_data'] = data
        
        feature_count = len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])
        
        # FINAL FIX: Show target variable info
        changes = data['Target_Change'].dropna()
        change_range = f"${changes.min():.2f} to ${changes.max():.2f}"
        change_std = changes.std()
        
        summary = f"""
FINAL FIXED Technical features created for {symbol}:
- Total features: {feature_count}
- Data points: {len(data)}
- Date range: {data.index[0].date()} to {data.index[-1].date()}

Key features include:
- Price ratios and returns
- Moving averages (5, 10, 20, 50 days)
- Volatility measures
- RSI, MACD, Bollinger Bands
- Volume indicators
- Trend signals

FINAL FIX - Target Variable:
- Predicting: PRICE CHANGES (not absolute prices)
- Change range: {change_range}
- Change std dev: ${change_std:.2f}
- This should give much better variance!

Data ready for FINAL FIXED model training!
        """
        return summary
        
    except Exception as e:
        return f"Error creating features for {symbol}: {str(e)}"

@tool
def train_decision_tree_model(symbol: str, test_size: float = 0.2, max_depth: int = 6) -> str:
    """Train a Random Forest model to predict stock price CHANGES - FINAL FIXED VERSION"""
    try:
        data_key = f'{symbol}_feature_data'
        if data_key not in globals():
            return f"No feature data found for {symbol}. Please run create_technical_features first."
        
        data = globals()[data_key].copy()
        
        feature_columns = [
            'Returns', 'Log_Returns', 'Price_Change', 'High_Low_Pct', 'Close_Open_Pct',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'Price_MA5_Ratio', 'Price_MA20_Ratio', 'MA5_MA20_Ratio',
            'Volatility_5', 'Volatility_20',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'Volume_Ratio', 'Price_Position_14',
            'Trend_5', 'Trend_10'
        ]
        
        data_clean = data.dropna()
        
        if len(data_clean) < 100:
            return f"Insufficient clean data for {symbol}. Need at least 100 rows."
        
        X = data_clean[feature_columns]
        y = data_clean['Target_Change']  # FINAL FIX: Predict price changes, not absolute prices
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # RandomForest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with scaled features
        model.fit(X_train_scaled, y_train)
        
        # Predict with scaled features
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # FINAL FIX: Calculate metrics on price changes
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model, scaler, and results
        globals()[f'{symbol}_model'] = model
        globals()[f'{symbol}_scaler'] = scaler
        globals()[f'{symbol}_feature_columns'] = feature_columns
        globals()[f'{symbol}_test_data'] = (X_test_scaled, y_test, y_pred_test)
        globals()[f'{symbol}_feature_importance'] = feature_importance
        
        # FINAL FIX: Better variance analysis for price changes
        pred_variance = np.var(y_pred_test)
        actual_variance = np.var(y_test)
        variance_ratio = pred_variance / actual_variance
        
        # FINAL FIX: Show prediction ranges for changes
        pred_range = f"${np.min(y_pred_test):.2f} to ${np.max(y_pred_test):.2f}"
        actual_range = f"${y_test.min():.2f} to ${y_test.max():.2f}"
        
        results = f"""
FINAL FIXED Random Forest Model Training Results for {symbol}:

Model Parameters:
- Algorithm: RandomForestRegressor (100 trees)
- Target: PRICE CHANGES (not absolute prices) ✅
- Max depth: {max_depth}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Feature scaling: StandardScaler applied

Performance Metrics (on price changes):
- Train RMSE: ${train_rmse:.2f}
- Test RMSE: ${test_rmse:.2f}
- Train MAE: ${train_mae:.2f}
- Test MAE: ${test_mae:.2f}
- Train R²: {train_r2:.3f}
- Test R²: {test_r2:.3f}

FINAL FIX - Prediction Analysis:
- Predicted change range: {pred_range}
- Actual change range: {actual_range}
- Prediction variance: {pred_variance:.2f}
- Actual variance: {actual_variance:.2f}
- Variance ratio: {variance_ratio:.3f} {'✅ (EXCELLENT!)' if variance_ratio > 0.3 else '✅ (Good)' if variance_ratio > 0.1 else '❌ (Still too flat)'}

Top 5 Most Important Features:
{feature_importance.head().to_string(index=False)}

✅ FINAL FIXED Model trained to predict price changes!
        """
        return results
        
    except Exception as e:
        return f"Error training model for {symbol}: {str(e)}"

@tool
def predict_stock_price(symbol: str, days_ahead: int = 1) -> str:
    """Predict future stock prices using price change predictions - FINAL FIXED VERSION"""
    try:
        model_key = f'{symbol}_model'
        if model_key not in globals():
            return f"No trained model found for {symbol}. Please train the model first."
        
        model = globals()[model_key]
        scaler = globals()[f'{symbol}_scaler']
        feature_columns = globals()[f'{symbol}_feature_columns']
        data = globals()[f'{symbol}_feature_data']
        
        # Get current price and features
        current_price = data['Close'].iloc[-1]
        latest_data = data.iloc[-1][feature_columns]
        
        # FINAL FIX: Predict price change, then add to current price
        latest_data_scaled = scaler.transform([latest_data])
        predicted_change = model.predict(latest_data_scaled)[0]  # FINAL FIX: This is a change in dollars
        predicted_price = current_price + predicted_change       # FINAL FIX: Add change to current price
        
        if f'{symbol}_test_data' in globals():
            _, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            recent_mae = mean_absolute_error(y_test[-20:], y_pred_test[-20:])
        else:
            recent_mae = abs(predicted_change) * 0.5
        
        prediction_change_pct = (predicted_change / current_price) * 100
        
        # FINAL FIX: Confidence intervals based on change predictions
        confidence_lower = predicted_price - recent_mae
        confidence_upper = predicted_price + recent_mae
        
        result = f"""
FINAL FIXED Stock Price Prediction for {symbol}:

Current Price: ${current_price:.2f}
Predicted Price ({days_ahead} day ahead): ${predicted_price:.2f}

FINAL FIX - Change Prediction:
- Predicted Change: ${predicted_change:.2f} ({prediction_change_pct:+.2f}%)
- Direction: {'📈 UP' if predicted_change > 0 else '📉 DOWN' if predicted_change < 0 else '➡️ FLAT'}

Confidence Interval (±1 MAE):
- Lower bound: ${confidence_lower:.2f}
- Upper bound: ${confidence_upper:.2f}

✅ FINAL FIX: Using RandomForest to predict price CHANGES!
        """
        return result
        
    except Exception as e:
        return f"Error predicting price for {symbol}: {str(e)}"

@tool
def backtest_model(symbol: str, start_date: str = "2023-01-01") -> str:
    """Perform backtesting on the trained model - FIXED VERSION"""
    try:
        model_key = f'{symbol}_model'
        if model_key not in globals():
            return f"No trained model found for {symbol}. Please train the model first."
        
        model = globals()[model_key]
        scaler = globals()[f'{symbol}_scaler']  # FIXED: Get scaler
        feature_columns = globals()[f'{symbol}_feature_columns']
        data = globals()[f'{symbol}_feature_data']
        
        data_backtest = data[data.index >= start_date].copy()
        
        if len(data_backtest) < 50:
            return f"Insufficient data for backtesting from {start_date}"
        
        data_clean = data_backtest.dropna()
        X = data_clean[feature_columns]
        y_actual = data_clean['Target_Price']
        
        # FIXED: Scale features for prediction
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        actual_direction = np.sign(data_clean['Target_Return'])
        pred_direction = np.sign(y_pred - data_clean['Close'])
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        data_clean['Predicted_Price'] = y_pred
        data_clean['Actual_Next_Price'] = y_actual
        data_clean['Prediction_Error'] = abs(y_pred - y_actual)
        data_clean['Prediction_Error_Pct'] = (data_clean['Prediction_Error'] / y_actual) * 100
        
        data_clean['Signal'] = np.where(y_pred > data_clean['Close'], 1, 0)
        data_clean['Strategy_Return'] = data_clean['Signal'].shift(1) * data_clean['Target_Return']
        data_clean['Cumulative_Strategy_Return'] = (1 + data_clean['Strategy_Return']).cumprod()
        data_clean['Cumulative_Market_Return'] = (1 + data_clean['Target_Return']).cumprod()
        
        strategy_total_return = (data_clean['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
        market_total_return = (data_clean['Cumulative_Market_Return'].iloc[-1] - 1) * 100
        
        globals()[f'{symbol}_backtest_results'] = data_clean
        
        # FIXED: Add variance analysis
        pred_variance = np.var(y_pred)
        actual_variance = np.var(y_actual)
        variance_ratio = pred_variance / actual_variance
        
        results = f"""
FIXED Backtesting Results for {symbol} (from {start_date}):

Model Performance:
- RMSE: ${rmse:.2f}
- MAE: ${mae:.2f}
- R² Score: {r2:.3f}
- Directional Accuracy: {directional_accuracy:.1f}%
- Prediction variance ratio: {variance_ratio:.3f} {'✅ (Good)' if variance_ratio > 0.1 else '❌ (Too flat)'}

Trading Strategy Performance:
- Strategy Total Return: {strategy_total_return:+.2f}%
- Market (Buy & Hold) Return: {market_total_return:+.2f}%
- Excess Return: {strategy_total_return - market_total_return:+.2f}%

Statistics:
- Total predictions: {len(data_clean)}
- Average prediction error: {data_clean['Prediction_Error_Pct'].mean():.2f}%
- Trading signals generated: {data_clean['Signal'].sum()}

✅ FIXED Backtest completed with RandomForest model!
        """
        return results
        
    except Exception as e:
        return f"Error during backtesting for {symbol}: {str(e)}"

@tool
def create_model_visualization(symbol: str, chart_type: str = "performance") -> str:
    """Create and save visualizations for model analysis - FIXED VERSION"""
    try:
        plots_dir = ensure_plots_directory()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if chart_type == "feature_importance":
            if f'{symbol}_feature_importance' not in globals():
                return f"No feature importance data for {symbol}. Train model first."
            
            feature_importance = globals()[f'{symbol}_feature_importance']
            
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['feature'].head(15), feature_importance['importance'].head(15))
            plt.title(f'Top 15 Feature Importance - {symbol} (RandomForest)')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            filename = f"{plots_dir}/{symbol}_feature_importance_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ FIXED Feature importance chart saved as: {filename}"
        
        elif chart_type == "prediction_vs_actual":
            if f'{symbol}_test_data' not in globals():
                return f"No test data for {symbol}. Train model first."
            
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.values[:50], label='Actual', alpha=0.7, linewidth=2)
            plt.plot(y_pred_test[:50], label='FIXED RandomForest Predicted', alpha=0.7, linewidth=2)
            plt.title(f'FIXED: Predicted vs Actual Prices - {symbol} (First 50 test samples)')
            plt.xlabel('Sample')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # FIXED: Add variance info to plot
            pred_var = np.var(y_pred_test[:50])
            actual_var = np.var(y_test.values[:50])
            plt.text(0.02, 0.98, f'Pred Var: {pred_var:.2f}\nActual Var: {actual_var:.2f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            filename = f"{plots_dir}/{symbol}_prediction_vs_actual_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ FIXED Prediction vs actual chart saved as: {filename}"
        
        elif chart_type == "backtest":
            if f'{symbol}_backtest_results' not in globals():
                return f"No backtest data for {symbol}. Run backtesting first."
            
            backtest_data = globals()[f'{symbol}_backtest_results']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Price and predictions
            ax1.plot(backtest_data.index, backtest_data['Close'], label='Actual Price', alpha=0.7)
            ax1.plot(backtest_data.index, backtest_data['Predicted_Price'], label='FIXED RandomForest Predicted', alpha=0.7)
            ax1.set_title(f'FIXED: {symbol} - Actual vs Predicted Prices')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Prediction errors
            ax2.plot(backtest_data.index, backtest_data['Prediction_Error_Pct'])
            ax2.set_title('FIXED: Prediction Error (%)')
            ax2.set_ylabel('Error %')
            ax2.grid(True, alpha=0.3)
            
            # Cumulative returns
            ax3.plot(backtest_data.index, backtest_data['Cumulative_Strategy_Return'], label='FIXED Strategy', linewidth=2)
            ax3.plot(backtest_data.index, backtest_data['Cumulative_Market_Return'], label='Buy & Hold', linewidth=2)
            ax3.set_title('FIXED: Cumulative Returns Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Trading signals
            signals = backtest_data[backtest_data['Signal'] == 1]
            ax4.plot(backtest_data.index, backtest_data['Close'], alpha=0.7, color='blue')
            ax4.scatter(signals.index, signals['Close'], color='green', marker='^', s=50, alpha=0.7)
            ax4.set_title('FIXED: Trading Signals (Green = Buy)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"{plots_dir}/{symbol}_backtest_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ FIXED Comprehensive backtest visualization saved as: {filename}"
        
        elif chart_type == "performance":
            if f'{symbol}_test_data' not in globals():
                return f"No model data for {symbol}. Train model first."
            
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            ax1.scatter(y_test, y_pred_test, alpha=0.6)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Price')
            ax1.set_ylabel('FIXED RandomForest Predicted Price')
            ax1.set_title(f'FIXED: {symbol} - Predicted vs Actual (Scatter)')
            ax1.grid(True, alpha=0.3)
            
            # Residuals
            residuals = y_test - y_pred_test
            ax2.scatter(y_pred_test, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('FIXED RandomForest Predicted Price')
            ax2.set_ylabel('Residuals')
            ax2.set_title('FIXED: Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"{plots_dir}/{symbol}_model_performance_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ FIXED Model performance visualization saved as: {filename}"
        
        elif chart_type == "all":
            # Create all visualizations at once
            results = []
            chart_types = ["performance", "feature_importance", "prediction_vs_actual", "backtest"]
            
            for ct in chart_types:
                try:
                    result = create_model_visualization(symbol, ct)
                    results.append(result)
                except Exception as e:
                    results.append(f"Failed to create {ct}: {str(e)}")
            
            return "✅ FIXED: All visualizations created:\n" + "\n".join(results)
        
        else:
            return f"Unknown chart type: {chart_type}. Available: 'performance', 'feature_importance', 'prediction_vs_actual', 'backtest', 'all'"
    
    except Exception as e:
        return f"Error creating visualization for {symbol}: {str(e)}"

@tool
def model_summary_report(symbol: str) -> str:
    """Generate a comprehensive summary report of the FIXED model and its performance"""
    try:
        report = f"COMPREHENSIVE FIXED MODEL REPORT - {symbol}\n"
        report += "=" * 60 + "\n\n"
        
        has_model = f'{symbol}_model' in globals()
        has_features = f'{symbol}_feature_data' in globals()
        has_backtest = f'{symbol}_backtest_results' in globals()
        has_importance = f'{symbol}_feature_importance' in globals()
        
        if not has_model:
            return f"No model found for {symbol}. Please train the model first."
        
        model = globals()[f'{symbol}_model']
        report += f"FIXED MODEL INFORMATION:\n"
        report += f"- Model type: {type(model).__name__} ✅\n"
        report += f"- Number of trees: {model.n_estimators}\n"
        report += f"- Max depth: {model.max_depth}\n"
        report += f"- Min samples split: {model.min_samples_split}\n"
        report += f"- Min samples leaf: {model.min_samples_leaf}\n"
        report += f"- Feature scaling: StandardScaler applied ✅\n\n"
        
        if has_features:
            data = globals()[f'{symbol}_feature_data']
            feature_columns = globals()[f'{symbol}_feature_columns']
            report += f"FEATURE INFORMATION:\n"
            report += f"- Total features: {len(feature_columns)}\n"
            report += f"- Data points: {len(data)}\n"
            report += f"- Date range: {data.index[0].date()} to {data.index[-1].date()}\n\n"
        
        if has_importance:
            feature_importance = globals()[f'{symbol}_feature_importance']
            report += f"TOP 10 IMPORTANT FEATURES:\n"
            for i, row in feature_importance.head(10).iterrows():
                report += f"  {i+1}. {row['feature']}: {row['importance']:.4f}\n"
            report += "\n"
        
        if f'{symbol}_test_data' in globals():
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            # FIXED: Add variance analysis
            pred_variance = np.var(y_pred_test)
            actual_variance = np.var(y_test)
            variance_ratio = pred_variance / actual_variance
            
            report += f"FIXED MODEL PERFORMANCE:\n"
            report += f"- RMSE: ${rmse:.2f}\n"
            report += f"- MAE: ${mae:.2f}\n"
            report += f"- R² Score: {r2:.3f}\n"
            report += f"- Prediction variance: {pred_variance:.2f}\n"
            report += f"- Variance ratio: {variance_ratio:.3f} {'✅ (Good)' if variance_ratio > 0.1 else '❌ (Too flat)'}\n\n"
        
        if has_backtest:
            backtest_data = globals()[f'{symbol}_backtest_results']
            strategy_return = (backtest_data['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
            market_return = (backtest_data['Cumulative_Market_Return'].iloc[-1] - 1) * 100
            
            report += f"FIXED BACKTESTING RESULTS:\n"
            report += f"- Strategy return: {strategy_return:+.2f}%\n"
            report += f"- Market return: {market_return:+.2f}%\n"
            report += f"- Excess return: {strategy_return - market_return:+.2f}%\n"
            report += f"- Total trades: {backtest_data['Signal'].sum()}\n\n"
        
        # Add visualization file paths if they exist
        plots_dir = 'plots'
        if os.path.exists(plots_dir):
            plot_files = [f for f in os.listdir(plots_dir) if f.startswith(symbol) and f.endswith('.png')]
            if plot_files:
                report += f"SAVED VISUALIZATIONS:\n"
                for plot_file in sorted(plot_files)[-4:]:
                    report += f"- {plots_dir}/{plot_file}\n"
                report += "\n"
        
        report += f"✅ FIXED Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
        
    except Exception as e:
        return f"Error generating summary report for {symbol}: {str(e)}"

# FIXED: Added quick test function
@tool
def quick_model_test(symbol: str) -> str:
    """Quick test to verify FIXED model is working properly"""
    try:
        if f'{symbol}_model' not in globals():
            return f"❌ No model found for {symbol}. Train model first."
        
        model = globals()[f'{symbol}_model']
        test_data = globals()[f'{symbol}_test_data']
        X_test, y_test, y_pred_test = test_data
        
        # Check prediction variance
        pred_variance = np.var(y_pred_test)
        actual_variance = np.var(y_test)
        variance_ratio = pred_variance / actual_variance
        
        # Check RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Check model type
        model_type = type(model).__name__
        is_random_forest = 'RandomForest' in model_type
        
        status = "✅ WORKING" if variance_ratio > 0.1 and rmse < 10 and is_random_forest else "❌ BROKEN"
        
        return f"""
FIXED Model Test Results for {symbol}:
Status: {status}

Model Analysis:
- Model type: {model_type} {'✅' if is_random_forest else '❌ (Should be RandomForest)'}
- Number of trees: {getattr(model, 'n_estimators', 'N/A')}
- Max depth: {model.max_depth}

Performance Metrics:
- Prediction variance: {pred_variance:.2f}
- Actual variance: {actual_variance:.2f}
- Variance ratio: {variance_ratio:.3f} {'✅ (Good)' if variance_ratio > 0.1 else '❌ (BAD - predictions too flat)'}
- RMSE: ${rmse:.2f} {'✅ (Good)' if rmse < 10 else '❌ (High error)'}

Prediction Range:
- Min prediction: ${np.min(y_pred_test):.2f}
- Max prediction: ${np.max(y_pred_test):.2f}
- Range: ${np.max(y_pred_test) - np.min(y_pred_test):.2f}

{'✅ FIXED Model is predicting properly with good variance!' if variance_ratio > 0.1 and is_random_forest else '❌ Model still has issues - check your changes!'}
        """
    except Exception as e:
        return f"Error testing model: {e}"
    
# ADD THESE FUNCTIONS TO YOUR func.py FILE
# Add these imports at the top of func.py with your other imports:

import joblib
import pickle
import json
from pathlib import Path

# ADD THESE FUNCTIONS AFTER YOUR EXISTING FUNCTIONS IN func.py:

def ensure_models_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    return models_dir

@tool
def save_trained_model(symbol: str, version: str = "latest") -> str:
    """Save the trained model and all components to disk permanently"""
    try:
        models_dir = ensure_models_directory()
        
        # Check if model exists in memory
        if f'{symbol}_model' not in globals():
            return f"❌ No trained model found for {symbol}. Train the model first."
        
        # Get all model components
        model = globals()[f'{symbol}_model']
        scaler = globals()[f'{symbol}_scaler']
        feature_columns = globals()[f'{symbol}_feature_columns']
        
        # Create symbol-specific directory
        symbol_dir = models_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Save model components
        model_path = symbol_dir / f"model_{version}.joblib"
        scaler_path = symbol_dir / f"scaler_{version}.joblib"
        features_path = symbol_dir / f"features_{version}.json"
        metadata_path = symbol_dir / f"metadata_{version}.json"
        
        # Save using joblib (faster and more reliable for sklearn models)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save feature columns and metadata as JSON
        with open(features_path, 'w') as f:
            json.dump(feature_columns, f)
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "version": version,
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, 'n_estimators', 'N/A'),
            "max_depth": model.max_depth,
            "n_features": len(feature_columns),
            "saved_at": datetime.now().isoformat(),
            "feature_columns": feature_columns
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create/update latest versions
        latest_model = symbol_dir / "model_latest.joblib"
        latest_scaler = symbol_dir / "scaler_latest.joblib"
        latest_features = symbol_dir / "features_latest.json"
        latest_metadata = symbol_dir / "metadata_latest.json"
        
        # Remove existing latest files if they exist
        for latest_file in [latest_model, latest_scaler, latest_features, latest_metadata]:
            if latest_file.exists():
                latest_file.unlink()
        
        # Copy versioned files to latest
        import shutil
        shutil.copy2(model_path, latest_model)
        shutil.copy2(scaler_path, latest_scaler)
        shutil.copy2(features_path, latest_features)
        shutil.copy2(metadata_path, latest_metadata)
        
        file_sizes = {
            "model": f"{model_path.stat().st_size / 1024:.1f} KB",
            "scaler": f"{scaler_path.stat().st_size / 1024:.1f} KB"
        }
        
        return f"""
✅ Model saved permanently for {symbol}!

📁 Saved Files:
- Model: {model_path} ({file_sizes['model']})
- Scaler: {scaler_path} ({file_sizes['scaler']})
- Features: {features_path}
- Metadata: {metadata_path}

🤖 Model Info:
- Type: {metadata['model_type']}
- Trees: {metadata['n_estimators']}
- Max Depth: {metadata['max_depth']}
- Features: {metadata['n_features']}
- Version: {version}

💾 Model persisted and ready for production deployment!
        """
        
    except Exception as e:
        return f"❌ Error saving model for {symbol}: {str(e)}"

@tool
def load_trained_model(symbol: str, version: str = "latest") -> str:
    """Load a previously trained model from disk"""
    try:
        models_dir = Path('models')
        symbol_dir = models_dir / symbol
        
        if not symbol_dir.exists():
            return f"❌ No saved models found for {symbol}. Train and save a model first."
        
        # Construct file paths
        model_path = symbol_dir / f"model_{version}.joblib"
        scaler_path = symbol_dir / f"scaler_{version}.joblib"
        features_path = symbol_dir / f"features_{version}.json"
        metadata_path = symbol_dir / f"metadata_{version}.json"
        
        # Check if all files exist
        missing_files = []
        for path, name in [(model_path, 'model'), (scaler_path, 'scaler'), 
                          (features_path, 'features'), (metadata_path, 'metadata')]:
            if not path.exists():
                missing_files.append(name)
        
        if missing_files:
            return f"❌ Missing files for {symbol} version {version}: {missing_files}"
        
        # Load model components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            feature_columns = json.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Store in global variables (same as training)
        globals()[f'{symbol}_model'] = model
        globals()[f'{symbol}_scaler'] = scaler
        globals()[f'{symbol}_feature_columns'] = feature_columns
        
        return f"""
✅ Model loaded successfully for {symbol}!

🤖 Loaded Model Info:
- Type: {metadata['model_type']}
- Trees: {metadata.get('n_estimators', 'N/A')}
- Max Depth: {metadata['max_depth']}
- Features: {metadata['n_features']}
- Version: {metadata['version']}
- Saved: {metadata['saved_at']}

🚀 Model ready for predictions without retraining!
        """
        
    except Exception as e:
        return f"❌ Error loading model for {symbol}: {str(e)}"

@tool
def list_saved_models() -> str:
    """List all saved models and their versions"""
    try:
        models_dir = Path('models')
        
        if not models_dir.exists():
            return "❌ No models directory found. No models have been saved yet."
        
        result = "📁 SAVED MODELS INVENTORY:\n" + "=" * 60 + "\n"
        
        total_models = 0
        for symbol_dir in models_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                result += f"\n📊 {symbol}:\n"
                
                # Find all model versions
                model_files = list(symbol_dir.glob("model_*.joblib"))
                
                if not model_files:
                    result += "   No models found\n"
                    continue
                
                total_models += len(model_files)
                for model_file in sorted(model_files):
                    version = model_file.stem.replace("model_", "")
                    metadata_file = symbol_dir / f"metadata_{version}.json"
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        size = f"{model_file.stat().st_size / 1024:.1f} KB"
                        saved_date = metadata.get('saved_at', 'Unknown')[:10]
                        model_type = metadata.get('model_type', 'Unknown')
                        n_trees = metadata.get('n_estimators', 'N/A')
                        
                        result += f"   • {version}: {model_type} ({n_trees} trees), {size}, {saved_date}\n"
                    else:
                        result += f"   • {version}: (no metadata available)\n"
        
        result += f"\n📈 Total Models: {total_models}"
        result += f"\n💡 Usage: load_trained_model(symbol='AAPL', version='latest')"
        result += f"\n🎯 Predict: smart_predict_stock_price(symbol='AAPL')"
        
        return result
        
    except Exception as e:
        return f"❌ Error listing models: {str(e)}"

@tool
def delete_saved_model(symbol: str, version: str = None) -> str:
    """Delete saved model(s) for a symbol"""
    try:
        models_dir = Path('models')
        symbol_dir = models_dir / symbol
        
        if not symbol_dir.exists():
            return f"❌ No saved models found for {symbol}"
        
        if version is None:
            # Delete entire symbol directory
            import shutil
            shutil.rmtree(symbol_dir)
            return f"✅ All models deleted for {symbol}"
        else:
            # Delete specific version
            files_to_delete = [
                symbol_dir / f"model_{version}.joblib",
                symbol_dir / f"scaler_{version}.joblib",
                symbol_dir / f"features_{version}.json",
                symbol_dir / f"metadata_{version}.json"
            ]
            
            deleted_count = 0
            for file_path in files_to_delete:
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
            
            return f"✅ Deleted {deleted_count} files for {symbol} version {version}"
        
    except Exception as e:
        return f"❌ Error deleting model: {str(e)}"

@tool 
def smart_predict_stock_price(symbol: str, days_ahead: int = 1) -> str:
    """Smart prediction that auto-loads model if not in memory"""
    try:
        # Check if model is already in memory
        if f'{symbol}_model' not in globals():
            # Try to load from disk
            load_result = load_trained_model.invoke({"symbol": symbol})
            
            if "❌" in load_result:
                return f"""
❌ No trained model found for {symbol} in memory or on disk.

🎯 Options:
1. Train a new model: create_technical_features('{symbol}') then train_decision_tree_model('{symbol}')
2. Check available models: list_saved_models()
3. Load existing model: load_trained_model('{symbol}')

{load_result}
                """
            else:
                # Model loaded successfully, continue with prediction
                pass
        
        # Now make prediction using existing function
        return predict_stock_price.invoke({"symbol": symbol, "days_ahead": days_ahead})
        
    except Exception as e:
        return f"❌ Error in smart prediction for {symbol}: {str(e)}"

@tool
def model_performance_summary(symbol: str) -> str:
    """Get a quick performance summary of the saved model"""
    try:
        models_dir = Path('models')
        symbol_dir = models_dir / symbol
        metadata_path = symbol_dir / "metadata_latest.json"
        
        if not metadata_path.exists():
            return f"❌ No model metadata found for {symbol}"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if model is loaded for additional stats
        if f'{symbol}_model' in globals():
            model = globals()[f'{symbol}_model']
            
            # Get test data if available
            if f'{symbol}_test_data' in globals():
                X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
                
                # Calculate current performance
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                
                pred_variance = np.var(y_pred_test)
                actual_variance = np.var(y_test)
                variance_ratio = pred_variance / actual_variance
                
                performance_info = f"""
📊 Live Performance Metrics:
- RMSE: ${rmse:.2f}
- MAE: ${mae:.2f}
- R² Score: {r2:.3f}
- Variance Ratio: {variance_ratio:.3f} {'✅' if variance_ratio > 0.1 else '❌'}
"""
            else:
                performance_info = "\n📊 Load model and run predictions to see live performance metrics."
        else:
            performance_info = "\n📊 Load model to see detailed performance metrics."
        
        return f"""
🤖 MODEL PERFORMANCE SUMMARY - {symbol}

📁 Model Info:
- Type: {metadata.get('model_type', 'Unknown')}
- Trees: {metadata.get('n_estimators', 'N/A')}
- Max Depth: {metadata.get('max_depth', 'N/A')}
- Features: {metadata.get('n_features', 'N/A')}
- Saved: {metadata.get('saved_at', 'Unknown')[:19]}

{performance_info}

🎯 Quick Actions:
- Load: load_trained_model('{symbol}')
- Predict: smart_predict_stock_price('{symbol}')
- Visualize: create_model_visualization('{symbol}', 'all')
        """
        
    except Exception as e:
        return f"❌ Error getting performance summary for {symbol}: {str(e)}"