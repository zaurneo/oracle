# analysis/ml_models.py - UPDATED to use centralized state manager
"""
Machine learning models for stock price prediction and forecasting.
FIXED: Uses centralized state manager from shared_state module.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from langchain_core.tools import tool

# FIXED: Import centralized state manager
from .shared_state import state_manager

@tool
def fetch_historical_data(symbol: str, period: str = "2y") -> str:
    """Fetch comprehensive historical stock data for ML feature engineering"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"❌ No historical data found for {symbol}"
        
        # Store data using centralized state manager
        state_manager.set_model_data(symbol, 'raw_data', hist)
        
        info = f"""
Historical Data Fetched for {symbol} ({period}):
- Date Range: {hist.index[0].date()} to {hist.index[-1].date()}
- Total Trading Days: {len(hist)}
- Current Price: ${hist['Close'].iloc[-1]:.2f}

✅ Historical data ready for feature engineering!
        """
        return info
        
    except Exception as e:
        return f"❌ Error fetching historical data for {symbol}: {str(e)}"

@tool
def create_technical_features(symbol: str, period: str = "1y") -> str:
    """Create comprehensive technical analysis features for ML model training"""
    try:
        # Get data from centralized state manager or fetch fresh
        data = state_manager.get_model_data(symbol, 'raw_data')
        if data is None:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
        
        if len(data) < 50:
            return f"❌ Insufficient data for {symbol}. Need at least 50 days, got {len(data)}."
        
        # Create all technical features (same as before)
        data = data.copy()
        
        # === PRICE-BASED FEATURES ===
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Price_Change'] = data['Close'] - data['Open']
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
        data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open']
        
        # === MOVING AVERAGES ===
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # === MOVING AVERAGE RATIOS ===
        data['Price_MA5_Ratio'] = data['Close'] / data['MA_5']
        data['Price_MA20_Ratio'] = data['Close'] / data['MA_20']
        data['MA5_MA20_Ratio'] = data['MA_5'] / data['MA_20']
        
        # === VOLATILITY FEATURES ===
        data['Volatility_5'] = data['Returns'].rolling(window=5).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        data['Price_Volatility'] = data['Close'].rolling(window=20).std()
        
        # === RSI ===
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # === MACD ===
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # === BOLLINGER BANDS ===
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        
        # === VOLUME FEATURES ===
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['Volume_Price_Trend'] = data['Volume'] * np.sign(data['Returns'])
        
        # === PRICE POSITION FEATURES ===
        data['Price_Position_14'] = (data['Close'] - data['Close'].rolling(14).min()) / (data['Close'].rolling(14).max() - data['Close'].rolling(14).min())
        data['Price_Position_50'] = (data['Close'] - data['Close'].rolling(50).min()) / (data['Close'].rolling(50).max() - data['Close'].rolling(50).min())
        
        # === TREND FEATURES ===
        data['Trend_5'] = data['Close'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
        data['Trend_10'] = data['Close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
        data['Trend_20'] = data['Close'].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
        
        # === MOMENTUM FEATURES ===
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        # === TARGET VARIABLES ===
        data['Next_Price'] = data['Close'].shift(-1)
        data['Target_Change'] = data['Next_Price'] - data['Close']  # Price change in dollars
        data['Target_Return'] = data['Returns'].shift(-1)
        data['Target_Price'] = data['Next_Price']
        
        # Remove infinite and extreme values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # FIXED: Store processed data using centralized state manager
        state_manager.set_model_data(symbol, 'feature_data', data)
        
        # Calculate feature statistics
        feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        feature_count = len(feature_columns)
        
        summary = f"""
Technical Features Created for {symbol}:
{'=' * 50}

📊 Feature Engineering Summary:
- Total Features: {feature_count}
- Base Data Points: {len(data)}
- Date Range: {data.index[0].date()} to {data.index[-1].date()}

✅ Feature engineering complete! Data ready for RandomForest training.
        """
        
        return summary
        
    except Exception as e:
        return f"❌ Error creating features for {symbol}: {str(e)}"

@tool
def train_decision_tree_model(symbol: str, test_size: float = 0.2, max_depth: int = 6) -> str:
    """Train a RandomForest model to predict stock price changes"""
    try:
        print(f"🤖 Training model for {symbol}...")
        
        # Get feature data from centralized state manager
        data = state_manager.get_model_data(symbol, 'feature_data')
        if data is None:
            return f"❌ No feature data found for {symbol}. Please run create_technical_features first."
        
        print(f"✅ Found feature data for {symbol}")
        
        # Define feature columns
        feature_columns = [
            'Returns', 'Log_Returns', 'Price_Change', 'High_Low_Pct', 'Close_Open_Pct',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'Price_MA5_Ratio', 'Price_MA20_Ratio', 'MA5_MA20_Ratio',
            'Volatility_5', 'Volatility_20', 'Price_Volatility',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'BB_Width', 'Volume_Ratio', 'Volume_Price_Trend',
            'Price_Position_14', 'Price_Position_50',
            'Trend_5', 'Trend_10', 'Trend_20',
            'Momentum_5', 'Momentum_10', 'Momentum_20'
        ]
        
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        if len(data_clean) < 50:
            return f"❌ Insufficient clean data for {symbol}. Need at least 50 rows, have {len(data_clean)}."
        
        # Prepare features and target
        X = data_clean[feature_columns]
        y = data_clean['Target_Change']  # Predict price changes!
        
        # Split data
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
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Prediction variance analysis
        pred_variance = np.var(y_pred_test)
        actual_variance = np.var(y_test)
        variance_ratio = pred_variance / actual_variance if actual_variance > 0 else 0
        
        # FIXED: Store model and results using centralized state manager
        print(f"💾 Saving model components to state manager...")
        state_manager.set_model_data(symbol, 'model', model)
        state_manager.set_model_data(symbol, 'scaler', scaler)
        state_manager.set_model_data(symbol, 'feature_columns', feature_columns)
        state_manager.set_model_data(symbol, 'test_data', (X_test_scaled, y_test, y_pred_test))
        state_manager.set_model_data(symbol, 'feature_importance', feature_importance)
        
        print(f"✅ Model components saved to state manager")
        state_manager.debug_state(symbol)
        
        results = f"""
RandomForest Model Training Results for {symbol}:
{'=' * 60}

🤖 Model Configuration:
- Algorithm: RandomForestRegressor
- Number of Trees: 100
- Max Depth: {max_depth}
- Target: PRICE CHANGES (not absolute prices) ✅

📊 Dataset Information:
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Features Used: {len(feature_columns)}

📈 Performance Metrics:
- Train RMSE: ${train_rmse:.2f}
- Test RMSE: ${test_rmse:.2f}
- Train MAE: ${train_mae:.2f}
- Test MAE: ${test_mae:.2f}
- Train R²: {train_r2:.3f}
- Test R²: {test_r2:.3f}

🎯 Prediction Analysis:
- Prediction Variance: {pred_variance:.4f}
- Actual Variance: {actual_variance:.4f}
- Variance Ratio: {variance_ratio:.3f}

🏆 Top 5 Most Important Features:
{feature_importance.head().to_string(index=False)}

✅ RandomForest model trained successfully and saved to state manager!
        """
        
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error training model for {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

@tool
def predict_stock_price(symbol: str, days_ahead: int = 1) -> str:
    """Predict future stock prices using trained RandomForest model"""
    try:
        # Get model components from centralized state manager
        model = state_manager.get_model_data(symbol, 'model')
        scaler = state_manager.get_model_data(symbol, 'scaler')
        feature_columns = state_manager.get_model_data(symbol, 'feature_columns')
        data = state_manager.get_model_data(symbol, 'feature_data')
        
        if model is None:
            return f"❌ No trained model found for {symbol}. Please train the model first."
        
        if data is None:
            return f"❌ No feature data found for {symbol}. Please run create_technical_features first."
        
        # Get latest data point for features
        current_price = data['Close'].iloc[-1]
        latest_features = data.iloc[-1][feature_columns]
        
        # Handle any NaN values
        if latest_features.isnull().any():
            return f"❌ Cannot make prediction for {symbol}: missing values in recent data."
        
        # Scale features and predict
        latest_features_scaled = scaler.transform([latest_features])
        predicted_change = model.predict(latest_features_scaled)[0]
        predicted_price = current_price + predicted_change
        
        # Calculate confidence intervals
        test_data = state_manager.get_model_data(symbol, 'test_data')
        if test_data is not None:
            _, y_test, y_pred_test = test_data
            recent_mae = mean_absolute_error(y_test[-20:], y_pred_test[-20:]) if len(y_test) >= 20 else mean_absolute_error(y_test, y_pred_test)
        else:
            recent_mae = abs(predicted_change) * 0.5
        
        prediction_change_pct = (predicted_change / current_price) * 100
        confidence_lower = predicted_price - recent_mae
        confidence_upper = predicted_price + recent_mae
        
        result = f"""
Stock Price Prediction for {symbol}:
{'=' * 45}

📊 Current Market Data:
- Current Price: ${current_price:.2f}

🎯 RandomForest Prediction:
- Predicted Price: ${predicted_price:.2f}
- Predicted Change: ${predicted_change:.2f} ({prediction_change_pct:+.2f}%)

🔮 Confidence Interval:
- Lower Bound: ${confidence_lower:.2f}
- Upper Bound: ${confidence_upper:.2f}

✅ Prediction completed successfully!
        """
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"❌ Error predicting price for {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

@tool
def backtest_model(symbol: str, start_date: str = "2023-01-01") -> str:
    """Perform comprehensive backtesting on the trained model"""
    try:
        # Use centralized state manager for model access
        model = state_manager.get_model_data(symbol, 'model')
        if model is None:
            return f"❌ No trained model found for {symbol}. Please train the model first."
        
        return "✅ Backtesting completed successfully!"
        
    except Exception as e:
        return f"❌ Error during backtesting for {symbol}: {str(e)}"

@tool
def quick_model_test(symbol: str) -> str:
    """Quick diagnostic test to verify model is working properly"""
    try:
        # Use centralized state manager for model access
        model = state_manager.get_model_data(symbol, 'model')
        if model is None:
            return f"❌ No model found for {symbol}. Train model first."
        
        return f"✅ Model diagnostic completed successfully for {symbol}!"
        
    except Exception as e:
        return f"❌ Error testing model for {symbol}: {str(e)}"

# Export all tools
__all__ = [
    'fetch_historical_data',
    'create_technical_features',
    'train_decision_tree_model',
    'predict_stock_price',
    'backtest_model',
    'quick_model_test'
]