# stock_forecasting_tools.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from langchain_core.tools import tool

@tool
def fetch_historical_data(symbol: str, period: str = "2y") -> str:
    """
    Fetch comprehensive historical stock data for feature engineering.
    period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"No data found for {symbol}"
        
        # Basic info
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

@tool
def create_technical_features(symbol: str, period: str = "1y") -> str:
    """
    Create comprehensive technical analysis features for ML model.
    Returns feature summary and saves data internally for model training.
    """
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
        
        # Target variable (next day's closing price)
        data['Target_Price'] = data['Close'].shift(-1)
        data['Target_Return'] = data['Returns'].shift(-1)
        
        # Save data globally for model training (in production, use a database)
        globals()[f'{symbol}_feature_data'] = data
        
        # Summary
        feature_count = len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])
        
        summary = f"""
Technical features created for {symbol}:
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

Target variables:
- Next day closing price
- Next day return

Data ready for model training!
        """
        
        return summary
        
    except Exception as e:
        return f"Error creating features for {symbol}: {str(e)}"

@tool
def train_decision_tree_model(symbol: str, test_size: float = 0.2, max_depth: int = 10) -> str:
    """
    Train a decision tree model to predict stock prices.
    Uses previously created features.
    """
    try:
        # Get the feature data
        data_key = f'{symbol}_feature_data'
        if data_key not in globals():
            return f"No feature data found for {symbol}. Please run create_technical_features first."
        
        data = globals()[data_key].copy()
        
        # Prepare features
        feature_columns = [
            'Returns', 'Log_Returns', 'Price_Change', 'High_Low_Pct', 'Close_Open_Pct',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'Price_MA5_Ratio', 'Price_MA20_Ratio', 'MA5_MA20_Ratio',
            'Volatility_5', 'Volatility_20',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Position', 'Volume_Ratio', 'Price_Position_14',
            'Trend_5', 'Trend_10'
        ]
        
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        if len(data_clean) < 100:
            return f"Insufficient clean data for {symbol}. Need at least 100 rows."
        
        # Prepare X and y
        X = data_clean[feature_columns]
        y = data_clean['Target_Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Train model
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
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
        
        # Save model and results
        globals()[f'{symbol}_model'] = model
        globals()[f'{symbol}_feature_columns'] = feature_columns
        globals()[f'{symbol}_test_data'] = (X_test, y_test, y_pred_test)
        globals()[f'{symbol}_feature_importance'] = feature_importance
        
        results = f"""
Decision Tree Model Training Results for {symbol}:

Model Parameters:
- Max depth: {max_depth}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}

Performance Metrics:
- Train RMSE: ${train_rmse:.2f}
- Test RMSE: ${test_rmse:.2f}
- Train MAE: ${train_mae:.2f}
- Test MAE: ${test_mae:.2f}
- Train R²: {train_r2:.3f}
- Test R²: {test_r2:.3f}

Top 5 Most Important Features:
{feature_importance.head().to_string(index=False)}

Model trained successfully and ready for predictions!
        """
        
        return results
        
    except Exception as e:
        return f"Error training model for {symbol}: {str(e)}"

@tool
def predict_stock_price(symbol: str, days_ahead: int = 1) -> str:
    """
    Predict future stock prices using the trained decision tree model.
    """
    try:
        # Check if model exists
        model_key = f'{symbol}_model'
        if model_key not in globals():
            return f"No trained model found for {symbol}. Please train the model first."
        
        model = globals()[model_key]
        feature_columns = globals()[f'{symbol}_feature_columns']
        data = globals()[f'{symbol}_feature_data']
        
        # Get the latest data point
        latest_data = data.iloc[-1][feature_columns]
        
        # Make prediction
        predicted_price = model.predict([latest_data])[0]
        current_price = data['Close'].iloc[-1]
        
        # Calculate prediction confidence based on recent model performance
        if f'{symbol}_test_data' in globals():
            _, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            recent_mae = mean_absolute_error(y_test[-20:], y_pred_test[-20:])  # Last 20 predictions
        else:
            recent_mae = abs(predicted_price - current_price) * 0.1  # Rough estimate
        
        prediction_change = predicted_price - current_price
        prediction_change_pct = (prediction_change / current_price) * 100
        
        # Simple confidence intervals (±1 MAE)
        confidence_lower = predicted_price - recent_mae
        confidence_upper = predicted_price + recent_mae
        
        result = f"""
Stock Price Prediction for {symbol}:

Current Price: ${current_price:.2f}
Predicted Price ({days_ahead} day ahead): ${predicted_price:.2f}

Change: ${prediction_change:.2f} ({prediction_change_pct:+.2f}%)

Confidence Interval (±1 MAE):
- Lower bound: ${confidence_lower:.2f}
- Upper bound: ${confidence_upper:.2f}

Note: Predictions are based on current technical indicators.
Market conditions can change rapidly.
        """
        
        return result
        
    except Exception as e:
        return f"Error predicting price for {symbol}: {str(e)}"

@tool
def backtest_model(symbol: str, start_date: str = "2023-01-01") -> str:
    """
    Perform backtesting on the trained model.
    start_date format: YYYY-MM-DD
    """
    try:
        # Check if model exists
        model_key = f'{symbol}_model'
        if model_key not in globals():
            return f"No trained model found for {symbol}. Please train the model first."
        
        model = globals()[model_key]
        feature_columns = globals()[f'{symbol}_feature_columns']
        data = globals()[f'{symbol}_feature_data']
        
        # Filter data from start_date
        data_backtest = data[data.index >= start_date].copy()
        
        if len(data_backtest) < 50:
            return f"Insufficient data for backtesting from {start_date}"
        
        # Prepare data
        data_clean = data_backtest.dropna()
        X = data_clean[feature_columns]
        y_actual = data_clean['Target_Price']
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        # Calculate directional accuracy
        actual_direction = np.sign(data_clean['Target_Return'])
        pred_direction = np.sign(y_pred - data_clean['Close'])
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Calculate trading simulation
        data_clean['Predicted_Price'] = y_pred
        data_clean['Actual_Next_Price'] = y_actual
        data_clean['Prediction_Error'] = abs(y_pred - y_actual)
        data_clean['Prediction_Error_Pct'] = (data_clean['Prediction_Error'] / y_actual) * 100
        
        # Simple trading strategy: buy if predicted price > current price
        data_clean['Signal'] = np.where(y_pred > data_clean['Close'], 1, 0)  # 1 = buy, 0 = hold/sell
        data_clean['Strategy_Return'] = data_clean['Signal'].shift(1) * data_clean['Target_Return']
        data_clean['Cumulative_Strategy_Return'] = (1 + data_clean['Strategy_Return']).cumprod()
        data_clean['Cumulative_Market_Return'] = (1 + data_clean['Target_Return']).cumprod()
        
        strategy_total_return = (data_clean['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
        market_total_return = (data_clean['Cumulative_Market_Return'].iloc[-1] - 1) * 100
        
        # Save backtest results
        globals()[f'{symbol}_backtest_results'] = data_clean
        
        results = f"""
Backtesting Results for {symbol} (from {start_date}):

Model Performance:
- RMSE: ${rmse:.2f}
- MAE: ${mae:.2f}
- R² Score: {r2:.3f}
- Directional Accuracy: {directional_accuracy:.1f}%

Trading Strategy Performance:
- Strategy Total Return: {strategy_total_return:+.2f}%
- Market (Buy & Hold) Return: {market_total_return:+.2f}%
- Excess Return: {strategy_total_return - market_total_return:+.2f}%

Statistics:
- Total predictions: {len(data_clean)}
- Average prediction error: {data_clean['Prediction_Error_Pct'].mean():.2f}%
- Trading signals generated: {data_clean['Signal'].sum()}

Backtest completed successfully!
        """
        
        return results
        
    except Exception as e:
        return f"Error during backtesting for {symbol}: {str(e)}"

@tool
def create_model_visualization(symbol: str, chart_type: str = "performance") -> str:
    """
    Create visualizations for model analysis.
    chart_type: 'performance', 'feature_importance', 'prediction_vs_actual', 'backtest'
    """
    try:
        if chart_type == "feature_importance":
            if f'{symbol}_feature_importance' not in globals():
                return f"No feature importance data for {symbol}. Train model first."
            
            feature_importance = globals()[f'{symbol}_feature_importance']
            
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['feature'].head(15), feature_importance['importance'].head(15))
            plt.title(f'Top 15 Feature Importance - {symbol}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return f"Feature importance chart created for {symbol}"
        
        elif chart_type == "prediction_vs_actual":
            if f'{symbol}_test_data' not in globals():
                return f"No test data for {symbol}. Train model first."
            
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.values[:50], label='Actual', alpha=0.7)
            plt.plot(y_pred_test[:50], label='Predicted', alpha=0.7)
            plt.title(f'Predicted vs Actual Prices - {symbol} (First 50 test samples)')
            plt.xlabel('Sample')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return f"Prediction vs actual chart created for {symbol}"
        
        elif chart_type == "backtest":
            if f'{symbol}_backtest_results' not in globals():
                return f"No backtest data for {symbol}. Run backtesting first."
            
            backtest_data = globals()[f'{symbol}_backtest_results']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Price and predictions
            ax1.plot(backtest_data.index, backtest_data['Close'], label='Actual Price', alpha=0.7)
            ax1.plot(backtest_data.index, backtest_data['Predicted_Price'], label='Predicted Price', alpha=0.7)
            ax1.set_title(f'{symbol} - Actual vs Predicted Prices')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Prediction errors
            ax2.plot(backtest_data.index, backtest_data['Prediction_Error_Pct'])
            ax2.set_title('Prediction Error (%)')
            ax2.set_ylabel('Error %')
            ax2.grid(True, alpha=0.3)
            
            # Cumulative returns
            ax3.plot(backtest_data.index, backtest_data['Cumulative_Strategy_Return'], label='Strategy', linewidth=2)
            ax3.plot(backtest_data.index, backtest_data['Cumulative_Market_Return'], label='Buy & Hold', linewidth=2)
            ax3.set_title('Cumulative Returns Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Trading signals
            signals = backtest_data[backtest_data['Signal'] == 1]
            ax4.plot(backtest_data.index, backtest_data['Close'], alpha=0.7, color='blue')
            ax4.scatter(signals.index, signals['Close'], color='green', marker='^', s=50, alpha=0.7)
            ax4.set_title('Trading Signals (Green = Buy)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return f"Comprehensive backtest visualization created for {symbol}"
        
        elif chart_type == "performance":
            # Performance summary chart
            if f'{symbol}_test_data' not in globals():
                return f"No model data for {symbol}. Train model first."
            
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            ax1.scatter(y_test, y_pred_test, alpha=0.6)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Price')
            ax1.set_ylabel('Predicted Price')
            ax1.set_title(f'{symbol} - Predicted vs Actual (Scatter)')
            ax1.grid(True, alpha=0.3)
            
            # Residuals
            residuals = y_test - y_pred_test
            ax2.scatter(y_pred_test, residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Price')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return f"Model performance visualization created for {symbol}"
        
        else:
            return f"Unknown chart type: {chart_type}. Available: 'performance', 'feature_importance', 'prediction_vs_actual', 'backtest'"
    
    except Exception as e:
        return f"Error creating visualization for {symbol}: {str(e)}"

@tool
def model_summary_report(symbol: str) -> str:
    """
    Generate a comprehensive summary report of the model and its performance.
    """
    try:
        report = f"COMPREHENSIVE MODEL REPORT - {symbol}\n"
        report += "=" * 60 + "\n\n"
        
        # Check what data is available
        has_model = f'{symbol}_model' in globals()
        has_features = f'{symbol}_feature_data' in globals()
        has_backtest = f'{symbol}_backtest_results' in globals()
        has_importance = f'{symbol}_feature_importance' in globals()
        
        if not has_model:
            return f"No model found for {symbol}. Please train the model first."
        
        # Model information
        model = globals()[f'{symbol}_model']
        report += f"MODEL INFORMATION:\n"
        report += f"- Model type: Decision Tree Regressor\n"
        report += f"- Max depth: {model.max_depth}\n"
        report += f"- Min samples split: {model.min_samples_split}\n"
        report += f"- Min samples leaf: {model.min_samples_leaf}\n\n"
        
        # Feature information
        if has_features:
            data = globals()[f'{symbol}_feature_data']
            feature_columns = globals()[f'{symbol}_feature_columns']
            report += f"FEATURE INFORMATION:\n"
            report += f"- Total features: {len(feature_columns)}\n"
            report += f"- Data points: {len(data)}\n"
            report += f"- Date range: {data.index[0].date()} to {data.index[-1].date()}\n\n"
        
        # Feature importance
        if has_importance:
            feature_importance = globals()[f'{symbol}_feature_importance']
            report += f"TOP 10 IMPORTANT FEATURES:\n"
            for i, row in feature_importance.head(10).iterrows():
                report += f"  {i+1}. {row['feature']}: {row['importance']:.4f}\n"
            report += "\n"
        
        # Model performance
        if f'{symbol}_test_data' in globals():
            X_test, y_test, y_pred_test = globals()[f'{symbol}_test_data']
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            report += f"MODEL PERFORMANCE:\n"
            report += f"- RMSE: ${rmse:.2f}\n"
            report += f"- MAE: ${mae:.2f}\n"
            report += f"- R² Score: {r2:.3f}\n\n"
        
        # Backtesting results
        if has_backtest:
            backtest_data = globals()[f'{symbol}_backtest_results']
            strategy_return = (backtest_data['Cumulative_Strategy_Return'].iloc[-1] - 1) * 100
            market_return = (backtest_data['Cumulative_Market_Return'].iloc[-1] - 1) * 100
            
            report += f"BACKTESTING RESULTS:\n"
            report += f"- Strategy return: {strategy_return:+.2f}%\n"
            report += f"- Market return: {market_return:+.2f}%\n"
            report += f"- Excess return: {strategy_return - market_return:+.2f}%\n"
            report += f"- Total trades: {backtest_data['Signal'].sum()}\n\n"
        
        # Current prediction
        try:
            latest_prediction = predict_stock_price(symbol, 1)
            report += f"LATEST PREDICTION:\n{latest_prediction}\n"
        except:
            report += f"LATEST PREDICTION: Unable to generate\n\n"
        
        report += f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
        
    except Exception as e:
        return f"Error generating summary report for {symbol}: {str(e)}"