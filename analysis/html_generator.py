# analysis/html_generator.py - Simplified HTML Generator with Plotly
"""
Simple HTML Generator that consolidates all analysis results into a clean HTML file with interactive Plotly charts.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

from langchain_core.tools import tool

# Import state manager
try:
    from .shared_state import state_manager
except ImportError:
    # Fallback state manager
    class FallbackStateManager:
        def __init__(self):
            self._data = {}
        def set_model_data(self, symbol, key, value):
            if symbol not in self._data:
                self._data[symbol] = {}
            self._data[symbol][key] = value
        def get_model_data(self, symbol, key, default=None):
            return self._data.get(symbol, {}).get(key, default)
    state_manager = FallbackStateManager()

def ensure_html_directory() -> Path:
    """Create HTML reports directory if it doesn't exist"""
    html_dir = Path('html_reports')
    html_dir.mkdir(exist_ok=True)
    return html_dir

def create_price_chart(symbol: str, data: pd.DataFrame) -> str:
    """Create interactive Plotly price chart"""
    try:
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price and Volume',
            yaxis_title='Price ($)',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        # Convert to JSON
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating price chart: {e}")
        return None

def create_prediction_chart(symbol: str, test_data: tuple) -> str:
    """Create interactive prediction vs actual chart"""
    try:
        if test_data is None:
            return None
            
        X_test, y_test, y_pred = test_data
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Model Predictions vs Actual',
            xaxis_title='Test Sample Index',
            yaxis_title='Price Change ($)',
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating prediction chart: {e}")
        return None

def create_feature_importance_chart(symbol: str, feature_importance: pd.DataFrame) -> str:
    """Create interactive feature importance chart"""
    try:
        if feature_importance is None:
            return None
            
        # Get top 15 features
        top_features = feature_importance.head(15)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features['importance'],
                colorscale='viridis'
            )
        ))
        
        fig.update_layout(
            title=f'{symbol} Top Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=500,
            margin=dict(l=150)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating feature importance chart: {e}")
        return None

def create_performance_metrics_chart(symbol: str, metrics: dict) -> str:
    """Create performance metrics visualization"""
    try:
        if not metrics:
            return None
            
        # Create gauge charts for key metrics
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=['R² Score', 'RMSE', 'MAE']
        )
        
        # R² Score gauge
        r2_score = metrics.get('r2_score', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=r2_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "R² Score"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 0.5], 'color': "lightgray"},
                       {'range': [0.5, 0.8], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 
                                'thickness': 0.75, 'value': 0.9}}
        ), row=1, col=1)
        
        # RMSE gauge
        rmse = metrics.get('rmse', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=rmse,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "RMSE"},
            gauge={'axis': {'range': [None, 10]},
                   'bar': {'color': "darkgreen"}}
        ), row=1, col=2)
        
        # MAE gauge
        mae = metrics.get('mae', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=mae,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "MAE"},
            gauge={'axis': {'range': [None, 10]},
                   'bar': {'color': "darkorange"}}
        ), row=1, col=3)
        
        fig.update_layout(height=300, showlegend=False)
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating performance metrics chart: {e}")
        return None

@tool
def collect_all_results(symbols: str) -> str:
    """Collect all analysis results for given stock symbols including data for Plotly charts
    
    Args:
        symbols: Comma-separated list of stock symbols (e.g., 'AAPL' or 'AAPL,GOOGL,TSLA')
        
    Returns:
        Summary of collected data
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        all_results = {}
        
        for symbol in symbol_list:
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'basic_data': {},
                'ml_analysis': {},
                'charts': {},
                'raw_data': None,
                'test_data': None,
                'feature_importance': None
            }
            
            # Collect basic data
            current_price = state_manager.get_model_data(symbol, 'current_price')
            if current_price:
                results['basic_data']['current_price'] = current_price
            
            raw_data = state_manager.get_model_data(symbol, 'raw_data')
            if raw_data is not None:
                results['basic_data']['data_points'] = len(raw_data)
                results['basic_data']['date_range'] = f"{raw_data.index[0].date()} to {raw_data.index[-1].date()}"
                results['raw_data'] = raw_data  # Store for chart generation
            
            # Collect ML analysis
            model_performance = state_manager.get_model_data(symbol, 'model_performance')
            if model_performance:
                results['ml_analysis']['performance'] = model_performance
            
            predictions = state_manager.get_model_data(symbol, 'predictions')
            if predictions:
                results['ml_analysis']['predictions'] = predictions
            
            # Collect data for charts
            test_data = state_manager.get_model_data(symbol, 'test_data')
            if test_data:
                results['test_data'] = test_data
            
            feature_importance = state_manager.get_model_data(symbol, 'feature_importance')
            if feature_importance is not None:
                results['feature_importance'] = feature_importance
                # Also store top features for display
                top_features = feature_importance.head(5).to_dict('records')
                results['ml_analysis']['top_features'] = top_features
            
            all_results[symbol] = results
        
        # Store collected results
        state_manager.set_model_data('HTML_REPORT', 'all_results', all_results)
        
        return f"✅ Collected comprehensive results for {len(symbol_list)} stocks: {', '.join(symbol_list)}"
        
    except Exception as e:
        return f"❌ Error collecting results: {str(e)}"

@tool 
def create_simple_html_report(title: str = "Stock Analysis Report") -> str:
    """Create a simple HTML report with all collected results and interactive Plotly charts
    
    Args:
        title: Report title
        
    Returns:
        Success message with file location
    """
    try:
        html_dir = ensure_html_directory()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get collected results
        all_results = state_manager.get_model_data('HTML_REPORT', 'all_results', {})
        
        if not all_results:
            return "❌ No results collected. Run collect_all_results first."
        
        # Generate charts for each stock
        charts_data = {}
        for symbol, results in all_results.items():
            charts_data[symbol] = {}
            
            # Price chart
            if results.get('raw_data') is not None:
                price_chart = create_price_chart(symbol, results['raw_data'])
                if price_chart:
                    charts_data[symbol]['price_chart'] = price_chart
            
            # Prediction chart
            if results.get('test_data') is not None:
                pred_chart = create_prediction_chart(symbol, results['test_data'])
                if pred_chart:
                    charts_data[symbol]['prediction_chart'] = pred_chart
            
            # Feature importance chart
            if results.get('feature_importance') is not None:
                feature_chart = create_feature_importance_chart(symbol, results['feature_importance'])
                if feature_chart:
                    charts_data[symbol]['feature_chart'] = feature_chart
            
            # Performance metrics chart
            if results.get('ml_analysis', {}).get('performance'):
                metrics_chart = create_performance_metrics_chart(symbol, results['ml_analysis']['performance'])
                if metrics_chart:
                    charts_data[symbol]['metrics_chart'] = metrics_chart
        
        # Create HTML with Plotly charts
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stock-section {{
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
        }}
        .metric {{
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 3px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            color: #333;
            font-size: 1.1em;
        }}
        .chart-container {{
            margin: 20px 0;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: right;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <h2>Analysis Summary</h2>
        <p>This report contains comprehensive analysis results for {len(all_results)} stock(s) with interactive charts.</p>
"""
        
        # Add results for each stock
        for symbol, results in all_results.items():
            html_content += f"""
        <div class="stock-section">
            <h2>{symbol} - Analysis Results</h2>
            
            <h3>Basic Data</h3>
"""
            
            # Add basic data
            basic_data = results.get('basic_data', {})
            if basic_data:
                for key, value in basic_data.items():
                    label = key.replace('_', ' ').title()
                    html_content += f"""
            <div class="metric">
                <span class="metric-label">{label}:</span>
                <span class="metric-value">{value}</span>
            </div>
"""
            
            # Add price chart
            if symbol in charts_data and 'price_chart' in charts_data[symbol]:
                html_content += f"""
            <div class="chart-container">
                <div id="price-chart-{symbol}"></div>
            </div>
"""
            
            # Add ML analysis
            ml_analysis = results.get('ml_analysis', {})
            if ml_analysis:
                html_content += """
            <h3>Machine Learning Analysis</h3>
"""
                
                # Add performance metrics chart
                if symbol in charts_data and 'metrics_chart' in charts_data[symbol]:
                    html_content += f"""
            <div class="chart-container">
                <div id="metrics-chart-{symbol}"></div>
            </div>
"""
                
                # Add prediction chart
                if symbol in charts_data and 'prediction_chart' in charts_data[symbol]:
                    html_content += f"""
            <div class="chart-container">
                <div id="prediction-chart-{symbol}"></div>
            </div>
"""
                
                # Add feature importance chart
                if symbol in charts_data and 'feature_chart' in charts_data[symbol]:
                    html_content += f"""
            <div class="chart-container">
                <div id="feature-chart-{symbol}"></div>
            </div>
"""
                
                # Top features table
                top_features = ml_analysis.get('top_features', [])
                if top_features:
                    html_content += """
            <h4>Top Predictive Features</h4>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
"""
                    for feature in top_features:
                        html_content += f"""
                    <tr>
                        <td>{feature.get('feature', 'N/A')}</td>
                        <td>{feature.get('importance', 0):.4f}</td>
                    </tr>
"""
                    html_content += """
                </tbody>
            </table>
"""
            
            html_content += """
        </div>
"""
        
        # Add JavaScript to render Plotly charts
        html_content += """
    </div>
    
    <script>
"""
        
        # Add chart rendering scripts
        for symbol, charts in charts_data.items():
            if 'price_chart' in charts:
                html_content += f"""
        // Render price chart for {symbol}
        var priceData_{symbol} = {charts['price_chart']};
        Plotly.newPlot('price-chart-{symbol}', priceData_{symbol}.data, priceData_{symbol}.layout);
"""
            
            if 'prediction_chart' in charts:
                html_content += f"""
        // Render prediction chart for {symbol}
        var predData_{symbol} = {charts['prediction_chart']};
        Plotly.newPlot('prediction-chart-{symbol}', predData_{symbol}.data, predData_{symbol}.layout);
"""
            
            if 'feature_chart' in charts:
                html_content += f"""
        // Render feature importance chart for {symbol}
        var featureData_{symbol} = {charts['feature_chart']};
        Plotly.newPlot('feature-chart-{symbol}', featureData_{symbol}.data, featureData_{symbol}.layout);
"""
            
            if 'metrics_chart' in charts:
                html_content += f"""
        // Render metrics chart for {symbol}
        var metricsData_{symbol} = {charts['metrics_chart']};
        Plotly.newPlot('metrics-chart-{symbol}', metricsData_{symbol}.data, metricsData_{symbol}.layout);
"""
        
        html_content += """
    </script>
</body>
</html>
"""
        
        # Save file
        filename = f"analysis_report_{timestamp}.html"
        html_file = html_dir / filename
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return f"""
✅ Interactive HTML Report Created!
📁 File: {html_file}
📊 Stocks analyzed: {', '.join(all_results.keys())}
📈 Interactive Plotly charts included
🌐 Open {filename} in your browser to view the interactive report!
"""
        
    except Exception as e:
        import traceback
        return f"❌ Error creating HTML report: {str(e)}\n{traceback.format_exc()}"

# Export tools
__all__ = [
    'collect_all_results',
    'create_simple_html_report'
]