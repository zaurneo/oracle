# analysis/basic_tools.py - Basic Stock Analysis Tools
"""
Basic stock analysis tools for data collection and fundamental analysis.

This module provides:
- Current stock price and company information
- Recent news and market sentiment
- Multi-stock comparison capabilities  
- Traditional technical analysis indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic company information
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        Formatted string with current price and key metrics
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else "N/A"

        # Store current price in state manager for HTML report
        if current_price != "N/A":
            from .shared_state import state_manager
            state_manager.set_model_data(symbol, 'current_price', current_price)
        
        return f"""
Stock Information for {symbol}:
Current Price: ${current_price:.2f}
Market Cap: {info.get('marketCap', 'N/A')}
P/E Ratio: {info.get('trailingPE', 'N/A')}
Volume: {info.get('volume', 'N/A'):,}
52W High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52W Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
        """
    except Exception as e:
        return f"Error getting stock data for {symbol}: {str(e)}"

@tool
def get_stock_news(symbol: str, max_articles: int = 5) -> str:
    """Get recent news articles for a stock
    
    Args:
        symbol: Stock ticker symbol
        max_articles: Maximum number of articles to return (default: 5)
        
    Returns:
        Formatted string with recent news headlines
    """
    try:
        stock = yf.Ticker(symbol)
        news = stock.news[:max_articles]
        
        if not news:
            return f"No recent news found for {symbol}"
        
        result = f"Recent news for {symbol}:\n" + "=" * 40 + "\n"
        
        for i, item in enumerate(news, 1):
            title = item.get('title', 'No title available')
            publisher = item.get('publisher', 'Unknown publisher')
            published = item.get('providerPublishTime', '')
            
            # Convert timestamp to readable date
            if published:
                try:
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(published).strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = 'Unknown date'
            else:
                date_str = 'Unknown date'
            
            result += f"{i}. {title}\n"
            result += f"   Source: {publisher} | {date_str}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error getting news for {symbol}: {str(e)}"

@tool
def compare_stocks(symbols: str, metric: str = "overview") -> str:
    """Compare multiple stocks across key metrics
    
    Args:
        symbols: Comma-separated list of stock symbols (e.g., 'AAPL,GOOGL,MSFT')
        metric: Comparison type - 'overview', 'valuation', 'performance'
        
    Returns:
        Formatted comparison table
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if len(symbol_list) > 10:
            return "Too many symbols. Please limit to 10 stocks for comparison."
        
        result = f"Stock Comparison - {metric.title()}\n"
        result += "=" * 60 + "\n"
        
        comparison_data = []
        
        for symbol in symbol_list:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                
                # Get 1-year performance
                hist_1y = stock.history(period="1y")
                if len(hist_1y) > 1:
                    year_return = ((hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[0]) - 1) * 100
                else:
                    year_return = 0
                
                stock_data = {
                    'Symbol': symbol,
                    'Price': f"${current_price:.2f}",
                    'Market Cap': format_market_cap(info.get('marketCap')),
                    'P/E Ratio': format_pe_ratio(info.get('trailingPE')),
                    '1Y Return': f"{year_return:+.1f}%",
                    'Volume': f"{info.get('volume', 0):,}",
                    'Sector': info.get('sector', 'N/A')[:15] + '...' if len(info.get('sector', '')) > 15 else info.get('sector', 'N/A')
                }
                
                comparison_data.append(stock_data)
                
            except Exception as e:
                comparison_data.append({
                    'Symbol': symbol,
                    'Price': 'Error',
                    'Market Cap': 'N/A',
                    'P/E Ratio': 'N/A',
                    '1Y Return': 'N/A',
                    'Volume': 'N/A',
                    'Sector': 'N/A'
                })
        
        # Format as table
        if metric == "overview":
            headers = ['Symbol', 'Price', 'Market Cap', 'P/E Ratio', '1Y Return', 'Sector']
        elif metric == "valuation":
            headers = ['Symbol', 'Price', 'P/E Ratio', 'Market Cap', 'Volume']
        else:  # performance
            headers = ['Symbol', 'Price', '1Y Return', 'Volume', 'Sector']
        
        # Create formatted table
        for header in headers:
            result += f"{header:<12} "
        result += "\n" + "-" * 60 + "\n"
        
        for stock_data in comparison_data:
            for header in headers:
                value = stock_data.get(header, 'N/A')
                result += f"{str(value):<12} "
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"

# Replace the get_technical_indicators function in analysis/basic_tools.py

@tool
def get_technical_indicators(symbol: str, period: str = "3mo") -> str:
    """Calculate and return traditional technical analysis indicators
    
    Args:
        symbol: Stock ticker symbol
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y')
        
    Returns:
        Formatted string with technical analysis results
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if len(hist) < 50:
            return f"Insufficient data for technical analysis of {symbol}. Need at least 50 days."
        
        # Current price
        current_price = hist['Close'].iloc[-1]
        
        # Moving averages
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
        
        # FIXED: Ensure we're working with Series for ewm calculations
        close_series = hist['Close']
        ema_12 = close_series.ewm(span=12).mean().iloc[-1]
        ema_26 = close_series.ewm(span=26).mean().iloc[-1]
        
        # RSI calculation
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD - FIXED: Use Series consistently
        macd_line = ema_12 - ema_26
        macd_signal_series = close_series.ewm(span=12).mean() - close_series.ewm(span=26).mean()
        macd_signal = macd_signal_series.ewm(span=9).mean().iloc[-1]
        macd_histogram = macd_line - macd_signal
        
        # Bollinger Bands
        bb_middle = close_series.rolling(20).mean().iloc[-1]
        bb_std = close_series.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # Volume analysis
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = hist['Volume'].iloc[-1] / avg_volume
        
        # Price momentum
        momentum_5d = (current_price / hist['Close'].iloc[-6] - 1) * 100 if len(hist) >= 6 else 0
        momentum_20d = (current_price / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else 0
        
        # Generate signals
        signals = []
        if current_price > sma_20:
            signals.append("Above 20-day MA (Bullish)")
        else:
            signals.append("Below 20-day MA (Bearish)")
            
        if rsi > 70:
            signals.append("RSI Overbought (>70)")
        elif rsi < 30:
            signals.append("RSI Oversold (<30)")
        else:
            signals.append(f"RSI Neutral ({rsi:.1f})")
            
        if macd_histogram > 0:
            signals.append("MACD Bullish")
        else:
            signals.append("MACD Bearish")
        
        result = f"""
Technical Analysis for {symbol} ({period}):
{'=' * 50}

Price Information:
- Current Price: ${current_price:.2f}
- 20-day SMA: ${sma_20:.2f}
- 50-day SMA: {f'${sma_50:.2f}' if sma_50 is not None else 'N/A (insufficient data)'}

Momentum Indicators:
- RSI (14): {rsi:.1f}
- 5-day momentum: {momentum_5d:+.2f}%
- 20-day momentum: {momentum_20d:+.2f}%

MACD:
- MACD Line: {macd_line:.2f}
- Signal Line: {macd_signal:.2f}
- Histogram: {macd_histogram:.2f}

Bollinger Bands:
- Upper: ${bb_upper:.2f}
- Middle: ${bb_middle:.2f}
- Lower: ${bb_lower:.2f}
- Position: {bb_position:.2%}

Volume Analysis:
- Current Volume: {hist['Volume'].iloc[-1]:,.0f}
- 20-day Avg Volume: {avg_volume:,.0f}
- Volume Ratio: {volume_ratio:.2f}x

Technical Signals:
{chr(10).join(f'• {signal}' for signal in signals)}

Overall Trend: {'BULLISH' if current_price > sma_20 and rsi < 70 and macd_histogram > 0 else 'BEARISH' if current_price < sma_20 and macd_histogram < 0 else 'NEUTRAL'}
"""
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error analyzing {symbol}: {str(e)}\n\nDetailed error:\n{error_details}"

# Helper functions
def format_market_cap(market_cap):
    """Format market cap for display"""
    if not market_cap or market_cap == 'N/A':
        return 'N/A'
    
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.1f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.1f}M"
    else:
        return f"${market_cap:,.0f}"

def format_pe_ratio(pe_ratio):
    """Format P/E ratio for display"""
    if not pe_ratio or pe_ratio == 'N/A' or pe_ratio <= 0:
        return 'N/A'
    return f"{pe_ratio:.1f}"

# Export all tools
__all__ = [
    'get_stock_price',
    'get_stock_news', 
    'compare_stocks',
    'get_technical_indicators'
]