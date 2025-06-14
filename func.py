
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool, InjectedToolCallId
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command


# Stock analysis tools
@tool
def get_stock_data(symbol: str) -> str:
    """Get basic stock information"""
    return f"Stock {symbol}: Price $150.25, Market Cap $2.1T, P/E 25.4, Volume 45M shares"

@tool  
def analyze_trends(symbol: str) -> str:
    """Analyze stock technical trends"""
    return f"{symbol} Analysis: Bullish trend, RSI 62, 20-day MA $145, 50-day MA $140, Strong buy signal"

@tool
def generate_report(symbol: str) -> str:
    """Generate investment recommendation report"""
    return f"INVESTMENT REPORT for {symbol}: Recommendation BUY, Target Price $175, Risk Level: Medium, Confidence: High"


# pip install -qU "langchain[anthropic]" langgraph yfinance
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool, InjectedToolCallId
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
import yfinance as yf


# Simple Yahoo Finance tools
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
        
        # Calculate indicators
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
        news = stock.news[:3]  # Get 3 recent articles
        
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
        
        for symbol in symbol_list[:5]:  # Max 5 stocks
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")
            price = hist['Close'].iloc[-1] if not hist.empty else 0
            
            result += f"{symbol}: ${price:.2f} | P/E: {info.get('trailingPE', 'N/A')} | Cap: {info.get('marketCap', 'N/A')}\n"
        
        return result
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"

