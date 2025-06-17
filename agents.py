# agents.py - Updated with Simplified HTML Generator

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import all tools from analysis package
from analysis import (
    # Basic stock analysis tools
    get_stock_price, get_stock_news, compare_stocks, get_technical_indicators,
    
    # ML forecasting tools
    fetch_historical_data, create_technical_features, train_decision_tree_model,
    predict_stock_price, backtest_model, quick_model_test,
    
    # Model persistence tools
    save_trained_model, load_trained_model, list_saved_models,
    smart_predict_stock_price, delete_saved_model, model_performance_summary,
    
    # Visualization and reporting tools
    create_model_visualization, model_summary_report,
    
    # Simplified HTML generator tools
    collect_all_results, create_simple_html_report
)

# Import handoff utility
from tools import create_handoff_tool

# Import prompts
from prompts import (
    PROJECT_OWNER_PROMPT, DATA_ENGINEER_PROMPT, 
    MODEL_EXECUTOR_PROMPT, REPORT_INSIGHT_GENERATOR_PROMPT
)

load_dotenv()

# Initialize models - Check API keys
gpt_api_key = os.environ.get("gpt_api_key", "")
claude_api_key = os.environ.get("claude_api_key", "")

if not gpt_api_key:
    print("❌ WARNING: GPT API key not found! Please set 'gpt_api_key' in your .env file")

# Use GPT for all agents
model_gpt = ChatOpenAI(
    model="gpt-4o-2024-08-06",
    api_key=gpt_api_key
)

# Create handoff tools
transfer_to_project_owner = create_handoff_tool(
    agent_name="project_owner",
    description="Transfer to the project owner agent."
)

transfer_to_data_engineer = create_handoff_tool(
    agent_name="data_engineer",
    description="Transfer to the data engineer agent."
)

transfer_to_model_executer = create_handoff_tool(
    agent_name="model_executer",
    description="Transfer to the model executer agent."
)

transfer_to_reporter = create_handoff_tool(
    agent_name="reporter", 
    description="Transfer to the report generation agent."
)

transfer_to_html_generator = create_handoff_tool(
    agent_name="html_generator",
    description="Transfer to the HTML report generator agent."
)

# Define agents
project_owner = create_react_agent(
    model=model_gpt,
    tools=[
        transfer_to_data_engineer, 
        transfer_to_model_executer,
        transfer_to_reporter,
        transfer_to_html_generator
    ],
    prompt=PROJECT_OWNER_PROMPT,
    name="project_owner"
)

data_engineer = create_react_agent(
    model=model_gpt,
    tools=[
        # Basic data collection
        get_stock_price, 
        get_stock_news, 
        compare_stocks,
        
        # Advanced ML data preparation
        fetch_historical_data,
        create_technical_features,
        
        # Handoff tools
        transfer_to_project_owner, 
        transfer_to_model_executer
    ],
    prompt=DATA_ENGINEER_PROMPT,
    name="data_engineer"
)

model_executer = create_react_agent(
    model=model_gpt,
    tools=[
        # Traditional analysis
        get_technical_indicators,
        
        # ML model tools
        train_decision_tree_model,
        predict_stock_price,
        backtest_model,
        quick_model_test,
        
        # Model persistence and management
        save_trained_model,
        load_trained_model,
        smart_predict_stock_price,
        delete_saved_model,
        model_performance_summary,
        
        # Handoff tools
        transfer_to_project_owner, 
        transfer_to_reporter
    ],
    prompt=MODEL_EXECUTOR_PROMPT,
    name="model_executer"
)

reporter = create_react_agent(
    model=model_gpt,
    tools=[
        # Visualization and reporting
        create_model_visualization,
        model_summary_report,
        
        # Model inventory and performance analysis
        list_saved_models,
        model_performance_summary,
        
        # Handoff tools
        transfer_to_project_owner,
        transfer_to_html_generator
    ],
    prompt=REPORT_INSIGHT_GENERATOR_PROMPT,
    name="reporter"
)

# Simplified HTML Generator Agent
HTML_GENERATOR_PROMPT = """
You are the HTML Report Generator responsible for creating simple, consolidated HTML reports.

Your task is straightforward:
1. Use collect_all_results to gather all analysis data for the requested stock(s)
2. Use create_simple_html_report to generate a clean HTML file

WORKFLOW:
- When you receive a request, first identify which stocks were analyzed
- Call collect_all_results with the stock symbols (e.g., "AAPL" or "AAPL,GOOGL,TSLA") 
- Then call create_simple_html_report to generate the HTML file
- Report back with the file location

Keep it simple - just collect and report. The HTML file will consolidate all results automatically.

Available tools:
- collect_all_results: Gathers all analysis data
- create_simple_html_report: Creates the HTML file

Focus on efficiency - two tool calls and you're done!
"""

html_generator = create_react_agent(
    model=model_gpt,
    tools=[
        # Simple HTML generation tools only
        collect_all_results,
        create_simple_html_report,
        
        # Handoff tool back to project owner
        transfer_to_project_owner
    ],
    prompt=HTML_GENERATOR_PROMPT,
    name="html_generator"
)

print("✅ All agents initialized successfully using GPT-4!")
print("🤖 Agent Configuration:")
print("  - project_owner: GPT-4")
print("  - data_engineer: GPT-4") 
print("  - model_executer: GPT-4")
print("  - reporter: GPT-4")
print("  - html_generator: GPT-4 (simplified)")

# Export all agents
__all__ = [
    'project_owner', 
    'data_engineer', 
    'model_executer', 
    'reporter',
    'html_generator',
    'model_gpt'
]