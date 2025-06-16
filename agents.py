# agents.py - OPTIMIZED VERSION with integrated handoff tools
# Eliminates need for separate handoff.py file

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import all tools from func.py
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
    create_model_visualization, model_summary_report
)

# Import handoff utility
from tools import create_handoff_tool

# Import prompts
from prompts import (
    PROJECT_OWNER_PROMPT, DATA_ENGINEER_PROMPT, 
    MODEL_EXECUTOR_PROMPT, REPORT_INSIGHT_GENERATOR_PROMPT
)

load_dotenv()

# Initialize models
model_gpt = ChatOpenAI(
    model="gpt-4o-2024-08-06",
    api_key=os.environ.get("gpt_api_key", "")
)

model_claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.environ.get("claude_api_key", "")
)

# Create handoff tools (integrated from handoff.py)
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

# Define agents with optimized tool sets
project_owner = create_react_agent(
    model=model_gpt,
    tools=[
        transfer_to_data_engineer, 
        transfer_to_model_executer,
        transfer_to_reporter
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
        
        # Model persistence and management (ENHANCED)
        save_trained_model,
        load_trained_model,
        smart_predict_stock_price,
        delete_saved_model,           # ✨ NEW
        model_performance_summary,    # ✨ NEW
        
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
        
        # Model inventory and performance analysis (ENHANCED)
        list_saved_models,
        model_performance_summary,    # ✨ NEW
        
        # Handoff tools
        transfer_to_project_owner
    ],
    prompt=REPORT_INSIGHT_GENERATOR_PROMPT,
    name="reporter"
)

# Export all agents for easy importing
__all__ = [
    'project_owner', 
    'data_engineer', 
    'model_executer', 
    'reporter',
    'model_gpt',
    'model_claude'
]