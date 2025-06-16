# agents.py - Clean version with no circular imports
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import tools
from func import (
    get_stock_price, get_stock_news, compare_stocks, get_technical_indicators,
    fetch_historical_data, create_technical_features, train_decision_tree_model,
    predict_stock_price, backtest_model, create_model_visualization, model_summary_report
)

# Import handoff tools
from handoff import (
    transfer_to_data_engineer, transfer_to_model_executer, transfer_to_reporter, transfer_to_project_owner
)

# Import prompts
from prompts import (
    PROJECT_OWNER_PROMPT, DATA_ENGINEER_PROMPT, MODEL_EXECUTOR_PROMPT, REPORT_INSIGHT_GENERATOR_PROMPT
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

# Define agents
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
        # Original data tools
        get_stock_price, 
        get_stock_news, 
        compare_stocks,
        # New forecasting data tools
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
        # Original analysis tools
        get_technical_indicators,
        # New ML model tools
        train_decision_tree_model,
        predict_stock_price,
        backtest_model,
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
        # Visualization and reporting tools
        create_model_visualization,
        model_summary_report,
        # Handoff tools
        transfer_to_project_owner
    ],
    prompt=REPORT_INSIGHT_GENERATOR_PROMPT,
    name="reporter"
)