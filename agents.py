# pip install -qU "langchain[anthropic]" langgraph
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool, InjectedToolCallId
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from tools import create_handoff_tool
from func import *
from handoff import *
from prompts import PROJECT_OWNER_PROMPT, DATA_ENGINEER_PROMPT, MODEL_EXECUTOR_PROMPT, MODEL_TESTER_PROMPT, QUALITY_ASSURANCE_PROMPT, REPORT_INSIGHT_GENERATOR_PROMPT
load_dotenv()

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
    tools=[transfer_to_data_engineer, 
           transfer_to_model_executer,
           transfer_to_reporter],
    prompt=PROJECT_OWNER_PROMPT,
    name="project_owner"
)

data_engineer = create_react_agent(
    model=model_gpt,
    tools=[get_stock_price, get_stock_news, compare_stocks,
           transfer_to_project_owner, 
           transfer_to_model_executer],
    prompt=DATA_ENGINEER_PROMPT,
    name="data_engineer"
)

model_executer = create_react_agent(
    model=model_gpt,
    tools=[get_technical_indicators, 
           transfer_to_project_owner, 
           transfer_to_reporter],
    prompt=MODEL_EXECUTOR_PROMPT,
    name="model_executer"
)

reporter = create_react_agent(
    model=model_gpt,
    tools=[transfer_to_project_owner],
    prompt=REPORT_INSIGHT_GENERATOR_PROMPT,
    name="reporter"
)