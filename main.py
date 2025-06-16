# main.py - Clean version with no circular imports
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, MessagesState

# Import conversation viewer from separate module
from conversation_viewer import ConversationViewer, full_diagnostic

# Import your agents
from agents import project_owner, data_engineer, model_executer, reporter

load_dotenv()

# Main execution
if __name__ == "__main__":
    import sys
    
    # Create the graph
    print("🔧 Building agent graph...")
    graph = (
        StateGraph(MessagesState)
        .add_node(project_owner)
        .add_node(data_engineer)
        .add_node(model_executer)
        .add_node(reporter)
        .add_edge(START, "project_owner")
        .compile()
    )
    
    # Check for diagnostic mode
    if len(sys.argv) > 1 and sys.argv[1] == "--full-diagnose":
        full_diagnostic(graph)
        sys.exit()
    
    # Create viewer and run conversation
    viewer = ConversationViewer()
    
    # Example query
    query = "analyze TSLA stock - research data, perform technical analysis, and create investment report"
    
    viewer.run(graph, query)
    viewer.save_log()