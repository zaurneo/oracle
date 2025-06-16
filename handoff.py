# handoff.py - Clean handoff tool instances
from tools import create_handoff_tool

# Handoff tools
transfer_to_project_owner = create_handoff_tool(
    agent_name="project_owner",
    description="Transfer to the project owner agent.",
)

transfer_to_data_engineer = create_handoff_tool(
    agent_name="data_engineer",
    description="Transfer to the data engineer agent.",
)

transfer_to_model_executer = create_handoff_tool(
    agent_name="model_executer",
    description="Transfer to the model executer agent.",
)

transfer_to_reporter = create_handoff_tool(
    agent_name="reporter", 
    description="Transfer to the report generation agent.",
)