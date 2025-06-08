from app.agent.langflow import workflow  # compiled in langflow.py
graph = workflow

def dummy_agent_response(user_message: str) -> str:
    result = graph.invoke({ "user_input": user_message })
    return result["response"]
