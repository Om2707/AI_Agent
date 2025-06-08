from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

# Define the state structure
class AgentState(TypedDict):
    messages: add_messages

# Create the StateGraph
graph_builder = StateGraph(AgentState)

# Set up the LLM with a system prompt
llm = init_chat_model("openai:gpt-4-1106-preview")
system_prompt = (
    "You are an AI assistant helping to define a project scope for innovation platforms. "
    "Ask the user the following questions in order:\n"
    "1. What stage is the project in? (Ideation, Planning, Execution)\n"
    "2. What is the primary goal of the project?\n"
    "3. Which platform are you targeting? (e.g., Topcoder, Kaggle, etc.)\n"
    "Once you have all three answers, propose a project scope in the format: "
    "'Project scope: Develop a [platform] application for [goal] in the [project_stage] stage.' "
    "Then ask, 'Does this scope sound right? (Yes/No)'. "
    "If the user says 'No', ask 'What needs to be changed?' and adapt the scope based on their feedback."
)

# Define the chatbot node
def chatbot(state: AgentState):
    # Messages include system prompt, previous messages, and the new user message
    messages = state["messages"]
    # Invoke the LLM with the current messages
    response = llm.invoke(messages)
    # Return the updated state (messages are automatically appended via add_messages)
    return {"messages": [response]}

# Add the chatbot node and edge
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# Set initial state with system prompt
initial_state = {"messages": [{"role": "system", "content": system_prompt}]}

# Compile the graph with a checkpointer and initial state
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, initial_state=initial_state)
