from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json
from ..rag.retriever import retriever_tool
from .logging import log_reasoning_trace, log_user_feedback
from .feedback import capture_feedback, adapt_prompt

# Define FieldData structure
class FieldData(TypedDict):
    value: Any
    reasoning: str
    confidence: Optional[float]

# Define AgentState structure
class AgentState(TypedDict):
    messages: add_messages
    field_data: Dict[str, FieldData]
    is_final: bool

# Define ProjectScope schema for validation
class ProjectScope(BaseModel):
    stage: str = Field(description="The stage of the project (Ideation, Planning, Execution)")
    goal: str = Field(description="The primary goal of the project")
    platform: str = Field(description="The target platform (e.g., Topcoder, Kaggle)")
    title: Optional[str] = Field(None, description="The title of the project")
    overview: Optional[str] = Field(None, description="A brief overview of the project")
    description: Optional[str] = Field(None, description="A detailed description")
    tags: Optional[str] = Field(None, description="Tags for the project")
    tech_stack: Optional[list] = Field(None, description="Technologies used")
    timeline: Optional[dict] = Field(None, description="Project timeline")

# Create StateGraph
graph_builder = StateGraph(AgentState)

# Set up LLM with tool-calling
llm = init_chat_model("openai:gpt-4-1106-preview", tools=[retriever_tool])

# Define system prompt
system_prompt = (
    "You are an AI assistant helping users define project scopes for innovation platforms. Guide them to gather:\n"
    "- Project stage (Ideation, Planning, Execution)\n"
    "- Primary goal\n"
    "- Target platform (e.g., Topcoder, Kaggle)\n"
    "- Platform-specific fields:\n"
    "  - Topcoder: title (required), overview (required), tech_stack (optional), timeline (optional)\n"
    "  - Kaggle: title (required), description (required), tags (required), timeline (optional)\n"
    "Extract information from user inputs (e.g., 'task management app on Topcoder' -> goal='task management app', platform='Topcoder'). Ask for missing information, starting with stage.\n"
    "Once all required fields are collected, propose the scope in this JSON format:\n"
    "{\n"
    "  \"fields\": {\n"
    "    \"stage\": {\"value\": \"[stage]\", \"reasoning\": \"[reasoning]\", \"confidence\": [optional confidence]},\n"
    "    \"goal\": {\"value\": \"[goal]\", \"reasoning\": \"[reasoning]\", \"confidence\": [optional confidence]},\n"
    "    \"platform\": {\"value\": \"[platform]\", \"reasoning\": \"[reasoning]\", \"confidence\": [optional confidence]},\n"
    "    \"title\": {\"value\": \"[title]\", \"reasoning\": \"[reasoning]\", \"confidence\": [optional confidence]},\n"
    "    // Include other fields as needed\n"
    "  }\n"
    "}\n"
    "Provide reasoning for each field (e.g., 'Provided by the user', 'Inferred from user input', 'Based on RAG recommendations'). Include confidence (0.0-1.0) for non-user-provided fields.\n"
    "After proposing, ask 'Does this scope sound right? (Yes/No)'.\n"
    "If 'Yes', use 'retrieve_similar_projects' with a query describing the project.\n"
    "If 'No', ask 'What needs to be changed?' and adjust based on feedback.\n"
    "If the platform is unknown, ask for clarification.\n"
)

# Define chatbot node
def chatbot(state: AgentState):
    # Check for feedback
    if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" and state["messages"][-2]["content"] == "What needs to be changed?":
        feedback = capture_feedback(state["messages"][-1]["content"])
        log_user_feedback(feedback, {field: data["value"] for field, data in state["field_data"].items()})
        adapted_prompt = adapt_prompt(feedback, state["field_data"])
        return {"messages": [{"role": "assistant", "content": adapted_prompt}]}

    # Call LLM
    result = llm.invoke(state["messages"])

    # Handle proposed scope
    if result.content.strip().startswith("{"):
        try:
            proposed_fields = json.loads(result.content.strip())
            if "fields" in proposed_fields:
                field_data = {}
                for field, data in proposed_fields["fields"].items():
                    field_data[field] = {
                        "value": data["value"],
                        "reasoning": data["reasoning"],
                        "confidence": data.get("confidence")
                    }
                # Log reasoning trace
                log_reasoning_trace(
                    "Proposed scope based on collected fields",
                    {field: data["value"] for field, data in field_data.items()},
                    accepted=False
                )
                # Ask for confirmation
                confirmation_question = {
                    "role": "assistant",
                    "content": f"Does this scope sound right? (Yes/No)"
                }
                return {"messages": [result, confirmation_question], "field_data": field_data}
        except json.JSONDecodeError:
            pass

    # Handle tool calls (RAG)
    if hasattr(result, "tool_calls") and result.tool_calls:
        tool_call = result.tool_calls[0]
        if tool_call["name"] == "retrieve_similar_projects":
            retrieved_docs = retriever_tool.run(tool_call["args"]["query"])
            recommendations = []
            for doc in retrieved_docs:
                metadata = doc.metadata
                recommendations.append({
                    "timeline": metadata.get("timeline", "Not specified"),
                    "judging_criteria": metadata.get("judging_criteria", ["Not specified"]),
                    "formatting": metadata.get("formatting", "Not specified")
                })
                # Update field_data with RAG recommendations
                if "timeline" not in state["field_data"]:
                    state["field_data"]["timeline"] = {
                        "value": {"submission_days": int(metadata.get("timeline", "3 months").split()[0]) * 30},
                        "reasoning": "Based on RAG recommendations from similar projects",
                        "confidence": 0.8
                    }
                if "tech_stack" not in state["field_data"]:
                    state["field_data"]["tech_stack"] = {
                        "value": ["Figma"] if "mockup" in state["field_data"].get("goal", {}).get("value", "").lower() else ["Python"],
                        "reasoning": "Inferred from project goal mentioning mockup or coding",
                        "confidence": 0.92
                    }
            recommendation_text = "Based on similar projects, here are some recommendations:\n"
            for i, rec in enumerate(recommendations):
                recommendation_text += f"- Project {i+1}:\n  - Timeline: {rec['timeline']}\n  - Judging Criteria: {', '.join(rec['judging_criteria'])}\n  - Formatting: {rec['formatting']}\n"
            # Log final reasoning trace
            log_reasoning_trace(
                "Scope accepted with RAG recommendations",
                {field: data["value"] for field, data in state["field_data"].items()},
                accepted=True
            )
            return {
                "messages": [{"role": "assistant", "content": f"Great! The scope is finalized.\n{recommendation_text}"}],
                "is_final": True
            }

    # Handle confirmation or rejection
    if state.get("field_data") and result.content.lower() in ["yes", "no"]:
        if result.content.lower() == "yes":
            query = f"Project on {state['field_data']['platform']['value']} for {state['field_data']['goal']['value']} in {state['field_data']['stage']['value']} stage with title '{state['field_data']['title']['value']}'"
            return {"messages": [{"role": "assistant", "content": f"Calling retrieve_similar_projects with query: {query}"}]}
        else:
            return {"messages": [{"role": "assistant", "content": "What needs to be changed?"}]}

    # Regular conversational response
    return {"messages": [result]}

# Add node and edge
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# Set initial state
initial_state = {
    "messages": [{"role": "system", "content": system_prompt}],
    "field_data": {},
    "is_final": False
}

# Compile graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, initial_state=initial_state)
