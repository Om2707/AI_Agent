from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
import json
from typing import Optional, Dict, Any
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from ..rag.retriever import retriever_tool
from .logging import log_reasoning_trace, log_user_feedback
from .feedback import capture_feedback, adapt_prompt

# Define FieldData structure (as a regular dictionary)
def FieldData(value: Any, reasoning: str, confidence: Optional[float] = None) -> Dict[str, Any]:
    return {"value": value, "reasoning": reasoning, "confidence": confidence}

# Define AgentState structure (as a plain dictionary instead of TypedDict)
def AgentState(messages: add_messages, field_data: Dict[str, Dict[str, Any]], is_final: bool) -> Dict[str, Any]:
    return {
        "messages": messages,
        "field_data": field_data,
        "is_final": is_final
    }

# Create StateGraph
graph_builder = StateGraph(AgentState)

# Set up LLM with tool-calling
llm = init_chat_model(
    "openai:gpt-4-1106-preview", 
    model_kwargs={"tools": [retriever_tool]}
)

# Define chatbot node
async def chatbot(state: Dict[str, Any]):
    # Check for feedback
    if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" and state["messages"][-2]["content"] == "What needs to be changed?":
        feedback = capture_feedback(state["messages"][-1]["content"])
        log_user_feedback(feedback, {field: data["value"] for field, data in state["field_data"].items()})
        adapted_prompt = adapt_prompt(feedback, state["field_data"])
        return {"messages": [{"role": "assistant", "content": adapted_prompt}]}

    # Call LLM (using the state directly)
    result = await llm.ainvoke(state["messages"])

    # Handle proposed scope
    if result.content.strip().startswith("{"):
        try:
            proposed_fields = json.loads(result.content.strip())
            if "fields" in proposed_fields:
                field_data = {}
                for field, data in proposed_fields["fields"].items():
                    field_data[field] = FieldData(data["value"], data["reasoning"], data.get("confidence"))
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
            retrieved_docs = await retriever_tool.run(tool_call["args"]["query"])  # Async call
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
                    state["field_data"]["timeline"] = FieldData(
                        {"submission_days": int(metadata.get("timeline", "3 months").split()[0]) * 30},
                        "Based on RAG recommendations from similar projects",
                        0.8
                    )
                if "tech_stack" not in state["field_data"]:
                    state["field_data"]["tech_stack"] = FieldData(
                        ["Figma"] if "mockup" in state["field_data"].get("goal", {}).get("value", "").lower() else ["Python"],
                        "Inferred from project goal mentioning mockup or coding",
                        0.92
                    )
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

# Compile graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
