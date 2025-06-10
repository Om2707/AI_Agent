from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.langflow import graph

app = FastAPI()

# Define ChatInput model using Pydantic (for JSON body input)
class ChatInput(BaseModel):
    message: str
    thread_id: str = "default"

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

@app.post("/chat")
async def chat(input: ChatInput):
    user_message = {"role": "user", "content": input.message}
    config = {"configurable": {"thread_id": input.thread_id}}

    # Initial state (no Pydantic models here, just plain dict)
    initial_state = {
        "messages": [{"role": "system", "content": system_prompt}],# Add system prompt
        "field_data": {},
        "is_final": False
    }
    initial_state["messages"].append(user_message)

    # Invoke the graph with the initial state
    final_state = await graph.ainvoke(initial_state, config=config)

    # Prepare response and handle final state
    if final_state.get("is_final"):
        final_spec = {field: data["value"] for field, data in final_state["field_data"].items()}
        reasoning_trace = [
            {
                "field": field,
                "source": data["reasoning"],
                "confidence": data.get("confidence")
            }
            for field, data in final_state["field_data"].items()
        ]
        final_output = {**final_spec, "reasoning_trace": reasoning_trace}
        return {"response": final_state["messages"][-1]["content"], "final_spec": final_output}

    return {"response": final_state["messages"][-1]["content"]}
