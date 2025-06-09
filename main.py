from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.langflow import graph

app = FastAPI()

class ChatInput(BaseModel):
    message: str
    thread_id: str = "default"

@app.post("/chat")
async def chat(input: ChatInput):
    user_message = {"role": "user", "content": input.message}
    config = {"configurable": {"thread_id": input.thread_id}}
    final_state = await graph.ainvoke({"messages": [user_message]}, config=config)
    
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
