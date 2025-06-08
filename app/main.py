from fastapi import FastAPI
from pydantic import BaseModel
from app.agent.langflow import graph

app = FastAPI()

# Define the input model with thread_id
class ChatInput(BaseModel):
    message: str
    thread_id: str = "default"

# Define the /chat endpoint
@app.post("/chat")
async def chat(input: ChatInput):
    # Prepare the user's message
    user_message = {"role": "user", "content": input.message}
    # Run the graph with the user's message and thread_id
    config = {"configurable": {"thread_id": input.thread_id}}
    final_state = await graph.ainvoke({"messages": [user_message]}, config=config)
    # Extract and return the assistant's response
    assistant_response = final_state["messages"][-1].content
    return {"response": assistant_response}