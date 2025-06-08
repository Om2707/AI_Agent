from pydantic import BaseModel

class UserInput(BaseModel):
    message: str

class AgentResponse(BaseModel):
    response: str
