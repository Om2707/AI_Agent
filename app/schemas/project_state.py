from typing import TypedDict, List, Dict, Optional

class ProjectState(TypedDict, total=False):
    user_input: str
    needs_clarification: bool
    clarification: str
    questions_asked: List[str]
    answers: Dict[str, str]
    stage: int
    final: bool
    response: str
