from typing import Dict

def capture_feedback(user_input: str) -> str:
    """Capture user feedback when a suggestion is rejected."""
    return user_input.strip()

def adapt_prompt(feedback: str, current_scope: Dict[str, str]) -> str:
    """Adapt the prompt based on user feedback."""
    if "title" in feedback.lower():
        return "What should be the correct title for the project?"
    elif "overview" in feedback.lower():
        return "Can you provide a better overview for the project?"
    else:
        return "How can I improve the project scope based on your feedback?"