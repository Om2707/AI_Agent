import json
from pathlib import Path
from datetime import datetime
from typing import Dict

LOG_FILE = Path(__file__).parent / "reasoning_traces.json"
FEEDBACK_LOG_FILE = Path(__file__).parent / "feedback_logs.json"

def log_reasoning_trace(reasoning: str, scope: Dict[str, str], accepted: bool):
    """Log reasoning trace for a suggestion."""
    log_entry = {
        "timestamp": str(datetime.now()),
        "reasoning": reasoning,
        "scope": scope,
        "accepted": accepted
    }
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

def log_user_feedback(feedback: str, original_scope: Dict[str, str]):
    """Log user feedback when a suggestion is rejected."""
    log_entry = {
        "timestamp": str(datetime.now()),
        "feedback": feedback,
        "original_scope": original_scope
    }
    if FEEDBACK_LOG_FILE.exists():
        with open(FEEDBACK_LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(FEEDBACK_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)
