# AI Agent for Project Scoping

This AI agent helps users define and refine project scopes for innovation platforms through an intelligent dialogue system.

## Features

### 1. Agent Orchestration Setup
- Built using LangGraph for state management and flow control
- Implements a state machine pattern with `StateGraph` for managing conversation flow
- Uses GPT-4 for natural language understanding and generation
- Integrates with FastAPI for HTTP endpoints

### 2. Memory Handling
- Conversation state maintained through `AgentState` TypedDict
- Persistent storage of reasoning traces in JSON format
- Feedback logging for continuous improvement
- Vector store (Qdrant) for semantic search of past projects

### 3. Prompt Strategies
- Dynamic prompt adaptation based on user feedback
- Schema-aware system prompts for different platforms (Topcoder, Kaggle)
- Confidence scoring for inferred fields
- Multi-turn dialogue for scope refinement

### 4. Schema Configurability
- Platform-specific field requirements
- Extensible field definitions with confidence scoring
- Support for optional and required fields
- Custom validation rules per platform

## Testing Workflow

1. **Setup Environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your OpenAI API key and other configurations
   ```

2. **Start Services**
   ```bash
   # Start Qdrant vector store
   docker run -p 6333:6333 qdrant/qdrant
   
   # Start the FastAPI server
   uvicorn app.main:app --reload
   ```

3. **Test Scenarios**

   a. Basic Project Scoping:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "I want to create a task management app on Topcoder", "thread_id": "test1"}'
   ```

   b. Scope Refinement:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "No, I want to focus on team collaboration features", "thread_id": "test1"}'
   ```

4. **Monitor Results**
   - Check reasoning traces in `app/agent/reasoning_traces.json`
   - Review feedback logs in `app/agent/feedback_logs.json`
   - Monitor vector store for similar project recommendations

## Architecture Components

1. **Scoping Dialogue Logic**
   - Implemented in `app/agent/langflow.py`
   - Handles multi-turn conversations
   - Manages state transitions
   - Processes user feedback

2. **Schema-aware Q&A Logic**
   - Platform-specific field validation
   - Dynamic field requirements
   - Confidence scoring for inferred fields
   - Structured output generation

3. **RAG Integration**
   - Vector store using Qdrant
   - Semantic search for similar projects
   - Past project recommendations
   - Embedding-based retrieval

4. **Reasoning Trace Generation**
   - Detailed logging of decision points
   - Confidence scoring
   - Field-level reasoning
   - Feedback incorporation

## Configuration

The system can be configured through environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4)
- `QDRANT_URL`: Vector store URL
- `EMBEDDING_MODEL`: Model for embeddings
- `MAX_ITERATIONS`: Maximum conversation turns
- `TEMPERATURE`: LLM temperature setting
