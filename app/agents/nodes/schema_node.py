from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.models.types import ConversationState, Platform, ChallengeType, ReasoningTrace
from app.services.schema_service import schema_service
from app.services.rag_service import rag_service
from app.config.settings import settings
import json

class SchemaNode:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main schema node logic"""
        conversation_state = ConversationState(**state.get("conversation_state", {}))
        
        if not conversation_state.schema:
            return self._load_schema(conversation_state, state)
        else:
            return self._handle_field_questions(conversation_state, state)
    
    def _load_schema(self, state: ConversationState, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Load the appropriate schema based on platform and challenge type"""
        if not state.platform or not state.challenge_type:
            return {
                "conversation_state": state.dict(),
                "response": "I need to know the platform and challenge type first. Let me help you select those.",
                "next_node": "scoping"
            }
        
        # Get schema
        platform = Platform(state.platform)
        challenge_type = ChallengeType(state.challenge_type)
        schema = schema_service.get_schema(platform, challenge_type)
        
        if not schema:
            return {
                "conversation_state": state.dict(),
                "response": f"I don't have a schema for {platform.value} {challenge_type.value} challenges. Let me help you with a different combination.",
                "next_node": "scoping"
            }
        
        # Update state with schema
        updated_state = state.dict()
        updated_state["schema"] = schema.dict()
        updated_state["required_fields"] = schema_service.get_required_fields(schema)
        
        # Get first field to work on
        next_field = self._get_next_field(updated_state)
        if next_field:
            updated_state["current_field"] = next_field
        
        # Create introductory message
        response = self._create_schema_intro_message(schema, state)
        
        return {
            "conversation_state": updated_state,
            "response": response,
            "next_node": "field_questions"
        }
    
    def _handle_field_questions(self, state: ConversationState, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle questions for specific fields"""
        user_input = full_state.get("user_input", "")
        current_field = state.current_field
        
        if not current_field:
            # All fields completed
            return self._generate_final_spec(state, full_state)
        
        # Get field definition
        field_def = schema_service.get_field_definition(state.schema, current_field)
        if not field_def:
            return self._move_to_next_field(state, full_state)
        
        # Check if user provided an answer
        if user_input and not self._is_question_response(user_input):
            # Process the user's answer
            return self._process_field_answer(state, full_state, current_field, user_input)
        else:
            # Ask question about current field
            return self._ask_field_question(state, full_state, current_field, field_def)
    
    def _ask_field_question(self, state: ConversationState, full_state: Dict[str, Any], 
                           field_name: str, field_def: Any) -> Dict[str, Any]:
        """Ask a question about a specific field"""
        
        # Get RAG context for this field
        context_query = f"{state.user_responses.get('confirmed_scope', state.user_input)} {field_name}"
        similar_challenges = rag_service.search_similar_challenges(context_query, limit=3)
        
        # Build context for LLM
        system_prompt = self._build_field_question_prompt(field_name, field_def, similar_challenges, state)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please ask a question about the '{field_name}' field for this challenge.")
        ]
        
        response = self.llm.invoke(messages)
        ai_response = response.content
        
        return {
            "conversation_state": state.dict(),
            "response": ai_response,
            "next_node": "field_questions"
        }
    
    def _process_field_answer(self, state: ConversationState, full_state: Dict[str, Any], 
                             field_name: str, user_answer: str) -> Dict[str, Any]:
        """Process user's answer for a field"""
        
        # Validate and process the answer
        processed_answer = self._process_answer_with_llm(state, field_name, user_answer)
        
        # Update state
        updated_state = state.dict()
        updated_state["user_responses"][field_name] = processed_answer["value"]
        updated_state["completed_fields"].append(field_name)
        
        # Add reasoning trace
        reasoning_trace = ReasoningTrace(
            field=field_name,
            source=f"user_input: {user_answer}",
            confidence=processed_answer.get("confidence", 0.8),
            reasoning=processed_answer.get("reasoning", f"User provided: {user_answer}")
        )
        updated_state["reasoning_traces"].append(reasoning_trace.dict())
        
        # Get next field
        next_field = self._get_next_field(updated_state)
        updated_state["current_field"] = next_field
        
        # Create response
        if next_field:
            response = f"Great! I've recorded your answer for {field_name}. Let me ask about the next field."
        else:
            response = "Perfect! I have all the information I need. Let me generate your challenge specification."
        
        return {
            "conversation_state": updated_state,
            "response": response,
            "next_node": "field_questions" if next_field else "spec_generation"
        }
    
    def _process_answer_with_llm(self, state: ConversationState, field_name: str, user_answer: str) -> Dict[str, Any]:
        """Use LLM to process and validate user's answer"""
        field_def = schema_service.get_field_definition(state.schema, field_name)
        
        system_prompt = f"""
        You are processing a user's answer for the '{field_name}' field in a challenge specification.
        
        Field Definition:
        - Type: {field_def.field_type}
        - Required: {field_def.required}
        - Description: {field_def.description or 'No description'}
        
        User's Answer: "{user_answer}"
        
        Your task:
        1. Extract the relevant information
        2. Format it according to the field type
        3. Provide reasoning for your processing
        
        Respond in JSON format:
        {{
            "value": [processed value - string, array, or object as needed],
            "confidence": [0.0-1.0],
            "reasoning": "explanation of how you processed the answer"
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_answer)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            return json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
            return {
                "value": user_answer,
                "confidence": 0.6,
                "reasoning": "Direct user input"
            }
    
    def _build_field_question_prompt(self, field_name: str, field_def: Any, 
                                   similar_challenges: List[Dict], state: ConversationState) -> str:
        """Build system prompt for asking field questions"""
        
        examples = ""
        if similar_challenges:
            examples = "\n\nExamples from similar challenges:\n"
            for challenge in similar_challenges[:2]:
                if field_name in challenge:
                    examples += f"- {challenge[field_name]}\n"
        
        context = f"""
        You are asking about the '{field_name}' field for a challenge specification.
        
        Field Details:
        - Type: {field_def.field_type}
        - Required: {field_def.required}
        - Description: {field_def.description or 'No description provided'}
        
        Challenge Context:
        - Platform: {state.platform}
        - Challenge Type: {state.challenge_type}
        - Scope: {state.user_responses.get('confirmed_scope', 'Not specified')}
        
        {examples}
        
        Ask a clear, specific question that will help the user provide the right information for this field.
        Make it conversational and helpful, not robotic.
        """
        
        return context
    
    def _get_next_field(self, state_dict: Dict[str, Any]) -> str:
        """Get the next field that needs to be filled"""
        completed = set(state_dict.get("completed_fields", []))
        required = state_dict.get("required_fields", [])
        
        # First, handle required fields
        for field in required:
            if field not in completed:
                return field
        
        # Then handle optional fields
        if state_dict.get("schema"):
            all_fields = list(state_dict["schema"]["fields"].keys())
            for field in all_fields:
                if field not in completed:
                    return field
        
        return None
    
    def _create_schema_intro_message(self, schema: Any, state: ConversationState) -> str:
        """Create an introductory message about the schema"""
        field_count = len(schema.fields)
        required_count = len([f for f in schema.fields.values() if f.required])
        
        return f"""
        Perfect! I've loaded the {schema.platform.value} {schema.challenge_type.value} challenge template.
        
        I'll need to collect information for {field_count} fields ({required_count} required, {field_count - required_count} optional).
        
        Let me start by asking about the most important details for your challenge.
        """
    
    def _is_question_response(self, text: str) -> bool:
        """Check if the text is asking a question rather than providing an answer"""
        question_indicators = ['?', 'what', 'how', 'when', 'where', 'why', 'which', 'can you', 'could you']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)
    
    def _move_to_next_field(self, state: ConversationState, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Move to the next field when current field is invalid"""
        updated_state = state.dict()
        next_field = self._get_next_field(updated_state)
        updated_state["current_field"] = next_field
        
        if next_field:
            return {
                "conversation_state": updated_state,
                "response": f"Let me ask about the {next_field} field instead.",
                "next_node": "field_questions"
            }
        else:
            return {
                "conversation_state": updated_state,
                "response": "I have all the information I need. Let me generate your challenge specification.",
                "next_node": "spec_generation"
            }
    
    def _generate_final_spec(self, state: ConversationState, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final challenge specification"""
        return {
            "conversation_state": state.dict(),
            "response": "Let me compile your complete challenge specification now.",
            "next_node": "spec_generation"
        }