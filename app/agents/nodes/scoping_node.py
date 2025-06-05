from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.models.types import ConversationState, ProjectStage, Platform, ChallengeType, ReasoningTrace
from app.services.rag_service import rag_service
from app.config.settings import settings
import re

class ScopingNode:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main scoping node logic"""
        conversation_state = ConversationState(**state.get("conversation_state", {}))
        
        if not conversation_state.scope_confirmed:
            return self._handle_scoping_dialogue(conversation_state, state)
        else:
            # Move to schema selection
            return self._handle_schema_selection(conversation_state, state)
    
    def _handle_scoping_dialogue(self, state: ConversationState, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the scoping conversation"""
        user_message = full_state.get("user_input", "")
        
        # Get similar challenges for context
        similar_challenges = rag_service.search_similar_challenges(state.user_input, limit=3)
        
        # Build context for the LLM
        system_prompt = self._build_scoping_system_prompt(similar_challenges)
        
        # Create conversation history
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in full_state.get("messages", []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add current user message
        if user_message:
            messages.append(HumanMessage(content=user_message))
        
        # Get LLM response
        response = self.llm.invoke(messages)
        ai_response = response.content
        
        # Parse the response to extract structured information
        parsed_info = self._parse_scoping_response(ai_response, state)
        
        # Update state
        updated_state = state.dict()
        updated_state.update(parsed_info["state_updates"])
        
        # Add reasoning trace
        if parsed_info.get("reasoning"):
            reasoning_trace = ReasoningTrace(
                field="scoping",
                source="scoping_dialogue",
                confidence=parsed_info.get("confidence", 0.7),
                reasoning=parsed_info["reasoning"]
            )
            updated_state["reasoning_traces"].append(reasoning_trace.dict())
        
        return {
            "conversation_state": updated_state,
            "response": ai_response,
            "next_node": "scoping" if not updated_state.get("scope_confirmed", False) else "schema_selection"
        }
    
    def _handle_schema_selection(self, state: ConversationState, full_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle platform and challenge type selection"""
        user_message = full_state.get("user_input", "")
        
        system_prompt = """
        You are helping the user select the appropriate platform and challenge type for their project.
        
        Available Platforms:
        - Topcoder: Great for design and development challenges
        - Kaggle: Best for data science and ML competitions
        - HeroX: Good for innovation challenges
        - Zindi: Africa-focused data science competitions
        - Internal: Company internal challenges
        
        Available Challenge Types:
        - Design: UI/UX design, mockups, prototypes
        - Development: Code implementation, APIs, applications
        - Data Science: ML models, data analysis, predictions
        - First2Finish: Quick implementation challenges
        - Bug Hunt: Finding and fixing bugs
        
        Based on the user's confirmed scope, recommend the best platform and challenge type.
        Respond in this format:
        
        RECOMMENDATION: [Platform] - [Challenge Type]
        REASONING: [Why this combination is best]
        CONFIDENCE: [0.0-1.0]
        
        If you need more information, ask specific questions.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Scope: {state.user_responses.get('confirmed_scope', state.user_input)}\nUser input: {user_message}")
        ]
        
        response = self.llm.invoke(messages)
        ai_response = response.content
        
        # Parse platform and challenge type
        platform_match = re.search(r'RECOMMENDATION:\s*(\w+)\s*-\s*(\w+)', ai_response)
        if platform_match:
            try:
                platform = Platform(platform_match.group(1))
                challenge_type = ChallengeType(platform_match.group(2))
                
                updated_state = state.dict()
                updated_state["platform"] = platform.value
                updated_state["challenge_type"] = challenge_type.value
                
                return {
                    "conversation_state": updated_state,
                    "response": ai_response,
                    "next_node": "schema_loading"
                }
            except ValueError:
                pass
        
        # If parsing failed or needs more info
        return {
            "conversation_state": state.dict(),
            "response": ai_response,
            "next_node": "schema_selection"
        }
    
    def _build_scoping_system_prompt(self, similar_challenges: List[Dict[str, Any]]) -> str:
        """Build system prompt for scoping dialogue"""
        similar_context = ""
        if similar_challenges:
            similar_context = "\n\nSimilar successful challenges for reference:\n"
            for i, challenge in enumerate(similar_challenges[:2], 1):
                similar_context += f"{i}. {challenge.get('title', 'Unknown')}\n   Scope: {challenge.get('overview', '')[:100]}...\n"
        
        return f"""
        You are an AI Copilot helping users define and scope challenges for platforms like Topcoder, Kaggle, etc.
        
        Your role is to:
        1. Understand the user's high-level goal
        2. Assess if it's feasible for a single challenge
        3. Guide them to a well-scoped unit of work
        4. NOT impose a scope, but help them find the right one
        
        Current conversation goal: Help the user scope their project appropriately.
        
        SCOPING GUIDELINES:
        - A good challenge should be completable in 3-21 days
        - Should have clear, measurable deliverables
        - Should not be too broad (entire apps) or too narrow (single function)
        - Should consider what's already done vs. what needs to be built
        
        CONVERSATION FLOW:
        1. Ask about project status and what's already done
        2. Clarify the specific part they want to focus on
        3. Propose a scoped version and get their confirmation
        4. Once they confirm, mark scope as confirmed
        
        RESPONSE FORMAT:
        - Ask thoughtful, specific questions
        - Propose concrete, actionable scopes
        - If user confirms scope, end with: "SCOPE_CONFIRMED: [brief description]"
        
        {similar_context}
        
        Remember: Guide, don't impose. Help them find their ideal scope.
        """
    
    def _parse_scoping_response(self, response: str, state: ConversationState) -> Dict[str, Any]:
        """Parse the scoping response for structured information"""
        result = {
            "state_updates": {},
            "reasoning": "",
            "confidence": 0.7
        }
        
        # Check if scope is confirmed
        scope_match = re.search(r'SCOPE_CONFIRMED:\s*(.+)', response, re.IGNORECASE)
        if scope_match:
            confirmed_scope = scope_match.group(1).strip()
            result["state_updates"]["scope_confirmed"] = True
            result["state_updates"]["user_responses"] = state.user_responses.copy()
            result["state_updates"]["user_responses"]["confirmed_scope"] = confirmed_scope
            result["reasoning"] = f"User confirmed scope: {confirmed_scope}"
            result["confidence"] = 0.9
        
        # Extract project stage if mentioned
        stage_patterns = {
            ProjectStage.IDEA: r'\b(idea|concept|brainstorm)\b',
            ProjectStage.DESIGN: r'\b(design|mockup|wireframe|prototype)\b',
            ProjectStage.DEVELOPMENT: r'\b(develop|code|implement|build)\b',
            ProjectStage.TESTING: r'\b(test|qa|debug)\b',
            ProjectStage.POC: r'\b(poc|proof.of.concept|pilot)\b',
            ProjectStage.EXISTING: r'\b(existing|already|current)\b'
        }
        
        for stage, pattern in stage_patterns.items():
            if re.search(pattern, response.lower()):
                result["state_updates"]["project_stage"] = stage.value
                break
        
        return result