import os
import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
from app.rag.settings import settings

class RAGService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        self.collection_name = "challenge_specs"
        self._ensure_collection_exists()
        self._load_sample_data()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def _load_sample_data(self):
        """Load sample challenge data into vector database"""
        sample_challenges = [
            {
                "title": "Mobile App UI Design for Food Delivery",
                "overview": "Design a modern, user-friendly interface for a food delivery mobile application",
                "platform": "Topcoder",
                "challenge_type": "Design",
                "tech_stack": ["Figma", "Sketch"],
                "timeline": {"submission_days": 7},
                "objectives": ["Create wireframes", "Design UI screens", "Provide style guide"],
                "category": "mobile_ui_design"
            },
            {
                "title": "E-commerce Product Recommendation API",
                "overview": "Develop a machine learning API that provides personalized product recommendations",
                "platform": "Topcoder",
                "challenge_type": "Development",
                "tech_stack": ["Python", "FastAPI", "TensorFlow", "PostgreSQL"],
                "timeline": {"submission_days": 14},
                "objectives": ["Build recommendation algorithm", "Create REST API", "Implement caching"],
                "category": "api_development"
            },
            {
                "title": "Customer Churn Prediction Model",
                "overview": "Build a machine learning model to predict customer churn for a subscription service",
                "platform": "Kaggle",
                "challenge_type": "Data Science",
                "tech_stack": ["Python", "Pandas", "Scikit-learn", "XGBoost"],
                "timeline": {"submission_days": 21},
                "objectives": ["Data analysis", "Feature engineering", "Model training", "Performance optimization"],
                "category": "machine_learning"
            },
            {
                "title": "React Dashboard Component Library",
                "overview": "Create a reusable component library for admin dashboards",
                "platform": "Topcoder",
                "challenge_type": "Development",
                "tech_stack": ["React", "TypeScript", "Storybook", "CSS-in-JS"],
                "timeline": {"submission_days": 10},
                "objectives": ["Build components", "Create documentation", "Implement tests"],
                "category": "frontend_development"
            }
        ]
        
        # Check if data already exists
        try:
            count = self.client.count(collection_name=self.collection_name)
            if count.count > 0:
                return  # Data already loaded
        except:
            pass
        
        # Add sample data
        for i, challenge in enumerate(sample_challenges):
            self.add_challenge(challenge, f"sample_{i}")
    
    def add_challenge(self, challenge_data: Dict[str, Any], challenge_id: str = None):
        """Add a challenge to the vector database"""
        if not challenge_id:
            challenge_id = f"challenge_{len(self.get_all_challenges())}"
        
        # Create text representation for embedding
        text_content = f"""
        Title: {challenge_data.get('title', '')}
        Overview: {challenge_data.get('overview', '')}
        Objectives: {', '.join(challenge_data.get('objectives', []))}
        Tech Stack: {', '.join(challenge_data.get('tech_stack', []))}
        Platform: {challenge_data.get('platform', '')}
        Challenge Type: {challenge_data.get('challenge_type', '')}
        Category: {challenge_data.get('category', '')}
        """.strip()
        
        # Generate embedding
        embedding = self.embedding_model.encode(text_content).tolist()
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=challenge_id,
                    vector=embedding,
                    payload=challenge_data
                )
            ]
        )
    
    def search_similar_challenges(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar challenges based on query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in Qdrant
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                challenge_data = result.payload
                challenge_data['similarity_score'] = result.score
                results.append(challenge_data)
            
            return results
        except Exception as e:
            print(f"Error searching challenges: {e}")
            return []
    
    def get_challenges_by_category(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get challenges by category"""
        try:
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value=category)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            )
            
            return [result.payload for result in search_results[0]]
        except Exception as e:
            print(f"Error getting challenges by category: {e}")
            return []
    
    def get_all_challenges(self) -> List[Dict[str, Any]]:
        """Get all challenges from the database"""
        try:
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True
            )
            
            return [result.payload for result in search_results[0]]
        except Exception as e:
            print(f"Error getting all challenges: {e}")
            return []
    
    def suggest_timeline(self, challenge_type: str, complexity: str = "medium") -> Dict[str, int]:
        """Suggest timeline based on similar challenges"""
        similar = self.search_similar_challenges(f"{challenge_type} {complexity}")
        
        if similar:
            # Average timeline from similar challenges
            timelines = [c.get('timeline', {}).get('submission_days', 7) for c in similar if c.get('timeline')]
            if timelines:
                avg_days = sum(timelines) // len(timelines)
                return {"submission_days": avg_days}
        
        # Default timelines by challenge type
        defaults = {
            "Design": {"submission_days": 7},
            "Development": {"submission_days": 14},
            "Data Science": {"submission_days": 21},
            "First2Finish": {"submission_days": 3},
            "Bug Hunt": {"submission_days": 5}
        }
        
        return defaults.get(challenge_type, {"submission_days": 7})
    
    def suggest_tech_stack(self, project_description: str, challenge_type: str) -> List[str]:
        """Suggest tech stack based on similar challenges"""
        query = f"{project_description} {challenge_type}"
        similar = self.search_similar_challenges(query, limit=3)
        
        tech_suggestions = []
        for challenge in similar:
            tech_stack = challenge.get('tech_stack', [])
            tech_suggestions.extend(tech_stack)
        
        # Return most common technologies
        from collections import Counter
        if tech_suggestions:
            common_tech = Counter(tech_suggestions).most_common(5)
            return [tech[0] for tech in common_tech]
        
        return []

# Global instance
rag_service = RAGService()