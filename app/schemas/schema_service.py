import json
import os
from typing import Dict, Optional
from app.models.types import ChallengeSchema, Platform, ChallengeType, FieldDefinition

class SchemaService:
    def __init__(self, schema_dir: str = "data/schemas"):
        self.schema_dir = schema_dir
        self._schemas: Dict[str, ChallengeSchema] = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all schema files from the schemas directory"""
        if not os.path.exists(self.schema_dir):
            os.makedirs(self.schema_dir, exist_ok=True)
            self._create_default_schemas()
        
        for filename in os.listdir(self.schema_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.schema_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        schema_data = json.load(f)
                        schema = ChallengeSchema(**schema_data)
                        key = f"{schema.platform.value}_{schema.challenge_type.value}"
                        self._schemas[key] = schema
                except Exception as e:
                    print(f"Error loading schema {filename}: {e}")
    
    def _create_default_schemas(self):
        """Create default schema files if they don't exist"""
        default_schemas = [
            {
                "platform": "Topcoder",
                "challenge_type": "Design",
                "fields": {
                    "title": {"required": True, "field_type": "text", "description": "Challenge title"},
                    "overview": {"required": True, "field_type": "text", "description": "Detailed overview"},
                    "objectives": {"required": True, "field_type": "array", "description": "List of objectives"},
                    "tech_stack": {"required": False, "field_type": "array", "description": "Technologies to use"},
                    "timeline": {"required": True, "field_type": "object", "description": "Timeline details"},
                    "prize_structure": {"required": True, "field_type": "object", "description": "Prize breakdown"},
                    "judging_criteria": {"required": True, "field_type": "array", "description": "Judging criteria"}
                }
            },
            {
                "platform": "Topcoder",
                "challenge_type": "Development",
                "fields": {
                    "title": {"required": True, "field_type": "text"},
                    "overview": {"required": True, "field_type": "text"},
                    "objectives": {"required": True, "field_type": "array"},
                    "tech_stack": {"required": True, "field_type": "array"},
                    "timeline": {"required": True, "field_type": "object"},
                    "prize_structure": {"required": True, "field_type": "object"},
                    "judging_criteria": {"required": True, "field_type": "array"},
                    "submission_requirements": {"required": True, "field_type": "array"},
                    "testing_requirements": {"required": False, "field_type": "text"}
                }
            },
            {
                "platform": "Kaggle",
                "challenge_type": "Data Science",
                "fields": {
                    "title": {"required": True, "field_type": "text"},
                    "overview": {"required": True, "field_type": "text"},
                    "objectives": {"required": True, "field_type": "array"},
                    "dataset_description": {"required": True, "field_type": "text"},
                    "evaluation_metric": {"required": True, "field_type": "text"},
                    "timeline": {"required": True, "field_type": "object"},
                    "prize_structure": {"required": True, "field_type": "object"},
                    "submission_format": {"required": True, "field_type": "text"}
                }
            }
        ]
        
        for schema_data in default_schemas:
            filename = f"{schema_data['platform']}_{schema_data['challenge_type']}.json"
            filepath = os.path.join(self.schema_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(schema_data, f, indent=2)
    
    def get_schema(self, platform: Platform, challenge_type: ChallengeType) -> Optional[ChallengeSchema]:
        """Get schema for specific platform and challenge type"""
        key = f"{platform.value}_{challenge_type.value}"
        return self._schemas.get(key)
    
    def get_available_schemas(self) -> List[Dict[str, str]]:
        """Get list of available schema combinations"""
        return [
            {
                "platform": schema.platform.value,
                "challenge_type": schema.challenge_type.value,
                "key": key
            }
            for key, schema in self._schemas.items()
        ]
    
    def get_required_fields(self, schema: ChallengeSchema) -> List[str]:
        """Get list of required field names from schema"""
        return [
            field_name for field_name, field_def in schema.fields.items()
            if field_def.required
        ]
    
    def get_field_definition(self, schema: ChallengeSchema, field_name: str) -> Optional[FieldDefinition]:
        """Get definition for a specific field"""
        return schema.fields.get(field_name)
    
    def validate_field_value(self, schema: ChallengeSchema, field_name: str, value: Any) -> bool:
        """Validate a field value against its schema definition"""
        field_def = self.get_field_definition(schema, field_name)
        if not field_def:
            return False
        
        # Basic type validation
        if field_def.field_type == "array" and not isinstance(value, list):
            return False
        elif field_def.field_type == "object" and not isinstance(value, dict):
            return False
        elif field_def.field_type == "text" and not isinstance(value, str):
            return False
        elif field_def.field_type == "number" and not isinstance(value, (int, float)):
            return False
        
        # Additional validation rules can be added here
        return True

# Global instance
schema_service = SchemaService()