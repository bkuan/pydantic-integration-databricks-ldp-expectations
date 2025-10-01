"""
Validator analysis module for understanding pydantic validation logic.
"""

import ast
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging

from .discovery import ModelInfo, ValidatorInfo
from .config import Config


@dataclass
class ValidationRule:
    """Represents a validation rule that can be converted to DLT expectation."""
    field_name: str
    rule_type: str  # e.g., "format", "range", "required", "custom"
    condition: str  # SQL condition for DLT expectation
    description: str
    action: str = "drop"  # drop, fail, warn
    
    @property
    def expectation_name(self) -> str:
        """Generate expectation name for DLT."""
        return f"{self.rule_type}_{self.field_name}".lower()


@dataclass
class ModelValidationSchema:
    """Complete validation schema for a pydantic model."""
    model_name: str
    model_path: str
    rules: List[ValidationRule]
    dependencies: Set[str]
    
    def get_rules_by_type(self, rule_type: str) -> List[ValidationRule]:
        """Get rules filtered by type."""
        return [rule for rule in self.rules if rule.rule_type == rule_type]


class ValidatorAnalyzer:
    """Analyzes pydantic validators and field constraints to generate DLT expectations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Built-in field type mappings
        self.type_mappings = {
            "str": self._analyze_string_field,
            "int": self._analyze_integer_field, 
            "float": self._analyze_float_field,
            "bool": self._analyze_boolean_field,
            "datetime": self._analyze_datetime_field,
            "date": self._analyze_date_field,
            "EmailStr": self._analyze_email_field,
        }
        
        # Common validation patterns
        self.validation_patterns = {
            "email": r"email.*RLIKE.*'[^']*'",
            "zipcode": r"zip.*RLIKE.*'\^\\d\{5\}",
            "phone": r"phone.*RLIKE.*'\+?[0-9]",
            "url": r"url.*RLIKE.*'https?://"
        }
    
    def analyze_model(self, model_info: ModelInfo, validators: Dict[str, ValidatorInfo]) -> ModelValidationSchema:
        """
        Analyze a pydantic model to generate validation schema.
        
        Args:
            model_info: Information about the pydantic model
            validators: Dictionary of available validator functions
            
        Returns:
            Complete validation schema for the model
        """
        self.logger.info(f"Analyzing model: {model_info.full_name}")
        
        rules = []
        dependencies = set()
        
        # Analyze each field in the model
        for field_name, field_info in model_info.fields.items():
            field_rules = self._analyze_field(field_name, field_info, validators)
            rules.extend(field_rules)
        
        # Analyze model-level validators
        for validator_name in model_info.validators:
            validator_rules = self._analyze_model_validator(validator_name, model_info, validators)
            rules.extend(validator_rules)
        
        # Analyze nested model dependencies
        dependencies.update(self._find_nested_dependencies(model_info))
        
        schema = ModelValidationSchema(
            model_name=model_info.name,
            model_path=model_info.full_name,
            rules=rules,
            dependencies=dependencies
        )
        
        self.logger.info(f"Generated {len(rules)} validation rules for {model_info.name}")
        return schema
    
    def _analyze_field(self, field_name: str, field_info: Dict[str, Any], validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """Analyze a single field to generate validation rules."""
        rules = []
        
        field_type = field_info["type"]
        field_default = field_info.get("default")
        annotation = field_info.get("annotation")
        
        # Check if field is required
        if field_default is None:
            rules.append(ValidationRule(
                field_name=field_name,
                rule_type="required",
                condition=f"{field_name} IS NOT NULL",
                description=f"{field_name} is required"
            ))
        
        # Analyze based on type
        base_type = self._extract_base_type(field_type)
        if base_type in self.type_mappings:
            type_rules = self.type_mappings[base_type](field_name, field_type, annotation)
            rules.extend(type_rules)
        
        # Check for custom validators in type annotations
        if "Annotated" in field_type:
            custom_rules = self._analyze_annotated_field(field_name, field_type, validators)
            rules.extend(custom_rules)
        
        # Check for Field constraints
        if "Field(" in str(annotation):
            constraint_rules = self._analyze_field_constraints(field_name, annotation)
            rules.extend(constraint_rules)
        
        return rules
    
    def _extract_base_type(self, type_str: str) -> str:
        """Extract base type from complex type annotation."""
        # Handle common patterns like Optional[str], List[int], etc.
        if "[" in type_str:
            # Extract type from generic like List[str] -> str
            match = re.search(r"(\w+)\[([^\]]+)\]", type_str)
            if match:
                container, inner = match.groups()
                if container in {"Optional", "Union"}:
                    # For Optional[str] -> str
                    return inner.split(",")[0].strip()
                elif container in {"List", "Set"}:
                    return inner.strip()
        
        # Clean up common type prefixes
        type_str = type_str.replace("typing.", "").replace("pydantic.", "")
        
        return type_str
    
    def _analyze_string_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze string field for validation rules."""
        rules = []
        
        # Basic string validation
        rules.append(ValidationRule(
            field_name=field_name,
            rule_type="type_check", 
            condition=f"{field_name} IS NOT NULL AND trim({field_name}) != ''",
            description=f"{field_name} must be a non-empty string"
        ))
        
        return rules
    
    def _analyze_integer_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze integer field for validation rules."""
        return [ValidationRule(
            field_name=field_name,
            rule_type="type_check",
            condition=f"{field_name} IS NOT NULL AND {field_name} RLIKE '^-?[0-9]+$'",
            description=f"{field_name} must be a valid integer"
        )]
    
    def _analyze_float_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze float field for validation rules.""" 
        return [ValidationRule(
            field_name=field_name,
            rule_type="type_check",
            condition=f"{field_name} IS NOT NULL AND {field_name} RLIKE '^-?[0-9]*\\.?[0-9]+$'",
            description=f"{field_name} must be a valid number"
        )]
    
    def _analyze_boolean_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze boolean field for validation rules."""
        return [ValidationRule(
            field_name=field_name,
            rule_type="type_check",
            condition=f"{field_name} IS NOT NULL AND {field_name} IN ('true', 'false', '1', '0')",
            description=f"{field_name} must be a valid boolean"
        )]
    
    def _analyze_datetime_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze datetime field for validation rules."""
        return [ValidationRule(
            field_name=field_name,
            rule_type="type_check", 
            condition=f"{field_name} IS NOT NULL AND {field_name} RLIKE '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'",
            description=f"{field_name} must be a valid datetime"
        )]
    
    def _analyze_date_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze date field for validation rules."""
        return [ValidationRule(
            field_name=field_name,
            rule_type="type_check",
            condition=f"to_date({field_name}, 'yyyy-MM-dd') IS NOT NULL",
            description=f"{field_name} must be a valid date"
        )]
    
    def _analyze_email_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze EmailStr field for validation rules."""
        return [ValidationRule(
            field_name=field_name,
            rule_type="format",
            condition=f"{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$'",
            description=f"{field_name} must be a valid email address"
        )]
    
    def _analyze_annotated_field(self, field_name: str, field_type: str, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """Analyze Annotated field with custom validators."""
        rules = []
        
        # Extract validator names from Annotated type
        validator_pattern = r"AfterValidator\((\w+)\)"
        matches = re.findall(validator_pattern, field_type)
        
        for validator_name in matches:
            # Find matching validator function
            for full_validator_name, validator_info in validators.items():
                if validator_info.name == validator_name:
                    custom_rule = self._convert_validator_to_rule(field_name, validator_info)
                    if custom_rule:
                        rules.append(custom_rule)
        
        return rules
    
    def _analyze_field_constraints(self, field_name: str, annotation: Any) -> List[ValidationRule]:
        """Analyze Field() constraints for validation rules."""
        rules = []
        
        # This would need to parse the Field() call from AST
        # For now, we'll implement basic pattern matching
        annotation_str = str(annotation) if annotation else ""
        
        # Look for min_length
        min_length_match = re.search(r"min_length=(\d+)", annotation_str)
        if min_length_match:
            min_length = int(min_length_match.group(1))
            rules.append(ValidationRule(
                field_name=field_name,
                rule_type="min_length",
                condition=f"length(COALESCE({field_name}, '')) >= {min_length}",
                description=f"{field_name} must be at least {min_length} characters"
            ))
        
        # Look for max_length
        max_length_match = re.search(r"max_length=(\d+)", annotation_str)
        if max_length_match:
            max_length = int(max_length_match.group(1))
            rules.append(ValidationRule(
                field_name=field_name,
                rule_type="max_length", 
                condition=f"length(COALESCE({field_name}, '')) <= {max_length}",
                description=f"{field_name} must be at most {max_length} characters"
            ))
        
        return rules
    
    def _convert_validator_to_rule(self, field_name: str, validator_info: ValidatorInfo) -> Optional[ValidationRule]:
        """Convert a custom validator function to a validation rule."""
        validator_name = validator_info.name
        
        # Built-in validator mappings
        validator_mappings = {
            "email_validator": ValidationRule(
                field_name=field_name,
                rule_type="custom",
                condition=f"{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$' AND length({field_name}) <= 254",
                description=f"{field_name} must be a valid email with business rules"
            ),
            "zipcode_validator": ValidationRule(
                field_name=field_name, 
                rule_type="custom",
                condition=f"{field_name} RLIKE '^\\d{{5}}(-\\d{{4}})?$'",
                description=f"{field_name} must be 5-digit or ZIP+4 format"
            ),
            "sex_validator": ValidationRule(
                field_name=field_name,
                rule_type="custom", 
                condition=f"{field_name} IN ('M', 'F')",
                description=f"{field_name} must be 'M' or 'F'"
            ),
            "past_date_validator": ValidationRule(
                field_name=field_name,
                rule_type="custom",
                condition=f"to_date({field_name}, 'yyyy-MM-dd') <= current_date()",
                description=f"{field_name} cannot be in the future"
            )
        }
        
        return validator_mappings.get(validator_name)
    
    def _analyze_model_validator(self, validator_name: str, model_info: ModelInfo, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """Analyze model-level validators."""
        # This would analyze @model_validator decorated functions
        # For now, return empty list
        return []
    
    def _find_nested_dependencies(self, model_info: ModelInfo) -> Set[str]:
        """Find nested model dependencies."""
        dependencies = set()
        
        for field_name, field_info in model_info.fields.items():
            field_type = field_info["type"]
            
            # Look for other model references in field types
            # This is a simplified check - would need more sophisticated parsing
            for other_class in model_info.imports.values():
                if other_class in field_type and other_class != model_info.full_name:
                    dependencies.add(other_class)
        
        return dependencies
    
    def generate_schema_summary(self, schema: ModelValidationSchema) -> Dict[str, Any]:
        """Generate a summary of the validation schema."""
        rule_counts = {}
        for rule in schema.rules:
            rule_counts[rule.rule_type] = rule_counts.get(rule.rule_type, 0) + 1
        
        return {
            "model_name": schema.model_name,
            "total_rules": len(schema.rules),
            "rule_breakdown": rule_counts,
            "dependencies": list(schema.dependencies),
            "required_fields": len(schema.get_rules_by_type("required")),
            "custom_validators": len(schema.get_rules_by_type("custom"))
        }

