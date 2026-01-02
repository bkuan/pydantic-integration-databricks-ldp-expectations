"""
Enhanced validator analysis module with AI-powered conversion and dependency analysis.
"""

import ast
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

from .discovery import ModelInfo, ValidatorInfo
from .config import Config, ValidatorAnalysisConfig
from .ai_utils import AIQueryExecutor, AIPromptTemplates, AIResponseParser


@dataclass
class ValidationRule:
    """Represents a validation rule that can be converted to DLT expectation."""
    field_name: str
    rule_type: str  # e.g., "format", "range", "required", "custom"
    condition: str  # SQL condition for DLT expectation
    description: str
    action: str = "drop"  # drop, fail, warn
    _unique_expectation_name: Optional[str] = None  # For custom expectation names
    
    @property
    def expectation_name(self) -> str:
        """Generate expectation name for DLT."""
        if self._unique_expectation_name:
            return self._unique_expectation_name
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
    """Enhanced analyzer for pydantic validators with AI conversion and dependency analysis."""
    
    def __init__(self, config: Config, analysis_config: ValidatorAnalysisConfig = None):
        self.config = config
        
        # Use analysis_config from main config if not provided separately
        # This ensures AI settings are controlled by the main config
        if analysis_config is None:
            self.analysis_config = config.validation.validator_analysis
            # Sync AI settings from main config
            self.analysis_config.ai_conversion_enabled = config.validation.ai.enabled
        else:
            self.analysis_config = analysis_config
            
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI query executor
        self.ai_executor = AIQueryExecutor(config, self.logger)
        
        # Validator dependency tracking
        self.discovered_validators = {}
        self.dependency_graph = {}
        self.complexity_cache = {}
        
        # Built-in field type mappings - Python to SQL type validation
        # These use proper SQL typeof() and TRY_CAST() instead of regex patterns
        self.type_mappings = {
            "str": self._analyze_string_field,      # -> typeof() = 'string'
            "int": self._analyze_integer_field,     # -> typeof() IN ('integer', 'bigint', 'int')
            "float": self._analyze_float_field,     # -> typeof() IN ('double', 'float', 'decimal')
            "bool": self._analyze_boolean_field,    # -> typeof() = 'boolean'
            "datetime": self._analyze_datetime_field, # -> typeof() IN ('timestamp', 'datetime')
            "date": self._analyze_date_field,       # -> typeof() = 'date'
            "EmailStr": self._analyze_email_field,  # -> Custom email validation
        }
        
        # Python to Databricks SQL type mapping reference
        self.python_to_sql_types = {
            "str": ["string", "varchar"],
            "int": ["integer", "bigint", "int", "smallint", "tinyint"],
            "float": ["double", "float", "decimal"],
            "bool": ["boolean"],
            "datetime": ["timestamp", "datetime"],
            "date": ["date"],
            "list": ["array"],
            "dict": ["map", "struct"]
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
        Analyze a pydantic model to generate validation schema with enhanced AI conversion.
        
        Args:
            model_info: Information about the pydantic model
            validators: Dictionary of available validator functions
            
        Returns:
            Complete validation schema for the model
        """
        self.logger.info(f"Analyzing model: {model_info.full_name}")
        
        # Store validators for dependency analysis
        self.discovered_validators = validators
        
        # Build dependency graph if AI conversion is enabled
        if self.analysis_config.ai_conversion_enabled:
            self._build_dependency_graph(validators)
        
        rules = []
        dependencies = set()
        
        # Analyze each field in the model
        for field_name, field_info in model_info.fields.items():
            field_rules = self._analyze_field(field_name, field_info, validators)
            rules.extend(field_rules)
        
        # Analyze field-level validators (@field_validator decorators)
        if hasattr(model_info, 'field_validators') and model_info.field_validators:
            field_validator_rules = self._process_field_validators(model_info, validators)
            rules.extend(field_validator_rules)
        
        # Skip the old model validator processing - handled later in dedicated section
        
        # Skip the old fallback logic since we now have specific handling for both types
        
        # Analyze nested model dependencies
        dependencies.update(self._find_nested_dependencies(model_info))
        
        # Add comprehensive custom validator rules (if AI enabled)
        if self.analysis_config.ai_conversion_enabled:
            comprehensive_rules = self._generate_comprehensive_ai_rules(model_info, validators)
            rules.extend(comprehensive_rules)
        
        # Add model-level validator rules (NEW - for @model_validator decorators)
        if model_info.validators:
            model_level_rules = self._process_model_validators(model_info, validators)
            rules.extend(model_level_rules)
        
        # Deduplicate rules before creating schema
        deduplicated_rules = self._deduplicate_validation_rules(rules)
        
        # Ensure globally unique expectation names to prevent dictionary key conflicts in @dlt.expect_all_or_drop
        final_rules = self._ensure_globally_unique_expectation_names(deduplicated_rules)
        
        # SAFETY NET: Clean all rule conditions and descriptions to eliminate ANY remaining verbose patterns
        cleaned_rules = self._clean_all_rule_content(final_rules)
        
        schema = ModelValidationSchema(
            model_name=model_info.name,
            model_path=model_info.full_name,
            rules=cleaned_rules,
            dependencies=dependencies
        )
        
        original_count = len(rules)
        deduplicated_count = len(deduplicated_rules)
        final_count = len(final_rules)
        
        if original_count != deduplicated_count:
            self.logger.info(f"Deduplicated {original_count - deduplicated_count} duplicate rules")
        
        self.logger.info(f"Generated {final_count} validation rules for {model_info.name} (all expectation names unique)")
        return schema
    
    def _analyze_field(self, field_name: str, field_info: Dict[str, Any], validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """Analyze a single field to generate validation rules."""
        rules = []
        
        field_type = field_info["type"]
        field_default = field_info.get("default")
        annotation = field_info.get("annotation")
        
        # Check if field is required
        if self._is_field_required(field_name, field_default, field_type):
            rules.append(ValidationRule(
                field_name=field_name,
                rule_type="required_field",
                condition=f"{field_name} IS NOT NULL",
                description=f"{field_name} is required"
            ))
        
        # Analyze based on type
        base_type = self._extract_base_type(field_type)
        if base_type in self.type_mappings:
            type_rules = self.type_mappings[base_type](field_name, field_type, annotation)
            rules.extend(type_rules)
        
        # Check for custom types and validators in type annotations
        if "Annotated" in field_type:
            custom_rules = self._analyze_annotated_field(field_name, field_type, validators)
            rules.extend(custom_rules)
        elif field_type not in ['str', 'int', 'float', 'bool', 'list', 'dict'] and not field_type.startswith('Optional['):
            # This is a custom type (like Email, ZipCode, etc.)
            # Check if there's a corresponding validator function first
            custom_type_rules = self._analyze_custom_type_with_validator_reference(field_name, field_type, validators)
            rules.extend(custom_type_rules)
        
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
    
    def _is_valid_sql_condition(self, condition: str) -> bool:
        """
        Check if a condition appears to be valid SQL that can be used in DLT expectations.
        
        This helps distinguish between actual SQL conditions and non-SQL text that should
        not be picked up for DLT expectations.
        """
        if not condition or not isinstance(condition, str):
            return False
        
        condition = condition.strip()
        
        # Skip reference conditions (not actual SQL)
        if condition.startswith("see validator:") or condition.startswith("-- See validator:"):
            return False
        
        # Skip obvious non-SQL patterns
        if condition.startswith("#") or condition.startswith("//"):
            return False
        
        # Check for SQL-like patterns
        sql_indicators = [
            # SQL functions
            "typeof(", "try_cast(", "length(", "coalesce(", "trim(",
            "current_date()", "current_timestamp()", "regexp_like(", "rlike",
            
            # SQL operators  
            " is not null", " is null", " in (", " not in (", 
            " and ", " or ", " between ", " like ", " not like ",
            
            # SQL keywords
            " case when ", " then ", " else ", " end",
            " where ", " having ",
            
            # Common SQL patterns
            "= '", "!= '", "<> '", "> ", "< ", ">= ", "<= ",
            "= true", "= false", "is true", "is false"
        ]
        
        condition_lower = condition.lower()
        
        # Must have at least one SQL indicator
        has_sql_indicator = any(indicator in condition_lower for indicator in sql_indicators)
        
        if not has_sql_indicator:
            return False
        
        # Additional validation: should not contain obvious Python code patterns
        python_patterns = [
            "def ", "class ", "import ", "from ", "raise ", "return ",
            "if __name__", "print(", "len(", ".split(", ".join(", ".lower()",
            "isinstance(", "hasattr(", "getattr(", "setattr("
        ]
        
        has_python_patterns = any(pattern in condition_lower for pattern in python_patterns)
        
        return has_sql_indicator and not has_python_patterns
    
    def _format_validation_description(self, base_description: str, condition: str, validator_name: str = None, rule_type: str = None) -> str:
        """
        Format validation description with SQL indicator if the condition is valid SQL.
        
        Custom validator rules should NOT have SQL indicators.
        This ensures only proper SQL conditions get picked up for DLT expectations.
        """
        # Custom validator rules should not have SQL indicators
        if rule_type == "custom_validator":
            return base_description
            
        if self._is_valid_sql_condition(condition):
            # Add SQL indicator for valid SQL conditions
            if validator_name:
                return f"SQL: {base_description} (from {validator_name})"
            else:
                return f"SQL: {base_description}"
        else:
            # No SQL indicator for non-SQL conditions
            return base_description
    
    def _is_field_required(self, field_name: str, field_default: Optional[str], field_type: str) -> bool:
        """
        Determine if a field is required based on its default value and type.
        
        A field is considered required if:
        1. It has no default value (field_default is None)
        2. It uses Field() without an explicit default (e.g., Field(pattern=...))
        3. It's not Optional[T] or Union[T, None]
        
        A field is NOT required if:
        1. It has an explicit default value (e.g., = None, = "default")
        2. It's Optional[T] type
        3. It uses Field(default=...) with an explicit default
        """
        
        # Check if it's an Optional type
        if self._is_optional_type(field_type):
            return False
        
        # No default at all - definitely required
        if field_default is None:
            return True
        
        # Check if it's a Field() call without an explicit default
        if field_default and field_default.startswith("Field("):
            # Parse the Field() call to see if it has a default parameter
            return not self._field_has_explicit_default(field_default)
        
        # Has some other default value - not required
        return False
    
    def _is_optional_type(self, field_type: str) -> bool:
        """Check if a field type is Optional or Union with None."""
        # Common patterns for optional types
        optional_patterns = [
            "Optional[",
            "Union[",  # Could be Union[str, None]
            "| None",  # Python 3.10+ syntax
            "typing.Optional[",
            "typing.Union["
        ]
        
        return any(pattern in field_type for pattern in optional_patterns)
    
    def _field_has_explicit_default(self, field_call: str) -> bool:
        """
        Check if a Field() call contains an explicit default parameter.
        
        Examples:
        - Field(pattern='...') -> False (no default)
        - Field(default='value') -> True (has default)
        - Field(default=None, pattern='...') -> True (has default)
        """
        
        # Simple parsing to check for default= parameter
        # This is a basic implementation - could be made more robust with actual AST parsing
        
        if "default=" in field_call:
            return True
        
        # Check for positional default (first parameter without name)
        # Field("default_value", pattern="...") 
        import re
        match = re.match(r"Field\(\s*['\"]([^'\"]*)['\"]", field_call)
        if match:
            return True
        
        # Check for other patterns that indicate a default
        if "default_factory=" in field_call:
            return True
        
        return False
    
    def _analyze_string_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze string field for validation rules."""
        rules = []
        
        # Basic string type validation using SQL typeof
        condition = f"typeof({field_name}) = 'string' AND trim(COALESCE({field_name}, '')) != ''"
        description = self._format_validation_description(
            f"{field_name} must be a non-empty string", condition
        )
        
        rules.append(ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check", 
            condition=condition,
            description=description
        ))
        
        return rules
    
    def _analyze_integer_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze integer field for validation rules."""
        condition = f"typeof({field_name}) IN ('integer', 'bigint', 'int') OR TRY_CAST({field_name} AS INTEGER) IS NOT NULL"
        description = self._format_validation_description(
            f"{field_name} must be a valid integer type", condition
        )
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check",
            condition=condition,
            description=description
        )]
    
    def _analyze_float_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze float field for validation rules.""" 
        condition = f"typeof({field_name}) IN ('double', 'float', 'decimal') OR TRY_CAST({field_name} AS DOUBLE) IS NOT NULL"
        description = self._format_validation_description(
            f"{field_name} must be a valid numeric type", condition
        )
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check",
            condition=condition,
            description=description
        )]
    
    def _analyze_boolean_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze boolean field for validation rules."""
        condition = f"typeof({field_name}) = 'boolean' OR TRY_CAST({field_name} AS BOOLEAN) IS NOT NULL"
        description = self._format_validation_description(
            f"{field_name} must be a valid boolean type", condition
        )
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check",
            condition=condition,
            description=description
        )]
    
    def _analyze_datetime_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze datetime field for validation rules."""
        condition = f"typeof({field_name}) IN ('timestamp', 'datetime') OR TRY_CAST({field_name} AS TIMESTAMP) IS NOT NULL"
        description = self._format_validation_description(
            f"{field_name} must be a valid datetime type", condition
        )
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check",
            condition=condition,
            description=description
        )]
    
    def _analyze_date_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze date field for validation rules."""
        condition = f"typeof({field_name}) = 'date' OR TRY_CAST({field_name} AS DATE) IS NOT NULL"
        description = self._format_validation_description(
            f"{field_name} must be a valid date type", condition
        )
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check",
            condition=condition,
            description=description
        )]
    
    def _analyze_email_field(self, field_name: str, field_type: str, annotation: Any) -> List[ValidationRule]:
        """Analyze EmailStr field for validation rules."""
        condition = f"{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$'"
        description = self._format_validation_description(
            f"{field_name} must be a valid email address", condition
        )
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="format",
            condition=condition,
            description=description
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
    
    def _analyze_custom_type(self, field_name: str, field_type: str) -> List[ValidationRule]:
        """Analyze custom type fields using AI conversion or fallback to basic validation."""
        
        # Try AI conversion if enabled
        if self.analysis_config.ai_conversion_enabled:
            try:
                return self._ai_convert_custom_type(field_name, field_type)
            except Exception as e:
                if self.analysis_config.ai_fallback_to_basic:
                    self.logger.warning(f"AI conversion failed for custom type {field_type}: {e}")
                else:
                    raise
        
        # Fallback to basic type validation
        return self._basic_custom_type_validation(field_name, field_type)
    
    def _analyze_custom_type_with_validator_reference(self, field_name: str, field_type: str, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """
        Analyze custom type fields, creating references to validator functions when they exist.
        
        This prevents duplication by having custom types reference validator functions rather 
        than duplicating the same validation logic.
        """
        
        # Check if there's a corresponding validator function for this custom type
        corresponding_validator = self._find_corresponding_validator(field_name, field_type, validators)
        
        if corresponding_validator:
            # Create a reference rule that points to the validator
            reference_rule = ValidationRule(
                field_name=field_name,
                rule_type="custom_type",
                condition=f"see validator: {corresponding_validator.name}",
                description=f"{field_name} uses custom type {field_type} (validation handled by {corresponding_validator.name})",
                action="reference"
            )
            
            # Also create the actual validator rule
            validator_rule = self._convert_validator_to_rule(field_name, corresponding_validator)
            
            rules = [reference_rule]
            if validator_rule:
                rules.append(validator_rule)
            
            return rules
        else:
            # No corresponding validator found, analyze the custom type directly
            return self._analyze_custom_type(field_name, field_type)
    
    def _find_corresponding_validator(self, field_name: str, field_type: str, validators: Dict[str, ValidatorInfo]) -> Optional[ValidatorInfo]:
        """
        Find a validator function that corresponds to a custom type.
        
        Maps custom types like 'Email', 'ZipCode' to validator functions like 
        'email_validator', 'zipcode_validator'.
        """
        
        # Common mappings between custom types and validator function names
        type_to_validator_mappings = {
            'Email': ['email_validator', 'validate_email'],
            'ZipCode': ['zipcode_validator', 'zip_validator', 'validate_zipcode', 'validate_zip'],
            'Sex': ['sex_validator', 'gender_validator', 'validate_sex'],
            'PastDate': ['past_date_validator', 'date_validator', 'validate_past_date'],
            'Name': ['name_validator', 'validate_name'],
            'Phone': ['phone_validator', 'validate_phone'],
            'URL': ['url_validator', 'validate_url'],
        }
        
        # Look for direct mappings first
        potential_validator_names = type_to_validator_mappings.get(field_type, [])
        
        # Also try field-specific patterns (e.g., user_email field with Email type)
        field_based_patterns = [
            f"{field_name}_validator",
            f"validate_{field_name}",
            f"{field_name.lower()}_validator",
            f"validate_{field_name.lower()}"
        ]
        potential_validator_names.extend(field_based_patterns)
        
        # Search through discovered validators
        for validator_key, validator_info in validators.items():
            validator_name = validator_info.name
            
            # Check direct name matches
            if validator_name in potential_validator_names:
                return validator_info
            
            # Check if validator name contains the field name or type name
            if (field_name.lower() in validator_name.lower() or 
                field_type.lower() in validator_name.lower()):
                return validator_info
        
        return None
    
    def _convert_validator_to_rule(self, field_name: str, validator_info: ValidatorInfo) -> Optional[ValidationRule]:
        """Convert a custom validator function to a validation rule using AI conversion."""
        validator_name = validator_info.name
        
        # Try AI conversion if enabled
        if self.analysis_config.ai_conversion_enabled:
            try:
                complexity = self._classify_validator_complexity(validator_info)
                return self._ai_convert_validator(field_name, validator_info, complexity)
            except Exception as e:
                if self.analysis_config.ai_fallback_to_basic:
                    self.logger.warning(f"AI conversion failed for validator {validator_name}: {e}")
                else:
                    raise
        
        # Fallback to basic validation
        return self._basic_validator_fallback(field_name, validator_info)
    
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
    
    def get_python_to_sql_type_mappings(self) -> Dict[str, List[str]]:
        """Get the Python to SQL type mappings for reference."""
        return self.python_to_sql_types.copy()
    
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
    
    # ===============================================================================
    # ENHANCED AI-POWERED ANALYSIS METHODS
    # ===============================================================================
    
    def _build_dependency_graph(self, validators: Dict[str, ValidatorInfo]) -> None:
        """Build dependency graph for all validators."""
        self.dependency_graph = {}
        
        for validator_key, validator_info in validators.items():
            validator_name = validator_info.name
            dependencies = self._detect_validator_dependencies(validator_info)
            self.dependency_graph[validator_name] = dependencies
    
    def _detect_validator_dependencies(self, validator_info: ValidatorInfo) -> List[str]:
        """Detect what other validators/functions this validator depends on."""
        dependencies = []
        
        if not validator_info.function_def:
            return dependencies
            
        func_ast = validator_info.function_def
        
        # Walk the AST to find function calls
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Call):
                # Handle different types of function calls
                if isinstance(node.func, ast.Name):
                    # Direct function call: some_validator(v)
                    func_name = node.func.id
                    if self._is_validator_function(func_name):
                        dependencies.append(func_name)
                        
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method(v) or module.function(v)
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        method_name = node.func.attr
                        full_name = f"{module_name}.{method_name}"
                        if not self._is_external_dependency(full_name):
                            dependencies.append(method_name)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _is_validator_function(self, func_name: str) -> bool:
        """Check if a function name refers to another validator."""
        # Check if it's in our discovered validators
        for validator_key, validator_info in self.discovered_validators.items():
            if validator_info.name == func_name:
                return True
        
        # Check if it looks like a validator (ends with _validator)
        return func_name.endswith('_validator')
    
    def _is_external_dependency(self, func_name: str) -> bool:
        """Check if this is an external library dependency."""
        external_patterns = [
            'validators.',  # validators library
            'pydantic.',    # pydantic functions
            're.',          # regex module
            'datetime.',    # datetime module
            'typing.',      # typing module
            'email_validator.',  # email-validator library
        ]
        return any(func_name.startswith(pattern) for pattern in external_patterns)
    
    def _classify_validator_complexity(self, validator_info: ValidatorInfo) -> str:
        """Classify validator complexity based on dependencies."""
        validator_name = validator_info.name
        
        if validator_name in self.complexity_cache:
            return self.complexity_cache[validator_name]
        
        if validator_name not in self.dependency_graph:
            self.complexity_cache[validator_name] = "SIMPLE"
            return "SIMPLE"
        
        dependencies = self.dependency_graph[validator_name]
        
        # Check for external dependencies
        if any(self._is_external_dependency(dep) for dep in dependencies):
            complexity = "EXTERNAL"
        
        # Check for circular dependencies
        elif self._has_circular_dependency(validator_name):
            complexity = "HIGH"
            
        # Check dependency count
        elif len(dependencies) == 0:
            complexity = "SIMPLE"
        elif len(dependencies) <= self.analysis_config.medium_max_dependencies:
            complexity = "MEDIUM"
        else:
            complexity = "HIGH"
        
        self.complexity_cache[validator_name] = complexity
        return complexity
    
    def _has_circular_dependency(self, validator_name: str, visited: Set[str] = None) -> bool:
        """Check if validator has circular dependencies."""
        if visited is None:
            visited = set()
            
        if validator_name in visited:
            return True  # Found cycle
            
        visited.add(validator_name)
        
        dependencies = self.dependency_graph.get(validator_name, [])
        for dep in dependencies:
            if self._has_circular_dependency(dep, visited.copy()):
                return True
                
        return False
    
    def _ai_convert_validator(self, field_name: str, validator_info: ValidatorInfo, complexity: str) -> ValidationRule:
        """Convert validator to validation rule based on complexity."""
        
        if complexity == "SIMPLE":
            return self._ai_convert_simple_validator(field_name, validator_info)
        elif complexity == "MEDIUM":
            return self._ai_convert_with_dependencies(field_name, validator_info)
        elif complexity == "HIGH":
            return self._ai_convert_complex_validator(field_name, validator_info)
        else:  # EXTERNAL
            return self._ai_convert_external_validator(field_name, validator_info)
    
    def _ai_convert_simple_validator(self, field_name: str, validator_info: ValidatorInfo) -> ValidationRule:
        """Convert simple validator with no dependencies using AI."""
        try:
            # Get the validator source code
            if validator_info.function_def:
                source_code = ast.unparse(validator_info.function_def)
            else:
                source_code = f"# Function definition not available for {validator_info.name}"
            
            # Create AI prompt using standardized template
            prompt = AIPromptTemplates.field_validator_conversion(field_name, source_code)
            
            # Use AI to convert (placeholder for actual ai_query implementation)
            sql_condition = self._execute_ai_query(prompt, validator_info)
            
            # Format description with SQL indicator if condition is valid SQL
            base_description = f"AI-converted validation from {validator_info.name}"
            formatted_description = self._format_validation_description(
                base_description, sql_condition, validator_info.name, "custom_validator"
            )
            
            return ValidationRule(
                field_name=field_name,
                rule_type="custom_validator",
                condition=sql_condition,
                description=formatted_description,
                action="drop"
            )
            
        except Exception as e:
            if self.analysis_config.ai_fallback_to_basic:
                return self._basic_validator_fallback(field_name, validator_info)
            else:
                raise e
    
    def _ai_convert_with_dependencies(self, field_name: str, validator_info: ValidatorInfo) -> ValidationRule:
        """Convert validator with dependencies by providing full context."""
        validator_name = validator_info.name
        dependencies = self.dependency_graph.get(validator_name, [])
        
        try:
            # Build complete context with all dependencies
            context_code = []
            for dep_name in dependencies:
                dep_validator = self._find_validator_by_name(dep_name)
                if dep_validator and dep_validator.function_def:
                    dep_source = ast.unparse(dep_validator.function_def)
                    context_code.append(f"# {dep_name}\n{dep_source}\n")
            
            main_source = ast.unparse(validator_info.function_def) if validator_info.function_def else f"# No source for {validator_name}"
            full_context = "\n".join(context_code)
            
            prompt = f"""
            Convert this Python validator function to equivalent SQL conditions for field '{field_name}'.
            
            Main validator function:
            {main_source}
            
            Dependent validator functions:
            {full_context}
            
            Generate SQL WHERE clause conditions that validate the same logic as the main function,
            taking into account all the dependent validation logic.
            Use Databricks SQL functions when possible.
            """
            
            sql_condition = self._execute_ai_query(prompt, validator_info)
            
            # Format description with SQL indicator if condition is valid SQL
            base_description = f"AI-converted validation from {validator_name} with dependencies: {dependencies}"
            formatted_description = self._format_validation_description(
                base_description, sql_condition, validator_name, "custom_validator"
            )
            
            return ValidationRule(
                field_name=field_name,
                rule_type="custom_validator",
                condition=sql_condition,
                description=formatted_description,
                action="drop"
            )
            
        except Exception as e:
            return self._basic_validator_fallback(field_name, validator_info, 
                                               f"Dependency resolution failed: {str(e)}")
    
    def _ai_convert_complex_validator(self, field_name: str, validator_info: ValidatorInfo) -> ValidationRule:
        """Handle complex validators with warnings."""
        validator_name = validator_info.name
        
        # Try AI conversion with warnings
        try:
            source_code = ast.unparse(validator_info.function_def) if validator_info.function_def else f"# No source for {validator_name}"
            
            prompt = f"""
            This is a complex Python validator with circular dependencies or high complexity.
            Convert to SQL if possible, otherwise suggest basic validation:
            
            {source_code}
            
            For field '{field_name}', provide SQL conditions or indicate if manual review is needed.
            Use Databricks SQL functions when possible.
            """
            
            sql_condition = self._execute_ai_query(prompt, validator_info)
            
            warning = "COMPLEX VALIDATOR: Manual review recommended"
            if self.analysis_config.include_manual_review_flags:
                warning += f". Circular dependencies or high complexity detected in {validator_name}"
            
            # Format description with SQL indicator if condition is valid SQL
            base_description = f"AI-converted (complex): {warning}"
            formatted_description = self._format_validation_description(
                base_description, sql_condition, validator_name, "custom_validator"
            )
            
            return ValidationRule(
                field_name=field_name,
                rule_type="custom_validator",
                condition=sql_condition,
                description=formatted_description,
                action="drop"
            )
            
        except Exception:
            # Fallback for complex cases
            return self._basic_validator_fallback(field_name, validator_info, 
                                               "MANUAL REVIEW NEEDED: Complex validator dependencies detected")
    
    def _ai_convert_external_validator(self, field_name: str, validator_info: ValidatorInfo) -> ValidationRule:
        """Handle validators with external dependencies."""
        warning = f"EXTERNAL DEPENDENCIES: {validator_info.name} uses external libraries"
        condition = f"typeof({field_name}) = 'string' AND {field_name} IS NOT NULL"
        
        # Format description with SQL indicator
        base_description = f"Basic validation only - {warning}"
        formatted_description = self._format_validation_description(
            base_description, condition, validator_info.name, "custom_validator"
        )
        
        return ValidationRule(
            field_name=field_name,
            rule_type="custom_validator", 
            condition=condition,
            description=formatted_description,
            action="drop"
        )
    
    def _ai_convert_custom_type(self, field_name: str, field_type: str) -> List[ValidationRule]:
        """Convert custom type using strict AI template to prevent verbose responses."""
        try:
            # Use strict AI prompt template
            from .ai_utils import AIPromptTemplates
            prompt = AIPromptTemplates.custom_type_conversion(field_name, field_type)
            
            sql_condition = self._execute_ai_query_for_type(prompt, field_type)
            
            # Format description with SQL indicator
            base_description = f"AI-converted validation for custom type {field_type}"
            formatted_description = self._format_validation_description(
                base_description, sql_condition
            )
            
            return [ValidationRule(
                field_name=field_name,
                rule_type="custom_type",
                condition=sql_condition,
                description=formatted_description,
                action="drop"
            )]
            
        except Exception as e:
            if self.analysis_config.ai_fallback_to_basic:
                return self._basic_custom_type_validation(field_name, field_type)
            else:
                raise
    
    def _basic_custom_type_validation(self, field_name: str, field_type: str) -> List[ValidationRule]:
        """Generate basic fallback validation for custom types."""
        # Basic validation based on common patterns
        if "email" in field_type.lower():
            condition = f"{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$'"
            description = f"{field_name} must be a valid email format"
        elif "zip" in field_type.lower():
            condition = f"{field_name} RLIKE '^\\d{{5}}(-\\d{{4}})?$'"
            description = f"{field_name} must be 5-digit or ZIP+4 format"
        elif "date" in field_type.lower():
            condition = f"typeof({field_name}) = 'date' OR TRY_CAST({field_name} AS DATE) IS NOT NULL"
            description = f"{field_name} must be a valid date"
        else:
            condition = f"typeof({field_name}) = 'string' AND trim(COALESCE({field_name}, '')) != ''"
            description = f"{field_name} must be a non-empty string (basic validation for {field_type})"
        
        # Format description with SQL indicator
        formatted_description = self._format_validation_description(description, condition)
        
        return [ValidationRule(
            field_name=field_name,
            rule_type="custom_type",
            condition=condition,
            description=formatted_description,
            action="drop"
        )]
    
    def _basic_validator_fallback(self, field_name: str, validator_info: ValidatorInfo, warning: str = None) -> ValidationRule:
        """Generate basic fallback validation rule."""
        base_condition = f"typeof({field_name}) = 'string' AND {field_name} IS NOT NULL"
        
        description = f"Basic type validation for {validator_info.name}"
        if warning:
            description = f"{warning}. {description}"
        
        # Format description with SQL indicator
        formatted_description = self._format_validation_description(
            description, base_condition, validator_info.name
        )
        
        return ValidationRule(
            field_name=field_name,
            rule_type="basic_type_check",
            condition=base_condition,
            description=formatted_description,
            action="drop"
        )
    
    def _find_validator_by_name(self, validator_name: str) -> Optional[ValidatorInfo]:
        """Find validator info by function name."""
        for validator_key, validator_info in self.discovered_validators.items():
            if validator_info.name == validator_name:
                return validator_info
        return None
    
    def _execute_ai_query(self, prompt: str, validator_info: ValidatorInfo) -> str:
        """Execute AI query for validator conversion using shared AI executor."""
        context_info = {'validator_info': validator_info}
        return self.ai_executor.execute_query(prompt, context_info)
    
    
    def _execute_ai_query_for_type(self, prompt: str, field_type: str) -> str:
        """Execute AI query for custom type conversion using shared AI executor."""
        context_info = {'field_type': field_type}
        return self.ai_executor.execute_query(prompt, context_info)
    
    def _generate_ai_response_for_type(self, prompt: str, field_type: str) -> str:
        """Generate AI response for custom type (without caching logic)."""
        # Check if we should use real AI query or mock responses
        if self.config.validation.ai.use_mock_responses:
            # Development/demo mode: use mock responses
            self.logger.info(f"Using mock AI response for custom type {field_type} (demo mode)")
            return self._mock_ai_response_for_type(field_type)
        else:
            # Production mode: use real Databricks ai_query()
            try:
                self.logger.info(f"Calling real ai_query for custom type {field_type} using model {self.config.validation.ai.model}")
                
                # ai_query is a SQL function in Databricks, not a PySpark function
                # Use spark.sql() to call the SQL ai_query function
                from pyspark.sql import SparkSession
                spark = SparkSession.getActiveSession()
                
                if spark is None:
                    raise ImportError("No active SparkSession - not in Databricks environment")
                
                # Escape the prompt for SQL
                escaped_prompt = prompt.replace("'", "''")  # Escape single quotes for SQL
                
                sql_query = f"""
                SELECT ai_query(
                    '{self.config.validation.ai.model}', 
                    '{escaped_prompt}'
                ) as ai_response
                """
                
                result_df = spark.sql(sql_query)
                ai_result = result_df.collect()[0]['ai_response']
                
                return ai_result
                
            except ImportError:
                self.logger.warning("ai_query not available (not in Databricks environment). Falling back to mock response.")
                if self.config.validation.ai.fallback_enabled:
                    return self._mock_ai_response_for_type(field_type)
                else:
                    raise Exception("ai_query not available and fallback is disabled")
            except Exception as e:
                self.logger.error(f"Real AI query failed for custom type {field_type}: {e}")
                if self.config.validation.ai.fallback_enabled:
                    self.logger.info("Falling back to mock response due to AI query failure")
                    return self._mock_ai_response_for_type(field_type)
                else:
                    raise
    
    def _deduplicate_validation_rules(self, rules: List[ValidationRule]) -> List[ValidationRule]:
        """Remove duplicate validation rules based on field name and condition similarity."""
        if not rules:
            return rules
        
        # Group rules by field name for efficient comparison
        rules_by_field = {}
        for rule in rules:
            if rule.field_name not in rules_by_field:
                rules_by_field[rule.field_name] = []
            rules_by_field[rule.field_name].append(rule)
        
        deduplicated_rules = []
        
        for field_name, field_rules in rules_by_field.items():
            # Keep track of unique conditions and rule types for this field
            unique_rules = []
            seen_conditions = set()
            seen_rule_signatures = set()
            
            for rule in field_rules:
                # Create a signature for this rule (field + type + normalized condition)
                normalized_condition = self._normalize_sql_condition(rule.condition)
                rule_signature = f"{rule.field_name}:{rule.rule_type}:{normalized_condition}"
                
                # Check for exact duplicates
                if rule_signature in seen_rule_signatures:
                    self.logger.debug(f"Skipping duplicate rule: {rule_signature}")
                    continue
                
                # Special handling for reference rules - always keep them
                if rule.action == "reference" or "see validator:" in rule.condition:
                    unique_rules.append(rule)
                    seen_rule_signatures.add(rule_signature)
                    continue
                
                # Check for similar conditions (to catch AI vs hardcoded variations)
                # But skip this check if we already have a reference rule for this field
                has_reference_rule = any(r.action == "reference" or "see validator:" in r.condition 
                                       for r in unique_rules if r.field_name == rule.field_name)
                
                is_duplicate = False
                if not has_reference_rule:
                    for seen_condition in seen_conditions:
                        if self._are_conditions_similar(normalized_condition, seen_condition, rule.field_name):
                            self.logger.debug(f"Skipping similar condition for {field_name}: {rule.condition}")
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_rules.append(rule)
                    seen_conditions.add(normalized_condition)
                    seen_rule_signatures.add(rule_signature)
            
            deduplicated_rules.extend(unique_rules)
        
        return deduplicated_rules
    
    def _normalize_sql_condition(self, condition: str) -> str:
        """Normalize SQL condition for comparison by removing whitespace and standardizing format."""
        if not condition:
            return ""
        
        # Basic normalization: remove extra whitespace, convert to lowercase for comparison
        normalized = ' '.join(condition.strip().split())
        normalized = normalized.lower()
        
        # Standardize common patterns
        normalized = normalized.replace(" is not null", " IS NOT NULL")
        normalized = normalized.replace(" is null", " IS NULL") 
        normalized = normalized.replace(" in (", " IN (")
        normalized = normalized.replace("typeof(", "TYPEOF(")
        normalized = normalized.replace("try_cast(", "TRY_CAST(")
        
        return normalized
    
    def _are_conditions_similar(self, condition1: str, condition2: str, field_name: str) -> bool:
        """Check if two SQL conditions are functionally similar."""
        if condition1 == condition2:
            return True
        
        # Check for common patterns that might be duplicated
        
        # Both are email validation patterns
        if (field_name == "email" or "email" in field_name.lower()) and \
           ("rlike" in condition1 and "rlike" in condition2) and \
           ("@" in condition1 and "@" in condition2):
            return True
        
        # Both are zipcode validation patterns  
        if (field_name == "zip" or "zip" in field_name.lower()) and \
           ("rlike" in condition1 and "rlike" in condition2) and \
           ("\\d{5}" in condition1 and "\\d{5}" in condition2):
            return True
        
        # Both are the same basic type check
        if f"typeof({field_name})" in condition1 and f"typeof({field_name})" in condition2:
            return True
        
        # Both are NOT NULL checks for the same field
        if f"{field_name} is not null" in condition1 and f"{field_name} is not null" in condition2:
            return True
        
        return False
    
    def _mock_ai_response_for_validator(self, validator_info: ValidatorInfo) -> str:
        """Mock AI response for validator conversion (replace with actual ai_query)."""
        validator_name = validator_info.name.lower()
        
        # Handle complex model validators with proper |||DESCRIPTION format
        if "customer_contact_completeness" in validator_name:
            return """(phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0) OR (email IS NOT NULL AND LENGTH(TRIM(email)) > 0)|||Customer must have either phone or email contact
(email IS NOT NULL AND email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$') OR (phone IS NOT NULL AND LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10)|||At least one valid contact method required"""
        
        elif "address_completeness" in validator_name:
            return """street IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL AND zip IS NOT NULL|||Complete address required for all customers
LENGTH(TRIM(street)) >= 5 AND LENGTH(TRIM(city)) >= 2 AND LENGTH(state) = 2 AND zip RLIKE '^[0-9]{5}(-[0-9]{4})?$'|||Valid address components with proper formatting"""
        
        elif "contact_method" in validator_name:
            return """(email IS NOT NULL) OR (phone IS NOT NULL) OR (street IS NOT NULL AND city IS NOT NULL)|||At least one contact method must be available
CASE WHEN email IS NOT NULL THEN email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$' ELSE TRUE END|||Email must be valid if provided"""
        
        elif "geographic_service" in validator_name:
            return """city IS NOT NULL AND state IS NOT NULL AND zip IS NOT NULL|||Geographic location must be complete
state IN ('AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY')|||State must be valid US state"""
        
        elif "city_state_consistency" in validator_name:
            return """city IS NOT NULL AND state IS NOT NULL|||City and state must both be provided for consistency validation
LENGTH(city) >= 2 AND LENGTH(state) = 2|||City and state must have reasonable lengths"""
        
        elif "phone_format" in validator_name:
            return """phone IS NULL OR LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10|||Phone must be 10 digits when provided
phone IS NULL OR LEFT(REGEXP_REPLACE(phone, '[^0-9]', ''), 1) NOT IN ('0', '1')|||Valid area code when phone provided"""
        
        # Handle simple field validators
        elif "email" in validator_name:
            return f"email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$' AND LENGTH(email) <= 254"
        elif "zipcode" in validator_name or "zip" in validator_name:
            return f"zip RLIKE '^\\d{{5}}(-\\d{{4}})?$'"
        elif "sex" in validator_name:
            return f"sex IN ('M', 'F')"
        elif "date" in validator_name and "past" in validator_name:
            return f"to_date(signup_date, 'yyyy-MM-dd') <= current_date()"
        elif "name" in validator_name:
            return f"length(name) >= 1 AND length(name) <= 80 AND name RLIKE '^[a-zA-Z\\s]+$'"
        else:
            field_name = validator_name.replace('_validator', '')
            return f"typeof({field_name}) = 'string' AND {field_name} IS NOT NULL"
    
    def _mock_ai_response_for_type(self, field_type: str) -> str:
        """Clean mock AI response for custom type conversion using strict format."""
        # Use the new strict mock responses
        from .ai_utils import AIResponseMocks
        conditions_list = AIResponseMocks.for_field_type(field_type.lower())
        
        # Return the first condition for simple single-condition cases
        if conditions_list and len(conditions_list) > 0:
            return conditions_list[0]['condition']
        else:
            return f"{field_type.lower()} IS NOT NULL"
    
    # ===============================================================================
    # COMPREHENSIVE AI-ENHANCED PROCESSING (Restored from Generator)
    # ===============================================================================
    
    def _generate_comprehensive_ai_rules(self, model_info: ModelInfo, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """
        Generate comprehensive custom validator rules for custom types.
        
        This restores the comprehensive processing that Generator was doing:
        - Custom type field processing (['Email', 'ZipCode', 'Sex', 'PastDate', 'Name'])
        - Full validator source code reading
        - Rich AI prompt generation  
        - Multi-condition SQL generation
        - Creates multiple 'custom_validator' rules per field
        """
        if not validators:
            return []
        
        self.logger.info(f"Generating comprehensive custom validator rules for {model_info.name}")
        
        # Custom field types that get comprehensive AI processing
        custom_field_types = ['Email', 'ZipCode', 'Sex', 'PastDate', 'Name']
        enhanced_rules = []
        
        for field_name, field_info in model_info.fields.items():
            field_type = field_info.get('type', '')
            
            if field_type in custom_field_types:
                # Find matching validator using the same logic Generator used
                validator_mapping = {
                    'Email': 'email_validator',
                    'ZipCode': 'zipcode_validator', 
                    'Sex': 'sex_validator',
                    'PastDate': 'past_date_validator',
                    'Name': 'name_validator'
                }
                
                expected_validator = validator_mapping.get(field_type)
                if expected_validator:
                    # Find validator info
                    validator_info = None
                    for v_name, v_info in validators.items():
                        if v_info.name == expected_validator:
                            validator_info = v_info
                            break
                    
                    if validator_info:
                        self.logger.info(f"Processing comprehensive validation for {field_name} ({field_type})")
                        
                        # Use strict AI processing with new templates
                        ai_conditions = self._get_strict_custom_validator_conditions(field_name, field_type, validator_info)
                        
                        # Create multiple ValidationRule objects (like Generator did)
                        for i, condition_info in enumerate(ai_conditions):
                            rule = ValidationRule(
                                field_name=field_name,
                                rule_type='custom_validator',  # All custom validator rules go here
                                condition=condition_info['condition'],
                                description=condition_info['description'],
                                action='drop'
                            )
                            enhanced_rules.append(rule)
                            self.logger.debug(f"Added custom validator rule: {condition_info['description']}")
        
        if enhanced_rules:
            self.logger.info(f"Generated {len(enhanced_rules)} comprehensive custom validator rules")
        else:
            self.logger.info("No custom field types found for comprehensive validation processing")
            
        return enhanced_rules
    
    def _get_strict_custom_validator_conditions(self, field_name: str, field_type: str, validator_info: ValidatorInfo) -> List[Dict[str, str]]:
        """
        Get strict custom validator conditions using new AI templates and executor.
        This ensures clean, non-verbose responses.
        """
        try:
            # Read validator source code
            from pathlib import Path
            import ast
            
            validator_file = Path(validator_info.file_path)
            validator_code = ""
            
            if validator_file.exists():
                with open(validator_file, 'r') as f:
                    source_code = f.read()
                
                # Extract specific validator function
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == validator_info.name:
                        validator_code = ast.get_source_segment(source_code, node)
                        break
            
            if not validator_code:
                # Fall back to clean mock responses
                return self._get_clean_fallback_conditions(field_name, field_type)
            
            # Use strict AI prompt template
            from .ai_utils import AIPromptTemplates
            prompt = AIPromptTemplates.custom_validator_conversion(field_name, field_type, validator_code)
            
            # Use the AI executor with strict context
            context_info = {
                'field_name': field_name,
                'field_type': field_type,
                'validator_info': validator_info
            }
            
            response = self.ai_executor.execute_query(prompt, context_info)
            
            # Parse response using strict parser
            from .ai_utils import AIResponseParser
            conditions = AIResponseParser.parse_json_conditions(response)
            
            if conditions:
                return conditions
            else:
                # Fallback to clean conditions if parsing fails
                return self._get_clean_fallback_conditions(field_name, field_type)
                
        except Exception as e:
            self.logger.warning(f"Error in strict validator processing for {field_name}: {e}")
            return self._get_clean_fallback_conditions(field_name, field_type)
    
    def _get_clean_fallback_conditions(self, field_name: str, field_type: str) -> List[Dict[str, str]]:
        """
        Generate clean, non-verbose fallback conditions for custom types.
        These follow the strict format requirements.
        """
        # Use clean fallback conditions from AIResponseMocks
        from .ai_utils import AIResponseMocks
        return AIResponseMocks.for_field_type(field_type.lower())
    
    def _process_field_validators(self, model_info: ModelInfo, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """
        Process @field_validator decorators to generate field-specific validation rules.
        
        This handles field-level validators that operate on individual field values,
        providing additional validation beyond type-based rules.
        """
        field_rules = []
        
        self.logger.info(f"Processing {len(model_info.field_validators)} field validators for {model_info.name}")
        
        for field_name, validator_method_names in model_info.field_validators.items():
            for validator_method_name in validator_method_names:
                self.logger.info(f"Processing field validator: {field_name} -> {validator_method_name}")
                
                # Extract validation logic from the method AST
                validator_rule = self._convert_field_validator_to_rule(
                    field_name, validator_method_name, model_info
                )
                
                if validator_rule:
                    field_rules.append(validator_rule)
        
        self.logger.info(f"Generated {len(field_rules)} field validator rules")
        return field_rules
    
    def _convert_field_validator_to_rule(self, field_name: str, validator_method_name: str, model_info: ModelInfo) -> Optional[ValidationRule]:
        """
        Convert a @field_validator decorated method to a ValidationRule.
        
        This analyzes the field validator method and generates appropriate SQL conditions.
        """
        try:
            # Find the validator method in the class definition
            validator_method = None
            for node in model_info.class_def.body:
                if isinstance(node, ast.FunctionDef) and node.name == validator_method_name:
                    validator_method = node
                    break
            
            if not validator_method:
                self.logger.warning(f"Field validator method {validator_method_name} not found in AST")
                return None
            
            # Analyze the validator method to extract validation logic
            if self.analysis_config.ai_conversion_enabled:
                # Use AI to convert the Python validation logic to SQL
                sql_condition = self._ai_convert_field_validator(field_name, validator_method, model_info)
            else:
                # Fallback: create a basic validation rule
                sql_condition = f"/* Field validation: {validator_method_name} */"
            
            # Create ValidationRule
            rule = ValidationRule(
                field_name=field_name,
                rule_type="field_validator",
                condition=sql_condition,
                description=self._format_validation_description(sql_condition, f"Field validation from @field_validator: {validator_method_name}"),
                action="drop"
            )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error converting field validator {validator_method_name}: {e}")
            return None
    
    def _ai_convert_field_validator(self, field_name: str, validator_method: ast.FunctionDef, model_info: ModelInfo) -> str:
        """
        Use AI to convert a field validator method to SQL condition.
        """
        try:
            # Extract the method source code
            # For now, create a basic SQL condition (AI conversion can be enhanced later)
            method_name = validator_method.name
            
            # Generate basic SQL based on common validation patterns
            if "positive" in method_name.lower():
                return f"{field_name} > 0"
            elif "format" in method_name.lower():
                if field_name == "phone":
                    return f"{field_name} IS NULL OR LENGTH(REGEXP_REPLACE({field_name}, '[^0-9]', '')) = 10"
                elif field_name == "street":
                    return f"{field_name} IS NOT NULL AND LENGTH(TRIM({field_name})) >= 5"
                else:
                    return f"{field_name} IS NOT NULL AND LENGTH(TRIM({field_name})) > 0"
            elif "state" in method_name.lower() and "code" in method_name.lower():
                return f"{field_name} IS NOT NULL AND LENGTH({field_name}) = 2 AND {field_name} RLIKE '^[A-Z]{{2}}$'"
            else:
                # Generic validation
                return f"{field_name} IS NOT NULL"
                
        except Exception as e:
            self.logger.error(f"Error in AI conversion for field validator: {e}")
            return f"/* Field validation: {validator_method.name} */"
    
    # Removed _analyze_model_validators_from_decorators - was causing duplicate processing
    # All model validator processing is now handled by _process_model_validators
    
    def _process_single_model_validator(self, validator_method_name: str, model_info: ModelInfo, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """
        Process a single @model_validator decorated method.
        """
        # Map model validator method names to corresponding validator functions
        validator_function_mapping = {
            'validate_contact_completeness': 'customer_contact_completeness_validator',
            'validate_address_completeness': 'address_completeness_validator',
            'validate_contact_method': 'contact_method_validator',
            'validate_geographic_service': 'geographic_service_validator',
            'validate_city_state_consistency': 'city_state_consistency_validator',
            'validate_phone_format_model': 'phone_format_validator'  # Fixed: was validate_phone_format
        }
        
        validator_function_name = validator_function_mapping.get(validator_method_name)
        
        if validator_function_name:
            # Find the validator function in our discovered validators
            validator_info = None
            for v_key, v_info in validators.items():
                if v_info.name == validator_function_name:
                    validator_info = v_info
                    break
            
            if validator_info:
                self.logger.info(f"Processing model validator: {validator_method_name} -> {validator_function_name}")
                
                # Convert complex validator logic to SQL rules
                if self.analysis_config.ai_conversion_enabled:
                    sql_rules = self._convert_model_validator_to_sql(validator_method_name, validator_info, model_info)
                    return sql_rules
                else:
                    # Fallback: create a basic placeholder rule
                    basic_rule = ValidationRule(
                        field_name="model_level",
                        rule_type="model_validator",
                        condition=f"/* {validator_function_name} validation */",
                        description=f"Model-level validation: {validator_method_name}",
                        action="drop"
                    )
                    return [basic_rule]
            else:
                self.logger.warning(f"Validator function {validator_function_name} not found for model validator {validator_method_name}")
        else:
            self.logger.warning(f"No mapping found for model validator: {validator_method_name}")
        
        return []
    
    def _process_model_validators(self, model_info: ModelInfo, validators: Dict[str, ValidatorInfo]) -> List[ValidationRule]:
        """
        Process @model_validator decorators to generate cross-field validation rules.
        
        This handles model-level validators that operate on the entire model instance
        rather than individual fields, enabling complex business logic validation.
        """
        model_rules = []
        
        self.logger.info(f"Processing {len(model_info.model_validators)} model validators for {model_info.name}")
        
        # Map model validator method names to corresponding validator functions
        validator_function_mapping = {
            'validate_contact_completeness': 'customer_contact_completeness_validator',
            'validate_address_completeness': 'address_completeness_validator',
            'validate_contact_method': 'contact_method_validator',
            'validate_geographic_service': 'geographic_service_validator',
            'validate_city_state_consistency': 'city_state_consistency_validator',
            'validate_phone_format_model': 'phone_format_validator'  # Note: this should be validate_phone_format_model, not validate_phone_format
        }
        
        for model_validator_name in model_info.model_validators:
            # Find the corresponding validator function
            validator_function_name = validator_function_mapping.get(model_validator_name)
            
            if validator_function_name:
                # Find the validator function in our discovered validators
                validator_info = None
                for v_key, v_info in validators.items():
                    if v_info.name == validator_function_name:
                        validator_info = v_info
                        break
                
                if validator_info:
                    self.logger.info(f"Processing model validator: {model_validator_name} -> {validator_function_name}")
                    
                    # Convert complex validator logic to SQL rules
                    if self.analysis_config.ai_conversion_enabled:
                        sql_rules = self._convert_model_validator_to_sql(model_validator_name, validator_info, model_info)
                        model_rules.extend(sql_rules)
                    else:
                        # Fallback: create a simple SQL placeholder that won't get filtered out
                        basic_rule = ValidationRule(
                            field_name="model_level",
                            rule_type="model_validator",
                            condition="1 = 1",  # Valid SQL that always passes
                            description=f"Model-level validation: {model_validator_name} (AI disabled - using placeholder)",
                            action="drop"
                        )
                        model_rules.append(basic_rule)
                else:
                    self.logger.warning(f"Validator function {validator_function_name} not found for model validator {model_validator_name}")
            else:
                self.logger.warning(f"No mapping found for model validator: {model_validator_name}")
        
        # Apply global deduplication across all model validator rules
        deduplicated_rules = self._deduplicate_model_validator_rules(model_rules)
        
        self.logger.info(f"Generated {len(deduplicated_rules)} model-level validation rules (after deduplication)")
        return deduplicated_rules
    
    def _deduplicate_model_validator_rules(self, model_rules: List[ValidationRule]) -> List[ValidationRule]:
        """
        Deduplicate model validator rules based on condition content and ensure unique expectation names.
        """
        seen_conditions = set()
        unique_rules = []
        import hashlib
        
        for rule in model_rules:
            # Create a unique key based on the condition content (ignoring whitespace differences)
            condition_key = re.sub(r'\s+', ' ', rule.condition.strip().lower())
            
            if condition_key not in seen_conditions:
                seen_conditions.add(condition_key)
                
                # Ensure truly unique expectation names using content hash
                if not rule._unique_expectation_name:  # Only set if not already set
                    condition_hash = hashlib.md5(rule.condition.encode()).hexdigest()[:8]
                    rule._unique_expectation_name = f"model_validator_{rule.field_name}_{condition_hash}".lower()
                
                unique_rules.append(rule)
            # If condition is duplicate, skip this rule
        
        return unique_rules
    
    def _ensure_globally_unique_expectation_names(self, rules: List[ValidationRule]) -> List[ValidationRule]:
        """
        Ensure all expectation names are globally unique to prevent dictionary key conflicts.
        
        This addresses the critical issue where duplicate keys in @dlt.expect_all_or_drop({...})
        cause validation rules to be silently overwritten.
        
        Args:
            rules: List of validation rules that may have duplicate expectation names
            
        Returns:
            List of rules with guaranteed unique expectation names using content-based hashes
        """
        import hashlib
        
        seen_expectation_names = set()
        unique_rules = []
        name_collision_count = 0
        
        for rule in rules:
            original_name = rule.expectation_name
            
            # If this expectation name is already used, create a unique hash-based name
            if original_name in seen_expectation_names:
                name_collision_count += 1
                # Use rule content (condition + description) to generate unique hash
                content_for_hash = f"{rule.condition}_{rule.description}_{rule.field_name}"
                content_hash = hashlib.md5(content_for_hash.encode()).hexdigest()[:8]
                
                # Create new unique name: rule_type_field_hash
                unique_name = f"{rule.rule_type}_{rule.field_name}_{content_hash}".lower()
                
                # Override the expectation name
                rule._unique_expectation_name = unique_name
                
                self.logger.debug(f"Resolved naming conflict: '{original_name}' -> '{unique_name}'")
            
            # Track this name as used
            seen_expectation_names.add(rule.expectation_name)
            unique_rules.append(rule)
        
        if name_collision_count > 0:
            self.logger.info(f"Resolved {name_collision_count} expectation name conflicts using content-based hashes")
        
        return unique_rules
    
    def _clean_all_rule_content(self, rules: List[ValidationRule]) -> List[ValidationRule]:
        """
        SAFETY NET: Clean all rule conditions and descriptions to eliminate verbose patterns.
        
        This method acts as a final filter to ensure no verbose responses slip through,
        regardless of their source (AI responses, fallbacks, cached data, etc.).
        
        Rules with invalid SQL conditions are completely removed.
        """
        from .ai_utils import AIResponseParser
        
        cleaned_rules = []
        verbose_patterns_cleaned = 0
        invalid_sql_removed = 0
        
        for rule in rules:
            original_condition = rule.condition
            original_description = rule.description
            
            # Use aggressive SQL cleaning for conditions
            cleaned_condition = AIResponseParser.clean_sql_condition(original_condition)
            
            # Use general verbose cleaning for descriptions
            cleaned_description = AIResponseParser.clean_verbose_content(original_description)
            
            # Log what was cleaned
            if cleaned_condition != original_condition:
                verbose_patterns_cleaned += 1
                self.logger.debug(f"Cleaned condition for {rule.field_name} {rule.rule_type}: '{original_condition[:50]}...' -> '{cleaned_condition[:50]}...'")
            
            if cleaned_description != original_description:
                self.logger.debug(f"Cleaned description for {rule.field_name} {rule.rule_type}")
            
            # Include rules with valid SQL conditions OR legitimate model validator placeholders
            should_include = False
            final_condition = cleaned_condition
            
            if cleaned_condition and cleaned_condition.strip():
                # Valid SQL condition
                should_include = True
            elif (rule.rule_type == 'model_validator' and 
                  original_condition.startswith('/*') and 'validation:' in original_condition):
                # Special case: preserve model validator placeholder comments
                should_include = True
                final_condition = original_condition  # Keep original comment
                self.logger.debug(f"Preserving model validator placeholder: {rule.field_name}")
            
            if should_include:
                # Additional validation - skip "see validator:" references
                if not final_condition.strip().lower().startswith('see validator'):
                    cleaned_rule = ValidationRule(
                        field_name=rule.field_name,
                        rule_type=rule.rule_type,
                        condition=final_condition,
                        description=cleaned_description,
                        action=rule.action,
                        _unique_expectation_name=rule._unique_expectation_name
                    )
                    cleaned_rules.append(cleaned_rule)
                else:
                    self.logger.debug(f"Skipped validator reference rule: {rule.field_name} {rule.rule_type}")
            else:
                invalid_sql_removed += 1
                self.logger.debug(f"Removed rule with invalid/empty SQL: {rule.field_name} {rule.rule_type} - '{original_condition[:50]}...'")
        
        # Summary logging
        if verbose_patterns_cleaned > 0:
            self.logger.info(f"SAFETY NET: Cleaned verbose patterns from {verbose_patterns_cleaned} rules")
        
        if invalid_sql_removed > 0:
            self.logger.info(f"SAFETY NET: Removed {invalid_sql_removed} rules with invalid SQL conditions")
        
        total_filtered = len(rules) - len(cleaned_rules)
        if total_filtered > 0:
            self.logger.info(f"SAFETY NET: Filtered out {total_filtered} invalid rules ({len(cleaned_rules)} valid rules remain)")
        
        return cleaned_rules
    
    def _clean_verbose_content(self, content: str) -> str:
        """
        Clean verbose patterns from rule content (conditions and descriptions).
        
        This removes:
        - WHERE clause headers
        - Code block markers
        - SQL comments
        - Verbose explanations
        - Multiple sections/alternatives
        """
        if not content or not isinstance(content, str):
            return content
            
        # Use the existing AIResponseParser cleanup method
        from .ai_utils import AIResponseParser
        cleaned = AIResponseParser.clean_sql_condition(content)
        
        # Additional cleanup for descriptions and complex patterns
        lines = cleaned.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip comment lines
            if line.startswith('--') or line.startswith('#'):
                continue
                
            # Skip explanation sections
            if any(pattern in line.lower() for pattern in [
                'explanation:', 'alternative', 'simplified version', 'more performant',
                'more concise version', 'databricks-specific', 'recommended combined'
            ]):
                continue
                
            # Skip WHERE clause headers
            if line.upper().startswith('WHERE'):
                continue
                
            # Skip markdown headers
            if line.startswith('#') or line.startswith('##'):
                continue
                
            # Clean individual lines
            if line.startswith('AND '):
                line = line[4:].strip()
            elif line.startswith('OR '):
                line = line[3:].strip()
                
            # Add non-empty cleaned lines
            if line:
                clean_lines.append(line)
        
        # Join lines and final cleanup
        result = ' '.join(clean_lines) if clean_lines else content
        
        # Remove any remaining code block markers
        result = result.replace('```sql', '').replace('```', '').strip()
        
        # If result is too short or generic, keep original but remove code blocks
        if len(result) < 10:
            result = content.replace('```sql', '').replace('```', '').strip()
        
        return result
    
    def _convert_model_validator_to_sql(self, model_validator_name: str, validator_info: ValidatorInfo, model_info: ModelInfo) -> List[ValidationRule]:
        """
        Convert a model-level validator function to SQL validation rules using AI.
        
        This analyzes the complex business logic in the validator function and
        generates multiple SQL conditions for DLT expectations.
        """
        try:
            # Read the validator function source code
            import ast
            from pathlib import Path
            
            validator_file = Path(validator_info.file_path)
            if not validator_file.exists():
                self.logger.error(f"Validator file not found: {validator_file}")
                return []
            
            with open(validator_file, 'r') as f:
                source_code = f.read()
            
            # Extract the specific validator function
            tree = ast.parse(source_code)
            validator_code = ""
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == validator_info.name:
                    validator_code = ast.get_source_segment(source_code, node)
                    break
            
            if not validator_code:
                self.logger.error(f"Could not extract source code for {validator_info.name}")
                return []
            
            # Create comprehensive AI prompt for cross-field validation
            model_fields = list(model_info.fields.keys())
            
            # Create AI prompt using standardized template
            prompt = AIPromptTemplates.model_validator_conversion(validator_code, model_fields)

            # Use AI conversion for model validator (include validator type in context)
            context_info = {
                'validator_info': validator_info,
                'validator_type': 'model_validator'
            }
            ai_response = self.ai_executor.execute_query(prompt, context_info)
            
            # Parse AI response into validation rules
            sql_rules = []
            expectation_name_counts = defaultdict(int)  # Track name usage for uniqueness
            
            if ai_response:
                lines = ai_response.strip().split('\n')
                
                for line in lines:
                    if '|||' in line:
                        condition, description = line.split('|||', 1)
                        condition = condition.strip()
                        description = description.strip()
                        
                        if condition and description:
                            # Determine appropriate field name for the rule
                            # For model-level rules, use the primary field or "model_level"
                            field_name = "model_level"
                            if "email" in condition.lower() and "phone" in condition.lower():
                                field_name = "contact_info"
                            elif "address" in description.lower() or ("street" in condition and "city" in condition):
                                field_name = "address_fields"
                            elif "email" in condition.lower():
                                field_name = "email"
                            elif "phone" in condition.lower():
                                field_name = "phone"
                            
                            # Create unique expectation name for model validators to avoid duplicates
                            base_expectation_name = f"model_validator_{field_name}"
                            expectation_name_counts[base_expectation_name] += 1
                            
                            # Add suffix only if this is not the first occurrence
                            if expectation_name_counts[base_expectation_name] == 1:
                                unique_expectation_name = base_expectation_name
                            else:
                                unique_expectation_name = f"{base_expectation_name}_{expectation_name_counts[base_expectation_name]}"
                            
                            rule = ValidationRule(
                                field_name=field_name,
                                rule_type="model_validator",
                                condition=condition,
                                description=f"SQL: {description}",
                                action="drop"
                            )
                            
                            # Create truly unique expectation name using content hash
                            import hashlib
                            condition_hash = hashlib.md5(condition.encode()).hexdigest()[:8]
                            rule._unique_expectation_name = f"model_validator_{field_name}_{condition_hash}".lower()
                            sql_rules.append(rule)
            
            if not sql_rules:
                # Fallback rule if AI conversion fails  
                fallback_rule = ValidationRule(
                    field_name="model_level",
                    rule_type="model_validator", 
                    condition=f"/* Complex validation: {validator_info.name} */",
                    description=f"Complex model validation from {model_validator_name}",
                    action="drop"
                )
                # Set unique expectation name for fallback based on validator name
                fallback_rule._unique_expectation_name = f"model_validator_{validator_info.name}".lower().replace('_validator', '')
                sql_rules.append(fallback_rule)
            
            self.logger.info(f"Generated {len(sql_rules)} SQL rules from {validator_info.name}")
            return sql_rules
            
        except Exception as e:
            self.logger.error(f"Error converting model validator {validator_info.name}: {e}")
            # Return fallback rule on error with unique name
            error_rule = ValidationRule(
                field_name="model_level",
                rule_type="model_validator",
                condition=f"/* Error processing {validator_info.name} */",
                description=f"Model validator: {model_validator_name} (conversion failed)",
                action="drop"
            )
            error_rule._unique_expectation_name = f"model_validator_error_{validator_info.name}".lower().replace('_validator', '')
            return [error_rule]
    
    def _get_comprehensive_validator_sql(self, field_name: str, field_type: str, validator_info: ValidatorInfo) -> List[Dict[str, str]]:
        """
        Use comprehensive AI processing to convert validator logic to multiple SQL conditions.
        
        This restores the detailed processing that Generator's _get_custom_validator_sql was doing:
        - Read full Python source code from validator files
        - Create detailed AI prompts with complete context  
        - Request structured JSON responses
        - Generate multiple SQL conditions per validator
        """
        try:
            import inspect
            import ast
            from pathlib import Path
            
            # Read the validator source code (same as Generator was doing)
            validator_file = Path(validator_info.file_path)
            validator_code = ""
            
            if validator_file.exists():
                with open(validator_file, 'r') as f:
                    source_code = f.read()
                
                # Extract just the specific validator function
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == validator_info.name:
                        validator_code = ast.get_source_segment(source_code, node)
                        break
                
                if not validator_code:
                    self.logger.warning(f"Could not extract source code for {validator_info.name}")
                    return self._get_fallback_ai_conditions(field_name, field_type)
                
                # Use strict AI prompt template to prevent verbose responses
                from .ai_utils import AIPromptTemplates
                prompt = AIPromptTemplates.custom_validator_conversion(field_name, field_type, validator_code)
                
                # Use strict AI executor to prevent verbose responses
                context_info = {
                    'field_name': field_name,
                    'field_type': field_type,
                    'validator_info': validator_info
                }
                
                response = self.ai_executor.execute_query(prompt, context_info)
                
                # Parse response using strict parser
                from .ai_utils import AIResponseParser
                conditions = AIResponseParser.parse_json_conditions(response)
                
                if conditions:
                    return conditions
                else:
                    return self._get_fallback_ai_conditions(field_name, field_type)
                
            else:
                self.logger.warning(f"Validator file not found: {validator_file}")
                return self._get_fallback_ai_conditions(field_name, field_type)
                
        except Exception as e:
            self.logger.error(f"Error in comprehensive AI processing for {field_name}: {e}")
            return self._get_fallback_ai_conditions(field_name, field_type)
    
    def _execute_comprehensive_ai_query(self, prompt: str, field_name: str, field_type: str, validator_info: ValidatorInfo) -> List[Dict[str, str]]:
        """
        Execute comprehensive AI query using strict AI executor.
        This method is now deprecated in favor of direct AI executor usage.
        """
        # This method is kept for backward compatibility but should not be used
        # Direct AI executor usage is preferred to ensure strict templates are used
        return self._get_fallback_ai_conditions(field_name, field_type)
    
    def _get_fallback_ai_conditions(self, field_name: str, field_type: str) -> List[Dict[str, str]]:
        """
        Generate comprehensive fallback conditions that demonstrate what AI would return.
        
        This provides the same multi-condition output that Generator was producing.
        """
        if field_type == 'Email':
            return [
                {"condition": f"{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$'", 
                 "description": "Valid email format"},
                {"condition": f"LENGTH({field_name}) <= 254", 
                 "description": "Email length within RFC 5321 limits"},
                {"condition": f"LENGTH(SPLIT({field_name}, '@')[0]) <= 64", 
                 "description": "Local part within limits"},
                {"condition": f"{field_name} NOT RLIKE '@(example|test|invalid|localhost)\\.(com|org|net)'", 
                 "description": "Exclude test domains"}
            ]
        elif field_type == 'ZipCode':
            return [
                {"condition": f"{field_name} RLIKE '^[0-9]{{5}}(-[0-9]{{4}})?$'", 
                 "description": "Valid US ZIP code format"},
                {"condition": f"LENGTH({field_name}) IN (5, 10)", 
                 "description": "ZIP code length validation"}
            ]
        elif field_type == 'Sex':
            return [
                {"condition": f"{field_name} IN ('M', 'F')", 
                 "description": "Must be M or F"},
                {"condition": f"LENGTH({field_name}) = 1", 
                 "description": "Single character validation"}
            ]
        elif field_type == 'PastDate':
            return [
                {"condition": f"{field_name} <= CURRENT_DATE()", 
                 "description": "Date must be in the past"},
                {"condition": f"{field_name} >= DATE('1900-01-01')", 
                 "description": "Reasonable date range validation"}
            ]
        elif field_type == 'Name':
            return [
                {"condition": f"LENGTH(TRIM({field_name})) > 0", 
                 "description": "Name must not be empty"},
                {"condition": f"LENGTH({field_name}) <= 100", 
                 "description": "Reasonable name length limit"},
                {"condition": f"{field_name} RLIKE '^[a-zA-Z\\s\\-\\']+$'", 
                 "description": "Valid name characters only"}
            ]
        else:
            return [
                {"condition": f"{field_name} IS NOT NULL", 
                 "description": f"Basic validation for {field_type}"}
            ]

