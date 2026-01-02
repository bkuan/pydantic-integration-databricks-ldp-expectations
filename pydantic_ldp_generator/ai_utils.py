"""
AI Utilities for Pydantic LDP Generator

Centralized module for AI query execution, prompt templates, caching, and mock responses.
Eliminates duplication between analyzer.py and model_generator.py.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .config import Config
from .discovery import ValidatorInfo


@dataclass 
class AIPromptTemplate:
    """Template for generating AI prompts with standardized structure."""
    system_context: str
    task_description: str
    input_section: str
    output_requirements: str
    examples: str
    validation_notes: str = ""


class AIQueryExecutor:
    """Centralized AI query execution with caching and error handling."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def execute_query(self, prompt: str, context_info: Dict[str, Any]) -> str:
        """
        Execute AI query with caching support.
        
        Args:
            prompt: The formatted AI prompt
            context_info: Context for caching and fallback (field_name, field_type, validator_info, etc.)
            
        Returns:
            str: AI response or mock response
        """
        if not self.config.validation.ai.enabled:
            raise Exception("AI conversion is disabled in configuration")
            
        # Try cache first if enabled
        cached_response = self._try_get_cached_response(prompt, context_info)
        if cached_response is not None:
            return cached_response
            
        # Generate new response
        response = self._generate_ai_response(prompt, context_info)
        
        # Cache the response if enabled
        self._try_cache_response(prompt, context_info, response)
        
        return response
        
    def _try_get_cached_response(self, prompt: str, context_info: Dict[str, Any]) -> Optional[str]:
        """Try to get cached response."""
        if not self.config.validation.ai.enable_cache:
            return None
            
        try:
            from .ai_cache import get_ai_cache
            cache = get_ai_cache(self.config)
            
            # Create cache key based on context type
            cache_key = self._build_cache_key(context_info)
            cached_response = cache.get(prompt, self.config.validation.ai.model, cache_key)
            
            if cached_response is not None:
                context_desc = self._get_context_description(context_info)
                self.logger.debug(f"Cache hit for {context_desc}")
                return cached_response
                
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
            
        return None
        
    def _try_cache_response(self, prompt: str, context_info: Dict[str, Any], response: str):
        """Try to cache the response."""
        if not self.config.validation.ai.enable_cache or not response:
            return
            
        try:
            from .ai_cache import get_ai_cache
            cache = get_ai_cache(self.config)
            cache_key = self._build_cache_key(context_info)
            cache.put(prompt, self.config.validation.ai.model, response, cache_key)
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
            
    def _build_cache_key(self, context_info: Dict[str, Any]) -> str:
        """Build cache key from context information."""
        if 'validator_info' in context_info:
            # For validator-based queries
            validator_info = context_info['validator_info']
            return f"validator_{validator_info.name}"
        elif 'field_type' in context_info:
            # For field type queries
            return f"field_type_{context_info['field_type']}"
        else:
            # Generic fallback
            return f"generic_{hash(str(context_info))}"
            
    def _get_context_description(self, context_info: Dict[str, Any]) -> str:
        """Get human-readable context description for logging."""
        if 'validator_info' in context_info:
            return context_info['validator_info'].name
        elif 'field_name' in context_info and 'field_type' in context_info:
            return f"{context_info['field_name']} ({context_info['field_type']})"
        else:
            return "unknown context"
            
    def _generate_ai_response(self, prompt: str, context_info: Dict[str, Any]) -> str:
        """Generate AI response (without caching logic)."""
        if self.config.validation.ai.use_mock_responses:
            return self._generate_mock_response(context_info)
        else:
            return self._call_real_ai_query(prompt, context_info)
            
    def _call_real_ai_query(self, prompt: str, context_info: Dict[str, Any]) -> str:
        """Call real Databricks AI query."""
        try:
            context_desc = self._get_context_description(context_info)
            self.logger.info(f"Calling real ai_query for {context_desc} using model {self.config.validation.ai.model}")
            
            # ai_query is a SQL function in Databricks, use spark.sql()
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
                return self._generate_mock_response(context_info)
            else:
                raise Exception("ai_query not available and fallback is disabled")
        except Exception as e:
            context_desc = self._get_context_description(context_info)
            self.logger.error(f"Real AI query failed for {context_desc}: {e}")
            if self.config.validation.ai.fallback_enabled:
                self.logger.info("Falling back to mock response due to AI query failure")
                return self._generate_mock_response(context_info)
            else:
                raise
                
    def _generate_mock_response(self, context_info: Dict[str, Any]) -> str:
        """Generate mock response for testing/demo mode."""
        context_desc = self._get_context_description(context_info)
        self.logger.info(f"Using mock AI response for {context_desc} (demo mode)")
        
        if 'validator_info' in context_info:
            validator_info = context_info['validator_info']
            # Check if this is a model validator based on context
            if 'model_validator' in context_info.get('validator_type', '') or 'model_level' in context_desc.lower():
                return AIResponseMocks.for_model_validator(validator_info)
            else:
                return AIResponseMocks.for_validator(validator_info)
        elif 'field_type' in context_info:
            # Convert field_type list response to JSON string for consistency
            field_type_conditions = AIResponseMocks.for_field_type(context_info['field_type'])
            import json
            return json.dumps(field_type_conditions)
        else:
            return AIResponseMocks.generic_fallback()


class AIPromptTemplates:
    """Standardized AI prompt templates for different validation scenarios."""
    
    @staticmethod
    def field_validator_conversion(field_name: str, validator_code: str) -> str:
        """Template for converting field validators to SQL."""
        return f"""Convert Python validator to clean SQL condition for Databricks DLT.

PYTHON VALIDATOR:
{validator_code}

FIELD: {field_name}

OUTPUT REQUIREMENTS:
- Single clean SQL condition expressions only (no WHERE, no comments, no markdown)
- Must PASS for valid data (not fail conditions)
- Use exact field name: {field_name}
- Use Databricks SQL: RLIKE, LENGTH, SPLIT, etc.
- Can generate multiple SQL condition expressions if the validator has multiple checks

EXAMPLES OF CORRECT OUTPUT:
email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{{2,}}$'
LENGTH(name) BETWEEN 1 AND 80
customer_id > 0

EXAMPLES OF INCORRECT (DO NOT DO):
WHERE email RLIKE '...' (no WHERE)
```sql ... ``` (no code blocks)  
-- Check email (no comments)

Return ONLY the clean SQL condition."""

    @staticmethod
    def custom_validator_conversion(field_name: str, field_type: str, validator_code: str) -> str:
        """Template for converting custom validators to multiple SQL conditions."""
        return f"""Convert Python validator to clean SQL conditions for Databricks DLT expectations.

PYTHON VALIDATOR:
{validator_code}

FIELD: {field_name} ({field_type})

OUTPUT REQUIREMENTS:
- Single clean SQL condition expressions only (no WHERE, no comments, no markdown)
- Must PASS for valid data (not fail conditions)
- Use exact field name: {field_name}
- Use Databricks SQL: RLIKE, LENGTH, SPLIT, etc.
- Can generate multiple SQL condition expressions if the validator has multiple checks


STRICT OUTPUT SCHEMA:
[
  {{"condition": "clean_sql_condition_here", "description": "brief_description"}},
  {{"condition": "another_condition", "description": "another_brief_description"}}
]

EXAMPLES OF CORRECT CONDITIONS:
- email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{{2,}}$'
- LENGTH(email) <= 254
- customer_id > 0
- phone IS NULL OR LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10

EXAMPLES OF INCORRECT (DO NOT DO):
- WHERE email IS NOT NULL (no WHERE keyword)
- ```sql ... ``` (no code blocks)
- -- Check email format (no comments)
- Detailed explanations (keep descriptions brief)

Return ONLY the JSON array. No other text."""

    @staticmethod 
    def model_validator_conversion(validator_code: str, model_fields: List[str]) -> str:
        """Template for converting model validators to cross-field SQL conditions."""
        return f"""Convert Pydantic model validator to clean SQL conditions for Databricks DLT.

VALIDATOR FUNCTION:
{validator_code}

AVAILABLE FIELDS: {', '.join(model_fields)}

OUTPUT REQUIREMENTS:
- Single clean SQL condition expressions only (no WHERE, no comments, no markdown)
- Must PASS for valid data (not fail conditions)
- Use exact field names from available fields list: {', '.join(model_fields)}
- Use Databricks SQL: RLIKE, LENGTH, SPLIT, etc.
- Can generate multiple SQL condition expressions if the validator has multiple checks
- Format: CONDITION|||DESCRIPTION

EXAMPLES OF CORRECT OUTPUT:
(phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0) OR (email IS NOT NULL AND LENGTH(TRIM(email)) > 0)|||Customer must have either phone or email contact
street IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL AND zip IS NOT NULL|||Complete address required for all customers
state IN ('AL','AK','AZ','AR','CA')|||Valid US state code

EXAMPLES OF INCORRECT (DO NOT DO):
WHERE phone IS NOT NULL (no WHERE)
```sql phone IS NOT NULL ``` (no code blocks)
-- Check phone format (no comments)
Very detailed description of validation logic (keep descriptions brief)

Return only the condition lines in CONDITION|||DESCRIPTION format."""

    @staticmethod
    def custom_type_conversion(field_name: str, field_type: str) -> str:
        """Template for converting custom types to SQL validation conditions."""
        return f"""Convert custom Pydantic type to clean SQL condition for Databricks DLT.

FIELD: {field_name}
TYPE: {field_type}

OUTPUT REQUIREMENTS:
- Single clean SQL condition expressions only (no WHERE, no comments, no markdown)
- Must PASS for valid data (not fail conditions)
- Use exact field name: {field_name}
- Use Databricks SQL: RLIKE, LENGTH, SPLIT, etc.
- Can generate multiple SQL condition expressions if the validator has multiple checks

EXAMPLES OF CORRECT OUTPUT:
{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{{2,}}$'
LENGTH(TRIM({field_name})) BETWEEN 1 AND 100
{field_name} RLIKE '^[0-9]{{5}}(-[0-9]{{4}})?$'
{field_name} < CURRENT_DATE()

EXAMPLES OF INCORRECT (DO NOT DO):
WHERE {field_name} IS NOT NULL (no WHERE)
```sql {field_name} RLIKE '...' ``` (no code blocks)
-- Check {field_type} format (no comments)

Return ONLY the clean SQL condition."""


class AIResponseMocks:
    """Mock AI responses for testing and demo mode."""
    
    @staticmethod
    def for_validator(validator_info: ValidatorInfo) -> str:
        """Generate mock response for a specific validator."""
        validator_name = validator_info.name.lower()
        
        if 'email' in validator_name:
            return "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'"
        elif 'phone' in validator_name:
            return "phone IS NULL OR LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10"
        elif 'name' in validator_name:
            return "LENGTH(TRIM(name)) BETWEEN 1 AND 100 AND name RLIKE '^[A-Za-z\\s\\-\\.]+$'"
        elif 'zip' in validator_name or 'postal' in validator_name:
            return "zip RLIKE '^[0-9]{5}(-[0-9]{4})?$'"
        elif 'date' in validator_name:
            return "signup_date IS NOT NULL AND signup_date < CURRENT_DATE()"
        elif 'positive' in validator_name or 'id' in validator_name:
            return "customer_id > 0"
        
        # Complex model validators - return actual SQL conditions instead of comments
        elif 'contact_completeness' in validator_name or 'customer_contact' in validator_name:
            return "(email IS NOT NULL AND LENGTH(TRIM(email)) > 0) OR (phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0)"
        elif 'address_completeness' in validator_name:
            return "(street IS NULL AND city IS NULL AND state IS NULL AND zip IS NULL) OR (street IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL AND zip IS NOT NULL)"
        elif 'contact_method' in validator_name:
            return "(SUBSTRING(zip, 1, 2) NOT IN ('90', '10', '94')) OR (SUBSTRING(zip, 1, 2) IN ('90', '10', '94') AND email IS NOT NULL AND LENGTH(TRIM(email)) > 0 AND phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0)"
        elif 'geographic_service' in validator_name:
            return "(state NOT IN ('HI', 'AK')) OR (state IN ('HI', 'AK') AND phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0)"
        elif 'city_state_consistency' in validator_name:
            return "(LOWER(TRIM(city)) NOT IN ('los angeles', 'new york', 'miami', 'austin', 'seattle', 'chicago')) OR (LOWER(TRIM(city)) = 'los angeles' AND state = 'CA') OR (LOWER(TRIM(city)) = 'new york' AND state = 'NY') OR (LOWER(TRIM(city)) = 'miami' AND state = 'FL') OR (LOWER(TRIM(city)) = 'austin' AND state = 'TX') OR (LOWER(TRIM(city)) = 'seattle' AND state = 'WA') OR (LOWER(TRIM(city)) = 'chicago' AND state = 'IL')"
        elif 'phone_format' in validator_name:
            return "LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10 AND SUBSTRING(REGEXP_REPLACE(phone, '[^0-9]', ''), 1, 1) NOT IN ('0', '1')"
        else:
            # Last resort - return a generic but valid SQL condition
            return "1 = 1"
            
    @staticmethod
    def for_field_type(field_type: str) -> List[Dict[str, str]]:
        """Generate clean, simple mock response for field type validation."""
        field_type = field_type.lower()
        
        if field_type == 'email':
            return [
                {"condition": "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'", 
                 "description": "Valid email format"},
                {"condition": "LENGTH(email) <= 254", 
                 "description": "Email length limit"}
            ]
        elif field_type == 'name':
            return [
                {"condition": "LENGTH(TRIM(name)) BETWEEN 1 AND 100", 
                 "description": "Name length validation"}
            ]
        elif field_type == 'zipcode':
            return [
                {"condition": "zip RLIKE '^[0-9]{5}(-[0-9]{4})?$'", 
                 "description": "Valid US ZIP code format"}
            ]
        else:
            return [
                {"condition": f"{field_type.lower()} IS NOT NULL", 
                 "description": f"Basic {field_type} validation"}
            ]
            
    @staticmethod
    def for_model_validator(validator_info: ValidatorInfo) -> str:
        """Generate mock response for model-level validators using line-by-line format expected by analyzer."""
        validator_name = validator_info.name.lower()
        
        # Return line-by-line format with ||| delimiter as expected by _convert_model_validator_to_sql
        if 'contact_completeness' in validator_name or 'customer_contact' in validator_name:
            return "(email IS NOT NULL AND LENGTH(TRIM(email)) > 0) OR (phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0)|||Customer must have either email or phone contact\n(sex != 'M') OR (sex = 'M' AND email IS NOT NULL AND LENGTH(TRIM(email)) > 0)|||Male customers must provide email address"
        elif 'address_completeness' in validator_name:
            return "(street IS NULL AND city IS NULL AND state IS NULL AND zip IS NULL) OR (street IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL AND zip IS NOT NULL)|||Address must be either completely empty or completely filled"
        elif 'contact_method' in validator_name:
            return "(SUBSTRING(zip, 1, 2) NOT IN ('90', '10', '94')) OR (SUBSTRING(zip, 1, 2) IN ('90', '10', '94') AND email IS NOT NULL AND LENGTH(TRIM(email)) > 0 AND phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0)|||High-cost zip codes require both email and phone"
        elif 'geographic_service' in validator_name:
            return "(state NOT IN ('HI', 'AK')) OR (state IN ('HI', 'AK') AND phone IS NOT NULL AND LENGTH(TRIM(phone)) > 0)|||Remote states require phone contact"
        elif 'city_state_consistency' in validator_name:
            return "(LOWER(TRIM(city)) NOT IN ('los angeles', 'new york', 'miami', 'austin', 'seattle', 'chicago')) OR (LOWER(TRIM(city)) = 'los angeles' AND state = 'CA') OR (LOWER(TRIM(city)) = 'new york' AND state = 'NY') OR (LOWER(TRIM(city)) = 'miami' AND state = 'FL') OR (LOWER(TRIM(city)) = 'austin' AND state = 'TX') OR (LOWER(TRIM(city)) = 'seattle' AND state = 'WA') OR (LOWER(TRIM(city)) = 'chicago' AND state = 'IL')|||Major cities must match their correct states"
        elif 'phone_format' in validator_name:
            return "LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10|||Phone must have exactly 10 digits\nSUBSTRING(REGEXP_REPLACE(phone, '[^0-9]', ''), 1, 1) NOT IN ('0', '1')|||Phone area code cannot start with 0 or 1"
        else:
            # Default model validator response
            return f"1 = 1|||Placeholder validation from {validator_info.name}"
    
    @staticmethod
    def generic_fallback() -> str:
        """Generic fallback mock response."""
        return "1 = 1"


class AIResponseParser:
    """Utilities for parsing AI responses into structured data."""
    
    @staticmethod
    def parse_json_conditions(response: str) -> List[Dict[str, str]]:
        """Parse JSON array of conditions from AI response."""
        try:
            # Clean up the response (remove any markdown, extra whitespace)
            clean_response = response.strip()
            if clean_response.startswith('```'):
                # Remove code block markers if present
                lines = clean_response.split('\n')
                clean_response = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
            
            # Parse JSON
            conditions = json.loads(clean_response)
            
            # Validate structure
            if not isinstance(conditions, list):
                raise ValueError("Response must be a JSON array")
                
            for condition in conditions:
                if not isinstance(condition, dict) or 'condition' not in condition or 'description' not in condition:
                    raise ValueError("Each condition must have 'condition' and 'description' fields")
                    
            return conditions
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.getLogger(__name__).warning(f"Failed to parse AI JSON response: {e}")
            return []
    
    @staticmethod
    def parse_delimited_conditions(response: str, delimiter: str = '|||') -> List[Dict[str, str]]:
        """Parse delimited conditions from AI response."""
        conditions = []
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                if delimiter in line:
                    condition, description = line.split(delimiter, 1)
                    condition = condition.strip()
                    description = description.strip()
                    
                    if condition and description:
                        conditions.append({
                            'condition': condition,
                            'description': description
                        })
                        
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to parse delimited AI response: {e}")
            
        return conditions
    
    @staticmethod
    def clean_sql_condition(condition: str) -> str:
        """
        Aggressively clean and validate SQL condition.
        Returns empty string if condition is not valid SQL.
        """
        if not condition or not condition.strip():
            return ""
            
        original_condition = condition.strip()
        condition = original_condition
        
        # AGGRESSIVE: Remove duplicate WHERE clauses (common AI error)
        condition = re.sub(r'\bWHERE\s+', '', condition, flags=re.IGNORECASE)
        
        # AGGRESSIVE: Remove EXISTS clauses and complex subqueries (usually verbose AI)
        condition = re.sub(r'\bAND\s+NOT\s+EXISTS\s*\([^)]+\)', '', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bNOT\s+EXISTS\s*\([^)]+\)', '', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bEXISTS\s*\([^)]+\)', '', condition, flags=re.IGNORECASE)
        
        # AGGRESSIVE: Remove ARRAY functions (usually verbose AI)
        condition = re.sub(r'\bARRAY_MIN\([^)]+\)', '1=1', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bTRANSFORM\([^)]+\)', '1=1', condition, flags=re.IGNORECASE)
        condition = re.sub(r'\bEXPLODE\([^)]+\)', '1=1', condition, flags=re.IGNORECASE)
        
        # AGGRESSIVE: If condition is longer than 200 chars, it's likely verbose AI
        if len(condition) > 200:
            return ""  # Filter out verbose responses
            
        # AGGRESSIVE: Remove obvious duplicate conditions
        # Split by AND and look for exact duplicates
        and_parts = re.split(r'\s+AND\s+', condition, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            # Remove exact duplicates
            unique_parts = []
            seen = set()
            for part in and_parts:
                normalized = part.strip().upper()
                if normalized not in seen:
                    seen.add(normalized)
                    unique_parts.append(part.strip())
            
            # If we removed duplicates, rebuild condition
            if len(unique_parts) < len(and_parts):
                condition = ' AND '.join(unique_parts)
        
        # Remove code block markers
        condition = re.sub(r'```[a-z]*\n?', '', condition)
        condition = re.sub(r'\n?```', '', condition)
        
        # Remove SQL comments
        condition = re.sub(r'--.*?(?=\n|$)', '', condition, flags=re.MULTILINE)
        condition = re.sub(r'/\*.*?\*/', '', condition, flags=re.DOTALL)
        
        # Remove markdown and explanations
        condition = re.sub(r'\*\*.*?\*\*', '', condition)
        condition = re.sub(r'^\d+\.\s*.*?(?=\n|$)', '', condition, flags=re.MULTILINE)
        condition = re.sub(r'#+\s*.*?(?=\n|$)', '', condition, flags=re.MULTILINE)
        
        # Remove explanatory text patterns
        condition = re.sub(r'Explanation:.*', '', condition, flags=re.DOTALL)
        condition = re.sub(r'Alternative.*', '', condition, flags=re.DOTALL)
        condition = re.sub(r'\(catches.*?\)', '', condition)
        condition = re.sub(r'\(equivalent.*?\)', '', condition)
        
        # Clean whitespace
        condition = re.sub(r'\s+', ' ', condition).strip()
        
        # AGGRESSIVE: Check for verbose patterns that indicate AI explanations
        verbose_indicators = [
            r'SELECT\s+1\s+FROM',  # Subqueries
            r'EXPLODE\s*\(',       # Explode functions
            r'TRANSFORM\s*\(',     # Transform functions  
            r'LENGTH\s*\(\s*part\s*\)',  # Nested function calls
            r'WHERE\s+LENGTH\s*\(',      # WHERE in subqueries
            r'AS\s+part\)',               # Aliasing in subqueries
        ]
        
        for pattern in verbose_indicators:
            if re.search(pattern, condition, re.IGNORECASE):
                return ""  # Filter out verbose SQL
        
        # AGGRESSIVE: If it contains both SPLIT and EXPLODE, it's verbose AI
        if 'SPLIT(' in condition.upper() and 'EXPLODE(' in condition.upper():
            return ""
        
        # AGGRESSIVE: If it has more than 8 AND clauses, it's likely verbose
        and_count = len(re.findall(r'\bAND\b', condition, re.IGNORECASE))
        if and_count > 8:
            return ""
        
        # Basic validation - must have valid SQL patterns
        valid_sql_indicators = ['IS NOT NULL', 'IS NULL', 'RLIKE', 'LIKE', 'REGEXP_LIKE', '=', '>', '<', '>=', '<=', 'IN', 'NOT IN', 'LENGTH(', 'TRIM(']
        has_valid_sql = any(indicator in condition.upper() for indicator in valid_sql_indicators)
        
        if not has_valid_sql:
            return ""
        
        # Final check - if it's still very long, it's probably verbose
        if len(condition) > 150:
            return ""
            
        return condition
    
    @staticmethod
    def clean_verbose_content(content: str) -> str:
        """
        Removes verbose patterns, markdown, and explanations from AI-generated content.
        This is a safety net for descriptions and other text content.
        """
        if not content:
            return ""

        # Remove markdown code blocks
        content = re.sub(r'```sql.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)

        # Remove SQL comments
        content = re.sub(r'--.*', '', content)

        # Remove "WHERE" if it's at the start of a line
        content = re.sub(r'^\s*WHERE\s+', '', content, flags=re.MULTILINE | re.IGNORECASE)

        # Remove common explanation headers/footers
        content = re.sub(r'#+\s*SQL Validation Conditions.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*Alternative Validation Patterns.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*Explanation:.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*Recommended Combined Validation.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*Databricks-Specific Functions.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*Strict Name Validation.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*International Name Support.*', '', content, flags=re.DOTALL)
        content = re.sub(r'#+\s*Simplified version \(more performant\):.*', '', content, flags=re.DOTALL)
        
        # Remove numbered explanations and markdown formatting
        content = re.sub(r'^\d+\.\s*\*\*.*?\*\*.*?(?=\n|$)', '', content, flags=re.MULTILINE)
        content = re.sub(r'\*\*.*?\*\*', '', content)  # Bold text
        
        # Remove "SQL:" prefix that sometimes appears
        content = re.sub(r'^\s*SQL:\s*', '', content, flags=re.IGNORECASE)

        # Remove excessive newlines and trim whitespace
        content = re.sub(r'\n\s*\n', '\n', content).strip()
        content = ' '.join(content.split())  # Normalize whitespace

        return content
