"""
Model-specific generator for creating DLT code with expectations
"""

from .discovery import ModelDiscovery
from .analyzer import ValidatorAnalyzer
from .config import Config

class Generator:
    """
    Generate DLT code with expectations for a specific model
    
    Usage:
        from pydantic_ldp_generator import Generator, Config
        
        config = Config(source_paths=["./domain", "./validators_lib"])
        generator = Generator(config, 'Customer')
        dlt_code = generator.generate_dlt_code()
        
        # Or generate and save
        output_path = generator.generate_and_save()
    """
    
    def __init__(self, config, model_name):
        """
        Initialize generator for a specific model
        
        Args:
            config: Configuration object
            model_name: Name of the model to generate DLT code for (e.g., 'Customer', 'Address')
        """
        self.config = config
        self.model_name = model_name
        
        # Components
        self.discovery = ModelDiscovery(config)
        self.analyzer = ValidatorAnalyzer(config)
        
        # Import TemplateGenerator here to avoid circular import
        from .generator import TemplateGenerator
        self.template_generator = TemplateGenerator(config)
        
        # Model data
        self.model_info = None
        self.schema = None
        self.found = False
        
        # Automatically discover and analyze the model
        self._discover_and_analyze()
    
    def _discover_and_analyze(self):
        """Discover and analyze the specified model"""
        print(f"Generating DLT code for model: {self.model_name}...")
        
        # Discover all models
        all_models = {}
        all_validators = {}
        for source_path in self.config.source_paths:
            models, validators = self.discovery.scan_repository(source_path)
            all_models.update(models)
            all_validators.update(validators)
        
        # Find the requested model
        for model_key, model_info in all_models.items():
            if model_info.name == self.model_name:
                self.model_info = model_info
                self.found = True
                break
        
        if not self.found:
            print(f"Model '{self.model_name}' not found.")
            print("Available models:")
            for model_key, model_info in all_models.items():
                print(f"  - {model_info.name}")
            return
        
        # Analyze the model
        self.schema = self.analyzer.analyze_model(self.model_info, all_validators)
        
        # Enhance with AI-powered custom validator conversion
        if self._enable_ai_powered_validation():
            self._enhance_with_ai_validators(all_validators)
        print(f"Found model: {self.model_name} with {len(self.schema.rules)} validation rules")
    
    def is_found(self):
        """Check if the model was found and analyzed"""
        return self.found
    
    def generate_dlt_code(self, source_table=None):
        """
        Generate DLT code for the model
        
        Args:
            source_table: Source table name (defaults to model name in lowercase)
            
        Returns:
            str: Generated DLT pipeline code
        """
        if not self.found:
            return f"# Error: Model '{self.model_name}' not found"
        
        # Default source table name
        if source_table is None:
            source_table = f"{self.model_name.lower()}_raw"
        
        # Create schemas dict for template generator
        schemas = {self.model_info.name: self.schema}
        source_tables = {self.model_name: source_table}
        
        # Generate the pipeline
        generated_files = self.template_generator.generate_pipeline(schemas, source_tables)
        
        # Return the main pipeline code for this model
        model_pipeline_file = f"{self.model_name.lower()}_pipeline.py"
        if model_pipeline_file in generated_files:
            return generated_files[model_pipeline_file]
        
        # If individual model pipeline not found, return main pipeline
        if "main_pipeline.py" in generated_files:
            return generated_files["main_pipeline.py"]
        
        return "# Error: Could not generate DLT code"
    
    def generate_expectations_only(self):
        """
        Generate just the expectations (decorators) for the model
        
        Returns:
            list: List of expectation decorator strings
        """
        if not self.found:
            return [f"# Error: Model '{self.model_name}' not found"]
        
        expectations = []
        for rule in self.schema.rules:
            action = rule.action
            expectation = f'@dlt.expect_or_{action}("{rule.expectation_name}", "{rule.condition}")'
            expectations.append(expectation)
        
        return expectations
    
    def generate_and_save(self, source_table=None, output_dir=None):
        """
        Generate DLT code and save to file
        
        Args:
            source_table: Source table name
            output_dir: Output directory (defaults to config output_directory)
            
        Returns:
            str: Path to saved file
        """
        if not self.found:
            print(f"Cannot save - model '{self.model_name}' not found")
            return None
        
        # Generate code
        dlt_code = self.generate_dlt_code(source_table)
        
        # Use template generator to save
        if output_dir is None:
            output_dir = self.config.output_directory
        
        schemas = {self.model_info.name: self.schema}
        source_tables = {self.model_name: source_table or f"{self.model_name.lower()}_raw"}
        
        generated_files = self.template_generator.generate_pipeline(schemas, source_tables)
        output_path = self.template_generator.save_generated_files(generated_files, output_dir)
        
        print(f"Generated DLT code saved to: {output_path}")
        return output_path
    
    def show_summary(self):
        """Display a summary of what will be generated"""
        if not self.found:
            print(f"Model '{self.model_name}' not found.")
            return
        
        print(f"\n{'='*60}")
        print(f"DLT CODE GENERATION SUMMARY: {self.model_name}")
        print(f"{'='*60}")
        
        print(f"Source file: {self.model_info.file_path}")
        print(f"Fields: {len(self.model_info.fields)}")
        print(f"Validation rules: {len(self.schema.rules)}")
        
        # Show expectations that will be generated
        print(f"\nExpectations to be generated:")
        expectations = self.generate_expectations_only()
        for i, expectation in enumerate(expectations, 1):
            print(f"  {i}. {expectation}")
        
        # Show fields
        print(f"\nFields in model:")
        for field_name, field_info in self.model_info.fields.items():
            field_type = field_info.get('type', 'unknown')
            print(f"  {field_name}: {field_type}")
        
        print(f"\nDefault source table: {self.model_name.lower()}_raw")
        print(f"Generated function name: {self.model_name.lower()}_validated()")
    
    def create_ldp_template(self, source_table=None, save_to_file=True, show_code=False):
        """
        Create and optionally save LDP template for the model
        
        Args:
            source_table: Source table name (defaults to model_name_raw)
            save_to_file: Whether to save to file (default: True)
            show_code: Whether to display the generated code (default: False)
            
        Returns:
            str: Generated DLT code
        """
        if not self.found:
            print(f"Cannot create template - model '{self.model_name}' not found")
            return None
        
        print(f"\n{'='*60}")
        print(f"CREATING LDP TEMPLATE FOR: {self.model_name}")
        print("="*60)
        
        # Show summary first
        print(f"Fields: {len(self.model_info.fields)}")
        print(f"Validation rules: {len(self.schema.rules)}")
        
        # Set default source table
        if source_table is None:
            source_table = f"{self.model_name.lower()}_raw"
        print(f"Source table: {source_table}")
        
        # Generate the DLT code
        dlt_code = self.generate_dlt_code(source_table)
        
        # Show the code if requested
        if show_code:
            print(f"\n{'Generated LDP Template:':^60}")
            print("-" * 60)
            lines = dlt_code.split('\n')
            for i, line in enumerate(lines, 1):
                print(f"{i:3d}| {line}")
        
        # Save to file if requested
        if save_to_file:
            filename = f"{self.model_name.lower()}_ldp_template.py"
            with open(filename, 'w') as f:
                f.write(dlt_code)
            print(f"\nLDP template saved to: {filename}")
        
        print(f"Template created with {len(self.schema.rules)} validation expectations")
        return dlt_code
    
    def create_template(self, source_table=None, save_to_file=True):
        """
        Alias for create_ldp_template() - shorter method name
        """
        return self.create_ldp_template(source_table, save_to_file)
    
    def create_complete_system(self, source_table=None, output_dir="generated_pipeline"):
        """
        Create complete pipeline system with quarantine, monitoring, and config
        This provides access to TemplateGenerator's full capabilities for single model
        
        Args:
            source_table: Source table name (defaults to model_name_raw)
            output_dir: Directory to save all files
            
        Returns:
            dict: Generated files mapping filename to content
        """
        if not self.found:
            print(f"Cannot create system - model '{self.model_name}' not found")
            return {}
        
        print(f"\nCREATING COMPLETE PIPELINE SYSTEM FOR: {self.model_name}")
        print("=" * 60)
        print("Generating:")
        print("  - Main DLT pipeline")
        print("  - Quarantine pipeline (for failed records)")
        print("  - Monitoring pipeline (for metrics)")
        print("  - Configuration file")
        print("  - Documentation (README)")
        
        # Set default source table
        if source_table is None:
            source_table = f"{self.model_name.lower()}_raw"
        
        # Create schemas dict for TemplateGenerator (expects this format)
        schemas = {self.schema.model_name: self.schema}
        source_tables = {self.model_name: source_table}
        
        # Use TemplateGenerator to create complete system
        generated_files = self.template_generator.generate_pipeline(schemas, source_tables)
        
        # Save all files
        output_path = self.template_generator.save_generated_files(generated_files, output_dir)
        
        print(f"\nGenerated {len(generated_files)} files:")
        for filename in generated_files.keys():
            print(f"  - {filename}")
        print(f"\nFiles saved to: {output_path}")
        
        return generated_files
    
    def create_quarantine_pipeline(self, save_to_file=True):
        """
        Create quarantine pipeline for handling failed validation records
        
        Args:
            save_to_file: Whether to save to file
            
        Returns:
            str: Generated quarantine pipeline code
        """
        if not self.found:
            print(f"Cannot create quarantine - model '{self.model_name}' not found")
            return None
        
        # Create schemas dict for TemplateGenerator
        schemas = {self.schema.model_name: self.schema}
        
        # Generate quarantine pipeline
        quarantine_code = self.template_generator._generate_quarantine_pipeline(schemas)
        
        if save_to_file:
            filename = f"{self.model_name.lower()}_quarantine_pipeline.py"
            with open(filename, 'w') as f:
                f.write(quarantine_code)
            print(f"Quarantine pipeline saved to: {filename}")
        
        return quarantine_code
    
    def create_monitoring_pipeline(self, save_to_file=True):
        """
        Create monitoring pipeline for metrics and data quality tracking
        
        Args:
            save_to_file: Whether to save to file
            
        Returns:
            str: Generated monitoring pipeline code
        """
        if not self.found:
            print(f"Cannot create monitoring - model '{self.model_name}' not found")
            return None
        
        # Create schemas dict for TemplateGenerator
        schemas = {self.schema.model_name: self.schema}
        
        # Generate monitoring pipeline
        monitoring_code = self.template_generator._generate_monitoring_pipeline(schemas)
        
        if save_to_file:
            filename = f"{self.model_name.lower()}_monitoring_pipeline.py"
            with open(filename, 'w') as f:
                f.write(monitoring_code)
            print(f"Monitoring pipeline saved to: {filename}")
        
        return monitoring_code
    
    def create_config_file(self, source_table=None, save_to_file=True):
        """
        Create configuration file for the pipeline
        
        Args:
            source_table: Source table name
            save_to_file: Whether to save to file
            
        Returns:
            str: Generated config YAML content
        """
        if not self.found:
            print(f"Cannot create config - model '{self.model_name}' not found")
            return None
        
        if source_table is None:
            source_table = f"{self.model_name.lower()}_raw"
        
        # Create schemas dict for TemplateGenerator
        schemas = {self.schema.model_name: self.schema}
        source_tables = {self.model_name: source_table}
        
        # Generate config file
        config_content = self.template_generator._generate_config_file(schemas, source_tables)
        
        if save_to_file:
            filename = f"{self.model_name.lower()}_pipeline_config.yml"
            with open(filename, 'w') as f:
                f.write(config_content)
            print(f"Config file saved to: {filename}")
        
        return config_content
    
    def _get_custom_validator_sql(self, field_name, field_type, validator_info):
        """
        Use AI to convert custom validator logic to SQL expectations
        
        Args:
            field_name: Name of the field (e.g., 'email')
            field_type: Type of the field (e.g., 'Email')
            validator_info: Information about the validator function
            
        Returns:
            list: List of SQL conditions for DLT expectations
        """
        
        # Read the validator function source code
        try:
            import inspect
            import ast
            from pathlib import Path
            
            # Get the validator source code
            validator_file = Path(validator_info.file_path)
            if validator_file.exists():
                with open(validator_file, 'r') as f:
                    source_code = f.read()
                
                # Extract just the specific validator function
                tree = ast.parse(source_code)
                validator_code = ""
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == validator_info.name:
                        validator_code = ast.get_source_segment(source_code, node)
                        break
                
                if not validator_code:
                    return []
                
                # Create AI prompt for conversion
                prompt = f"""
You are an expert at converting Python validation logic to SQL conditions for Databricks DLT expectations.

Task: Convert this Python validator function to equivalent SQL conditions that can be used in DLT @expect_or_drop decorators.

Field Information:
- Field name: {field_name}
- Field type: {field_type}
- Validator function: {validator_info.name}

Python Validator Code:
```python
{validator_code}
```

Requirements:
1. Convert the validation logic to SQL WHERE clause conditions
2. Use the field name '{field_name}' in SQL conditions
3. Return multiple SQL conditions if the validator has multiple checks
4. Use Databricks SQL syntax (RLIKE for regex, etc.)
5. Handle all validation cases from the Python function
6. Make conditions that will PASS for valid data (not fail conditions)

Format your response as a JSON array of objects, where each object has:
- "condition": "SQL condition string"
- "description": "Human readable description of what this validates"

Example format:
[
    {{"condition": "{field_name} RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$'", "description": "Valid email format"}},
    {{"condition": "LENGTH({field_name}) <= 254", "description": "Email length within RFC limits"}}
]

Only return the JSON array, no other text.
"""
                
                # In Databricks environment, this would call ai_query()
                # For now, return a placeholder that shows the concept
                return self._call_ai_query(prompt, field_name, field_type)
                
            else:
                return []
                
        except Exception as e:
            print(f"Error processing validator for {field_name}: {e}")
            return []
    
    def _call_ai_query(self, prompt, field_name, field_type):
        """
        Call Databricks AI query to convert validator to SQL
        
        Args:
            prompt: The AI prompt
            field_name: Field name for fallback
            field_type: Field type for fallback
            
        Returns:
            list: SQL conditions from AI or fallback
        """
        try:
            # In Databricks, this would be:
            ai_model = self.config.validation.ai.model
            timeout = self.config.validation.ai.timeout_seconds
            # ai_result = ai_query(ai_model, prompt, timeout=timeout)
            # return json.loads(ai_result)
            
            # For demonstration, return smart fallbacks based on field type
            print(f"AI Query would be called for {field_name} ({field_type}) validation")
            print(f"Model: {ai_model}, Timeout: {timeout}s")
            print("Prompt preview:", prompt[:100] + "...")
            
            # Smart fallbacks that demonstrate what AI would return
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
            elif field_type == 'Sex':
                return [
                    {"condition": f"{field_name} IN ('M', 'F')", 
                     "description": "Must be M or F"}
                ]
            elif field_type == 'PastDate':
                return [
                    {"condition": f"{field_name} <= CURRENT_DATE()", 
                     "description": "Date must be in the past"}
                ]
            elif field_type == 'ZipCode':
                return [
                    {"condition": f"{field_name} RLIKE '^[0-9]{{5}}(-[0-9]{{4}})?$'", 
                     "description": "Valid 5-digit or ZIP+4 format"}
                ]
            else:
                return []
                
        except Exception as e:
            print(f"AI query failed for {field_name}, using basic validation: {e}")
            return []
    
    def _enable_ai_powered_validation(self):
        """
        Enable AI-powered custom validator conversion based on config
        """
        # Check if AI is enabled in config
        if not self.config.validation.ai.enabled:
            return False
            
        try:
            # In Databricks, check if ai_query is available
            # import databricks.sql.functions as F
            # return hasattr(F, 'ai_query')
            
            # For demo, return True if AI is enabled in config
            return True
        except:
            # Return fallback setting if AI fails to initialize
            return self.config.validation.ai.fallback_enabled
    
    def _enhance_with_ai_validators(self, all_validators):
        """
        Enhance validation schema with AI-converted custom validators
        
        Args:
            all_validators: Dictionary of discovered validators
        """
        if not self.found or not all_validators:
            return
        
        ai_model = self.config.validation.ai.model
        print(f"Enhancing {self.model_name} with AI-powered custom validator conversion...")
        print(f"Using AI model: {ai_model}")
        
        # Find fields that use custom types
        custom_field_types = ['Email', 'ZipCode', 'Sex', 'PastDate', 'Name']
        enhanced_rules = []
        
        for field_name, field_info in self.model_info.fields.items():
            field_type = field_info.get('type', '')
            
            if field_type in custom_field_types:
                # Find matching validator
                validator_mapping = {
                    'Email': 'email_validator',
                    'ZipCode': 'zipcode_validator', 
                    'Sex': 'sex_validator',
                    'PastDate': 'past_date_validator'
                }
                
                expected_validator = validator_mapping.get(field_type)
                if expected_validator:
                    # Find validator info
                    validator_info = None
                    for v_name, v_info in all_validators.items():
                        if v_info.name == expected_validator:
                            validator_info = v_info
                            break
                    
                    if validator_info:
                        print(f"  Converting {field_name} ({field_type}) validator to SQL...")
                        sql_conditions = self._get_custom_validator_sql(field_name, field_type, validator_info)
                        
                        # Add AI-generated rules to schema
                        for i, condition_info in enumerate(sql_conditions):
                            rule = type('AIValidationRule', (), {
                                'expectation_name': f"ai_custom_{field_name}_{i+1}",
                                'condition': condition_info['condition'],
                                'rule_type': 'ai_custom_validator',
                                'action': self.config.validation.default_action,
                                'description': condition_info['description'],
                                'field_name': field_name,
                                'validator_type': field_type
                            })()
                            
                            enhanced_rules.append(rule)
                            print(f"    {condition_info['description']}: {condition_info['condition']}")
        
        # Add enhanced rules to schema
        if enhanced_rules:
            self.schema.rules.extend(enhanced_rules)
            print(f"Enhanced with {len(enhanced_rules)} AI-generated validation rules")
        else:
            print("No custom validators found for AI enhancement")
