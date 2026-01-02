"""
Model-specific generator for creating DLT code with expectations
"""

from .discovery import ModelDiscovery
from .analyzer import ValidatorAnalyzer
from .config import Config
from .ai_utils import AIQueryExecutor, AIPromptTemplates, AIResponseParser

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
        
        # AI executor for consistent AI handling
        import logging
        self.ai_executor = AIQueryExecutor(config, logging.getLogger(__name__))
        
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
        
        # Discover all models from configured paths
        all_models, all_validators = self.discovery.scan_repository()
        
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
        
        # Analyze the model (AI enhancement handled by ValidatorAnalyzer based on config)
        self.schema = self.analyzer.analyze_model(self.model_info, all_validators)
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
        Generate just the expectations (decorators) for the model.
        
        Filters out reference rules ('see validator:' conditions) to match
        the DLT template generation behavior.
        
        Returns:
            list: List of valid expectation decorator strings
        """
        if not self.found:
            return [f"# Error: Model '{self.model_name}' not found"]
        
        expectations = []
        for rule in self.schema.rules:
            # Skip reference rules that start with 'see validator:' 
            if rule.condition.startswith('see validator:'):
                continue
                
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
                
                # Create AI prompt using standardized template
                prompt = AIPromptTemplates.custom_validator_conversion(field_name, field_type, validator_code)
                
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
        Call AI query using shared AI executor with caching support
        
        Args:
            prompt: The AI prompt
            field_name: Field name for context
            field_type: Field type for context
            
        Returns:
            list: SQL conditions from AI or fallback
        """
        try:
            context_info = {
                'field_name': field_name,
                'field_type': field_type
            }
            
            response_str = self.ai_executor.execute_query(prompt, context_info)
            
            # Parse response - both real and mock responses are now JSON strings
            if response_str:
                return AIResponseParser.parse_json_conditions(response_str)
            else:
                # Fallback if no response
                return []
                
        except Exception as e:
            print(f"Unexpected error in AI query for {field_name}: {e}")
            return []
    
    def _generate_ai_response_for_field(self, prompt, field_name, field_type, ai_model, timeout):
        """Generate AI response for field validation (without caching logic)."""
        # Check if we should use real AI query or mock responses
        if self.config.validation.ai.use_mock_responses:
            # Development/demo mode: use mock responses
            print(f"Using mock AI response for {field_name} ({field_type}) validation (demo mode)")
            print(f"Would use Model: {ai_model}, Timeout: {timeout}s")
            print("Prompt preview:", prompt[:100] + "...")
            return "mock_response"  # Processed by _get_mock_response_for_field
        else:
            # Production mode: use real Databricks ai_query()
            try:
                print(f"Calling real ai_query for {field_name} ({field_type}) using model {ai_model}")
                
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
                    '{ai_model}', 
                    '{escaped_prompt}'
                ) as ai_response
                """
                
                result_df = spark.sql(sql_query)
                ai_result = result_df.collect()[0]['ai_response']
                return ai_result
                
            except ImportError:
                print("ai_query not available (not in Databricks environment). Using basic fallback.")
                if self.config.validation.ai.fallback_enabled:
                    return f'[{{"condition": "{field_name} IS NOT NULL", "description": "Basic {field_type} validation"}}]'
                else:
                    return "[]"
            except Exception as e:
                print(f"Real AI query failed for {field_name}: {e}")
                if self.config.validation.ai.fallback_enabled:
                    return f'[{{"condition": "{field_name} IS NOT NULL", "description": "Fallback {field_type} validation"}}]'
                else:
                    return "[]"
    
    def _get_mock_response_for_field(self, field_name, field_type):
        """Get mock response data for field type."""
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
    
