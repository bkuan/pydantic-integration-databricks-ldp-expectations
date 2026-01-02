"""
Model Selector for the pydantic_dlt_generator library
"""

from .discovery import ModelDiscovery
from .analyzer import ValidatorAnalyzer
from .config import Config

class ModelSelector:
    """
    Select and analyze a specific model directly by name
    
    Usage:
        from pydantic_ldp_generator import ModelSelector, Config
        
        config = Config(source_paths=["./domain", "./validators_lib"])
        selector = ModelSelector(config, 'Customer')
        selector.show_analysis()
        
        # Or get the data programmatically:
        rules = selector.get_validation_rules()
        fields = selector.get_fields()
        custom_rules = selector.get_custom_validator_rules()
    """
    
    def __init__(self, config, model_name):
        """
        Initialize with configuration and specific model name
        
        Args:
            config: Configuration object (AI enhancement controlled via config.validation.ai.enabled)
            model_name: Name of the model to analyze (e.g., 'Customer', 'Address')
        """
        self.config = config
        self.model_name = model_name
        
        # Discovery and analysis components
        self.discovery = ModelDiscovery(self.config)
        self.analyzer = ValidatorAnalyzer(self.config)
        
        # Model data
        self.model_info = None
        self.schema = None
        self.all_validators = {}
        self.found = False
        
        # Automatically discover and select the model
        self._discover_and_select()
    
    def _discover_and_select(self):
        """Discover all models and find the requested one"""
        print(f"Looking for model: {self.model_name}...")
        
        # Discover all models from configured paths
        all_models, all_validators = self.discovery.scan_repository()
        self.all_validators = all_validators
        
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
        self.schema = self.analyzer.analyze_model(self.model_info, self.all_validators)
        print(f"Found and analyzed model: {self.model_name}")
    
    def is_found(self):
        """Check if the model was found"""
        return self.found
    
    def get_fields(self):
        """Get all fields of the model"""
        if not self.found:
            return {}
        return self.model_info.fields
    
    def get_validation_rules(self):
        """Get all validation rules for the model"""
        if not self.found:
            return []
        return self.schema.rules
    
    def get_custom_validator_rules(self):
        """
        Get custom validator rules for fields with custom types or validators.
        
        Returns a list of dictionaries for programmatic use. These same rules
        are displayed in show_analysis() under 'VALIDATION RULES BY TYPE'.
        """
        if not self.found:
            return []
        
        custom_rules = []
        
        # Get all validation rules and filter for custom ones
        for rule in self.schema.rules:
            if rule.rule_type in ['custom_type', 'custom_validator', 'field_validator', 'model_validator']:
                custom_rules.append({
                    'field_name': rule.field_name,
                    'rule_type': rule.rule_type,
                    'condition': rule.condition,
                    'description': rule.description,
                    'expectation_name': rule.expectation_name
                })
        
        return custom_rules
    
    def get_custom_validator_opportunities(self):
        """Deprecated: Use get_custom_validator_rules() instead"""
        # Keep for backward compatibility but redirect to new method
        return self.get_custom_validator_rules()
    
    def get_available_validators(self):
        """Get all discovered validators"""
        return self.all_validators
    
    def _get_potential_validator(self, field_type):
        """Map field type to potential validator"""
        mapping = {
            'Email': 'email_validator',
            'ZipCode': 'zipcode_validator', 
            'Sex': 'sex_validator',
            'PastDate': 'past_date_validator',
            'Name': 'name_validator'
        }
        return mapping.get(field_type, 'unknown')
    
    def show_analysis(self):
        """Display complete analysis of the model"""
        if not self.found:
            print(f"Cannot show analysis - model '{self.model_name}' not found.")
            return
        
        print(f"\n{'='*60}")
        print(f"MODEL ANALYSIS: {self.model_name}")
        print(f"{'='*60}")
        
        print(f"Source file: {self.model_info.file_path}")
        print(f"Total fields: {len(self.model_info.fields)}")
        
        # Show fields
        print(f"\nFIELDS:")
        for field_name, field_info in self.model_info.fields.items():
            field_type = field_info.get('type', 'unknown')
            print(f"  {field_name}: {field_type}")
        
        # Show validation rules
        print(f"\nVALIDATION RULES BY TYPE ({len(self.schema.rules)} total):")
        
        # Group rules by the new rule types
        rule_groups = {
            'required_field': [],
            'basic_type_check': [],
            'custom_type': [],
            'custom_validator': [],  # All custom validator rules consolidated here
            'field_validator': [],   # @field_validator decorated methods
            'model_validator': []    # @model_validator decorated methods (cross-field)
        }
        
        for rule in self.schema.rules:
            rule_type = rule.rule_type
            if rule_type in rule_groups:
                rule_groups[rule_type].append(rule)
            else:
                # Handle any other rule types
                if 'other' not in rule_groups:
                    rule_groups['other'] = []
                rule_groups['other'].append(rule)
        
        for rule_type, rules in rule_groups.items():
            if rules:
                print(f"\n  {rule_type.upper().replace('_', ' ')} ({len(rules)} rules):")
                for rule in rules:
                    # Special formatting for reference rules
                    if rule.action == "reference" or "see validator:" in rule.condition:
                        print(f"    • {rule.field_name}: {rule.condition}")
                        if rule.description:
                            print(f"      Type annotation references validator function")
                    else:
                        print(f"    • {rule.field_name}: {rule.condition}")
                        if rule.description and rule_type in ['custom_type', 'custom_validator']:
                            print(f"      Description: {rule.description}")
        
        # Show summary statistics
        custom_rule_count = len([r for r in self.schema.rules if r.rule_type in ['custom_type', 'custom_validator', 'field_validator', 'model_validator']])
        print(f"\nSUMMARY:")
        print(f"  • Total validation rules: {len(self.schema.rules)}")
        print(f"  • Custom validation rules: {custom_rule_count}")
    
    def get_summary(self):
        """Get a summary dictionary of all model information"""
        if not self.found:
            return None
        
        return {
            'model_name': self.model_name,
            'file_path': self.model_info.file_path,
            'field_count': len(self.model_info.fields),
            'fields': {name: info.get('type', 'unknown') for name, info in self.model_info.fields.items()},
            'validation_rule_count': len(self.schema.rules),
            'validation_rules': [
                {
                    'name': rule.expectation_name,
                    'condition': rule.condition,
                    'type': rule.rule_type,
                    'action': rule.action
                } for rule in self.schema.rules
            ],
            'custom_validator_rules': self.get_custom_validator_rules(),
            'available_validators': list(self.all_validators.keys())
        }
    
