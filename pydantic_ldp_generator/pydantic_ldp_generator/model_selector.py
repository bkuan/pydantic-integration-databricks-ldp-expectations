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
        opportunities = selector.get_custom_validator_opportunities()
    """
    
    def __init__(self, config, model_name):
        """
        Initialize with configuration and specific model name
        
        Args:
            config: Configuration object (like ModelDiscovery pattern)
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
        
        # Discover all models
        all_models = {}
        for source_path in self.config.source_paths:
            models, validators = self.discovery.scan_repository(source_path)
            all_models.update(models)
            self.all_validators.update(validators)
        
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
    
    def get_custom_validator_opportunities(self):
        """Get fields that could use custom validators"""
        if not self.found:
            return []
        
        opportunities = []
        custom_field_types = ['Email', 'ZipCode', 'Sex', 'PastDate', 'Name']
        
        for field_name, field_info in self.model_info.fields.items():
            field_type = field_info.get('type', '')
            if field_type in custom_field_types:
                opportunities.append({
                    'field_name': field_name,
                    'field_type': field_type,
                    'potential_validator': self._get_potential_validator(field_type)
                })
        
        return opportunities
    
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
        print(f"\nCURRENT VALIDATION RULES ({len(self.schema.rules)}):")
        for i, rule in enumerate(self.schema.rules, 1):
            print(f"  {i}. {rule.expectation_name}")
            print(f"     SQL: {rule.condition}")
            print(f"     Type: {rule.rule_type}, Action: {rule.action}")
        
        # Show custom validator opportunities
        opportunities = self.get_custom_validator_opportunities()
        print(f"\nCUSTOM VALIDATOR OPPORTUNITIES ({len(opportunities)}):")
        if opportunities:
            for opp in opportunities:
                print(f"  {opp['field_name']} ({opp['field_type']})")
                print(f"    Could use: {opp['potential_validator']}")
                print(f"    Benefit: Enhanced validation beyond basic required/type checks")
        else:
            print("  No custom validator opportunities identified")
        
        # Show discovered validators
        print(f"\nAVAILABLE VALIDATORS ({len(self.all_validators)}):")
        for validator_name, validator_info in self.all_validators.items():
            print(f"  {validator_info.name}")
            print(f"    File: {validator_info.file_path}")
            if validator_info.docstring:
                doc = validator_info.docstring[:80] + "..." if len(validator_info.docstring) > 80 else validator_info.docstring
                print(f"    Description: {doc}")
    
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
            'custom_opportunities': self.get_custom_validator_opportunities(),
            'available_validators': list(self.all_validators.keys())
        }
