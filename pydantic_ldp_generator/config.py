"""
Configuration management for pydantic LDP generator.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration for generated pipelines."""
    catalog: str = "main"
    schema: str = "default" 
    table_prefix: str = ""
    
    def get_full_table_name(self, table_name: str) -> str:
        """Get fully qualified table name."""
        prefix = f"{self.table_prefix}_" if self.table_prefix else ""
        return f"{self.catalog}.{self.schema}.{prefix}{table_name}"


@dataclass
class AIConfig:
    """Configuration for AI-powered validation"""
    enabled: bool = True
    model: str = "databricks-claude-sonnet-4"
    fallback_enabled: bool = True  # Use basic rules if AI fails
    timeout_seconds: int = 30
    use_mock_responses: bool = True  # Set to False to use real ai_query() in Databricks
    enable_cache: bool = True  # Cache AI responses to avoid duplicate queries
    cache_size_limit: int = 1000  # Maximum number of cached responses

@dataclass
class ValidatorAnalysisConfig:
    """Configuration for enhanced validator analysis with dependency handling."""
    # Complexity handling
    max_dependency_depth: int = 3
    allow_circular_dependencies: bool = False
    handle_external_dependencies: bool = False
    
    # AI conversion settings  
    ai_conversion_enabled: bool = True
    ai_fallback_to_basic: bool = True
    ai_timeout_seconds: int = 30
    
    # Complexity thresholds
    simple_max_dependencies: int = 0       # 0 deps = simple
    medium_max_dependencies: int = 2       # 1-2 deps = medium
    high_dependency_threshold: int = 3     # 3+ deps = high
    
    # Output preferences
    generate_warnings: bool = True
    include_manual_review_flags: bool = True
    verbose_descriptions: bool = True

@dataclass
class ValidationConfig:
    """Validation behavior configuration."""
    default_action: str = "drop"  # drop, fail, warn
    strict_mode: bool = True
    generate_quarantine: bool = True
    generate_monitoring: bool = True
    ai: AIConfig = field(default_factory=AIConfig)
    validator_analysis: ValidatorAnalysisConfig = field(default_factory=ValidatorAnalysisConfig)
    
    def __post_init__(self):
        if self.default_action not in {"drop", "fail", "warn"}:
            raise ValueError("default_action must be 'drop', 'fail', or 'warn'")


@dataclass
class TemplateConfig:
    """Template generation configuration."""
    include_comments: bool = True
    include_metadata: bool = True
    generate_tests: bool = True
    spark_conf: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration class for pydantic DLT generator."""
    
    # Repository scanning
    source_paths: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/tests/*", "*/test_*", "*/__pycache__/*", "*/.*"
    ])
    
    # Model discovery
    model_base_classes: List[str] = field(default_factory=lambda: [
        "pydantic.BaseModel", "BaseModel"
    ])
    validator_modules: List[str] = field(default_factory=list)
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    
    # Output
    output_directory: str = "generated_pipelines"
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        # Handle nested configurations
        if "database" in config_dict:
            config_dict["database"] = DatabaseConfig(**config_dict["database"])
        
        if "validation" in config_dict:
            validation_data = config_dict["validation"]
            if "ai" in validation_data:
                validation_data["ai"] = AIConfig(**validation_data["ai"])
            config_dict["validation"] = ValidationConfig(**validation_data)
            
        if "template" in config_dict:
            config_dict["template"] = TemplateConfig(**config_dict["template"])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.source_paths:
            raise ValueError("At least one source path must be specified")
        
        for path in self.source_paths:
            if not Path(path).exists():
                raise ValueError(f"Source path does not exist: {path}")
        
        # Validate output directory is writable
        output_path = Path(self.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        if not os.access(output_path, os.W_OK):
            raise ValueError(f"Output directory is not writable: {self.output_directory}")


def load_default_config() -> Config:
    """Load default configuration with sensible defaults."""
    return Config(
        source_paths=["."],
        database=DatabaseConfig(
            catalog="main",
            schema="dlt_validation",
            table_prefix="validated"
        ),
        validation=ValidationConfig(
            default_action="drop",
            strict_mode=True,
            generate_quarantine=True,
            generate_monitoring=True
        ),
        template=TemplateConfig(
            include_comments=True,
            include_metadata=True,
            generate_tests=True,
            spark_conf={
                "spark.databricks.dlt.validation.enabled": "true",
                "spark.databricks.dlt.expectations.enabled": "true"
            }
        )
    )

