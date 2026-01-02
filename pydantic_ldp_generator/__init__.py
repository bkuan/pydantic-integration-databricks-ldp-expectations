"""
Pydantic LDP Generator Library
=============================

A Python library that scans pydantic repositories to understand validators and data models,
and generates Databricks Lakeflow Declarative Pipeline templates with expectations on-demand.

Transform your Python data validation logic into production-ready Databricks DLT pipelines 
with intelligent SQL expectations. This library bridges the gap between Pydantic model 
definitions and Databricks data quality frameworks, automatically converting complex Python 
validators into equivalent SQL conditions using AI.

Key Capabilities:
- AI-Powered Validation Conversion: Automatically converts custom Python validators to SQL
- Flexible Usage: High-level Generator API or low-level component control
- Intelligent Analysis: Discovers models, analyzes validators, generates comprehensive rules
- Production-Ready: Complete pipeline systems with quarantine and monitoring
- Enterprise Features: Configurable AI models, database placement, validation actions

Main Components:
- ModelDiscovery: Scans repositories for pydantic models and custom validators
- ValidatorAnalyzer: Analyzes custom validators and field constraints with AI enhancement  
- TemplateGenerator: Creates complete DLT pipeline systems with quarantine and monitoring
- Generator: High-level API for single-model DLT code generation
- Configuration: Comprehensive control over AI, database, and validation behavior

Quick Start:
    from pydantic_ldp_generator import Generator, Config
    
    config = Config(source_paths=["domain", "validators"])
    generator = Generator(config, 'Customer')
    generator.create_ldp_template('customer_table')
    # Result: AI-enhanced DLT pipeline ready for Databricks deployment
    
Standalone Discovery:
    from pydantic_ldp_generator import ModelDiscovery, Config
    
    config = Config(source_paths=["domain", "validators"])  
    discovery = ModelDiscovery(config)
    models, validators = discovery.scan_repository()  # Scans all configured paths
    
    # Analyze discoveries
    for model_name, model_info in models.items():
        print(f"{model_info.name}: {len(model_info.fields)} fields")
        deps = discovery.get_model_dependencies(model_info.full_name)
        if deps: print(f"  Dependencies: {[d.split('.')[-1] for d in deps]}")
    
    # Find patterns
    email_models = {n: i for n, i in models.items() 
                   if any('email' in f.lower() for f in i.fields.keys())}
    print(f"Models with email fields: {len(email_models)}")
    
    # Result: Complete model and validator discovery with analysis
"""

from .discovery import ModelDiscovery
from .analyzer import ValidatorAnalyzer
from .generator import TemplateGenerator
from .config import Config, AIConfig, ValidationConfig, ValidatorAnalysisConfig, DatabaseConfig, TemplateConfig
from .model_selector import ModelSelector
from .model_generator import Generator
from .ai_cache import AIQueryCache, get_ai_cache, clear_ai_cache
from .ai_utils import AIQueryExecutor, AIPromptTemplates, AIResponseParser

__all__ = [
    "ModelDiscovery",
    "ValidatorAnalyzer", 
    "TemplateGenerator",
    "Config",
    "AIConfig",
    "ValidationConfig",
    "ValidatorAnalysisConfig",
    "DatabaseConfig", 
    "TemplateConfig",
    "ModelSelector",
    "Generator",
    "AIQueryCache",
    "get_ai_cache", 
    "clear_ai_cache",
    "AIQueryExecutor",
    "AIPromptTemplates", 
    "AIResponseParser"
]
