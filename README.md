# Pydantic LDP Generator

A Python library that automatically converts Pydantic data models and validation logic into production-ready Databricks Lakeflow Declarative Pipeline (LDP) templates with intelligent SQL expectations.

## Overview

The Pydantic LDP Generator bridges the gap between Python data validation logic from Pydantic and Databricks Lakeflow Declarative Pipeline. It scans Pydantic repositories to understand models and validators, then generates templates for LDP pipelines with AI-powered validation conversion to create LDP expectations.

<img width="1227" height="543" alt="image" src="https://github.com/user-attachments/assets/35f4eaab-c9b0-43d1-a7b0-a8b4c6917950" />


## Key Features

- **AI-Powered Validation Conversion**: Automatically converts custom Python validators to SQL using Databricks AI
- **Intelligent Model Discovery**: Scans repositories to find Pydantic models and custom validators
- **Pipeline Generation**: Creates starting point for LDP pipeline with quarantine and monitoring
- **Flexible Configuration**: Supports various validation actions, database settings


## Quick Start

### Basic Usage

```python
from pydantic_ldp_generator import Generator, Config

# Configure source paths to pydantic models and validators
config = Config(source_paths=["domain", "validators_lib"])

# Generate LDP code for a specific model
generator = Generator(config, 'Customer')
generator.create_ldp_template('customer_table')
```

### Generate LDP pipeline code

```python
generator.create_complete_system('customer_table', 'generated_pipeline')
```

## Architecture

### Core Components

- **ModelDiscovery**: Scans repositories for Pydantic models and custom validators
- **ValidatorAnalyzer**: Analyzes validation logic and converts to SQL expectations
- **TemplateGenerator**: Creates complete LDP pipeline systems
- **Generator**: High-level API for single-model LDP code generation
- **ModelSelector**: Interactive model analysis and selection

### Configuration

The library supports comprehensive configuration through the `Config` class:

```python
from pydantic_ldp_generator import Config, AIConfig, ValidationConfig, DatabaseConfig

config = Config(
    source_paths=["domain", "validators_lib"],
    database=DatabaseConfig(
        catalog="main",
        schema="LDP_validation",
        table_prefix="validated"
    ),
    validation=ValidationConfig(
        default_action="drop",
        strict_mode=True,
        generate_quarantine=True,
        generate_monitoring=True,
        ai=AIConfig(
            enabled=True,
            model="databricks-claude-sonnet-4",
            timeout_seconds=30
        )
    )
)
```

## Usage Examples

### Model Analysis

```python
from pydantic_ldp_generator import ModelSelector, Config

config = Config(source_paths=["domain", "validators_lib"])
selector = ModelSelector(config, 'Customer')
selector.show_analysis()
```

### Custom Validator Integration

The library automatically discovers and converts custom validators:

```python
# Custom validator in your codebase
def email_validator(email: str) -> str:
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        raise ValueError("Invalid email format")
    return email

# Automatically converted to SQL:
# @LDP.expect_or_drop("email_format", "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'")
```

## Generated Files

When creating the LDP code, the generator produces:

- `main_pipeline.py` - Main LDP pipeline with all models
- `quarantine_pipeline.py` - Quarantine management system
- `monitoring_pipeline.py` - Data quality monitoring
- `pipeline_config.yml` - Configuration metadata

## Validation Rules

The library automatically generates validation rules for:

- **Type Validation**: String, integer, float, boolean, datetime fields
- **Format Validation**: Email, URL, phone number patterns
- **Range Validation**: Min/max length, numeric ranges
- **Custom Validators**: AI-converted Python validation logic
- **Required Fields**: Null checks and mandatory field validation

## AI Integration

The library leverages Databricks AI_query to convert complex Python validators to SQL:

- **Model**: Configurable AI model (default: databricks-claude-sonnet-4)
- **Timeout**: Configurable timeout for AI queries
- **Fallback**: Basic validation rules if AI conversion fails
- **Custom Types**: Automatic detection and conversion of custom field types

## Database Configuration

Supports flexible database placement:

```python
database_config = DatabaseConfig(
    catalog="main",
    schema="LDP_validation", 
    table_prefix="validated"
)
```

## Validation Actions

Configure how validation failures are handled:

- **drop**: Remove invalid records (default)
- **fail**: Fail the pipeline on validation errors
- **warn**: Log warnings but continue processing

## Advanced Features

### Quarantine Management

Failed records are automatically quarantined with detailed failure analysis:

```python
# Quarantine pipeline includes:
# - Failed validation details
# - Timestamp tracking
# - Failure type analysis
# - Model-specific quarantine tables
```

### Monitoring and Metrics

Comprehensive data quality monitoring:

```python
# Monitoring tables include:
# - data_quality_metrics: Overall quality metrics
# - failure_analysis: Validation failure breakdown
# - daily_quality_trends: Quality trends over time
```

### Template Customization

Extensive template customization options:

```python
template_config = TemplateConfig(
    include_comments=True,
    include_metadata=True,
    generate_tests=True,
    spark_conf={
        "spark.databricks.LDP.validation.enabled": "true",
        "spark.databricks.LDP.expectations.enabled": "true"
    }
)
```

## Requirements

- Python 3.8+
- Pydantic 2.0+
- Databricks Runtime 13.0+
- Jinja2 (for template generation)
- PyYAML (for configuration)


For issues, questions, or contributions, please refer to the project repository or contact the maintainers.
