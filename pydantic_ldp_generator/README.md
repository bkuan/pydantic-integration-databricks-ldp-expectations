# Pydantic LDP Generator

**Transform your Pydantic data models into production-ready Databricks Lakeflow Declarative Pipelines**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Databricks](https://img.shields.io/badge/platform-Databricks-orange.svg)](https://www.databricks.com/)

---

## Overview

The **Pydantic LDP Generator** is a Python library that bridges the gap between Pydantic model definitions and Databricks data quality frameworks. It automatically scans your Python repositories for Pydantic models and validators, analyzes validation logic, and generates complete Databricks DLT (Delta Live Tables) pipelines with SQL-based data quality expectations.

### Key Features

- **🤖 AI-Powered Validation Conversion**: Automatically converts Python validators to equivalent Databricks SQL conditions using AI
- **🔍 Intelligent Model Discovery**: Scans codebases to find Pydantic models and standalone validators
- **📊 Complete Pipeline Generation**: Creates production-ready DLT pipelines with expectations
- **🔄 Quarantine Handling**: Generates quarantine pipelines for failed validation records
- **📈 Built-in Monitoring**: Creates data quality monitoring and metrics pipelines
- **⚙️ Highly Configurable**: Extensive configuration for AI models, validation behavior, and output

---

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install pydantic jinja2 pyyaml

# For Databricks deployment, ensure pyspark is available
```

---

## Quick Start

### Basic Usage

```python
from pydantic_ldp_generator import Generator, Config

# Configure source paths where your Pydantic models are located
config = Config(source_paths=["./domain", "./validators"])

# Generate DLT code for a specific model
generator = Generator(config, 'Customer')
generator.create_ldp_template('customer_source_table')
```

### Output

The generator produces a complete Databricks DLT pipeline:

```python
from pyspark import pipelines as dp

def get_customer_validation_rules():
    return {
        "required_field_email": "email IS NOT NULL",
        "format_email": "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
        "custom_validator_zipcode": "zip RLIKE '^[0-9]{5}(-[0-9]{4})?$'",
        # ... more rules
    }

@dp.table(comment="Validated Customer records")
@dp.expect_all_or_drop(get_customer_validation_rules())
def customer_validated():
    return dp.read("customer_source_table")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pydantic LDP Generator                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   Model     │───▶│   Validator      │───▶│   Template    │  │
│  │  Discovery  │    │    Analyzer      │    │   Generator   │  │
│  └─────────────┘    └──────────────────┘    └───────────────┘  │
│         │                    │                      │          │
│         ▼                    ▼                      ▼          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Scans for  │    │  AI-powered      │    │  Generates    │  │
│  │  Pydantic   │    │  Python→SQL      │    │  DLT code &   │  │
│  │  Models     │    │  conversion      │    │  pipelines    │  │
│  └─────────────┘    └──────────────────┘    └───────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. ModelDiscovery

Scans Python repositories to discover Pydantic models and validators.

```python
from pydantic_ldp_generator import ModelDiscovery, Config

config = Config(source_paths=["./domain", "./validators"])
discovery = ModelDiscovery(config)

# Scan all configured paths
models, validators = discovery.scan_repository()

# Explore discovered models
for model_name, model_info in models.items():
    print(f"{model_info.name}: {len(model_info.fields)} fields")
    print(f"  Field validators: {model_info.field_validators}")
    print(f"  Model validators: {model_info.model_validators}")

# Find model dependencies
deps = discovery.get_model_dependencies(model_info.full_name)
```

### 2. ValidatorAnalyzer

Analyzes Pydantic validators and converts them to SQL validation rules.

```python
from pydantic_ldp_generator import ValidatorAnalyzer, Config

config = Config(source_paths=["./domain"])
analyzer = ValidatorAnalyzer(config)

# Analyze a discovered model
schema = analyzer.analyze_model(model_info, validators)

# Get validation rules
for rule in schema.rules:
    print(f"{rule.rule_type}: {rule.field_name}")
    print(f"  SQL: {rule.condition}")
    print(f"  Description: {rule.description}")
```

**Supported Validation Types:**

| Rule Type | Description |
|-----------|-------------|
| `required_field` | Fields marked as required (no default value) |
| `basic_type_check` | Python type to SQL type validation |
| `custom_type` | Custom Pydantic types (Email, ZipCode, etc.) |
| `custom_validator` | Standalone validator functions |
| `field_validator` | `@field_validator` decorated methods |
| `model_validator` | `@model_validator` decorated methods (cross-field) |

### 3. Generator (High-Level API)

The simplest way to generate DLT code for a single model.

```python
from pydantic_ldp_generator import Generator, Config

config = Config(source_paths=["./domain", "./validators"])
generator = Generator(config, 'Customer')

# Generate and save DLT template
generator.create_ldp_template('customer_raw')

# Generate complete pipeline system
generator.create_complete_system(
    source_table='customer_raw',
    output_dir='./generated_pipelines'
)

# Show what will be generated
generator.show_summary()
```

**Generated Files:**

- `main_pipeline.py` - Main DLT pipeline with expectations
- `quarantine_pipeline.py` - Pipeline for failed validation records
- `monitoring_pipeline.py` - Data quality metrics and trends
- `validation_rules.py` - Reusable validation rules module
- `pipeline_config.yml` - Configuration metadata
- `README.md` - Generated documentation

### 4. ModelSelector (Interactive Analysis)

Explore and analyze models interactively.

```python
from pydantic_ldp_generator import ModelSelector, Config

config = Config(source_paths=["./domain"])
selector = ModelSelector(config, 'Customer')

# Show complete analysis
selector.show_analysis()

# Get data programmatically
fields = selector.get_fields()
rules = selector.get_validation_rules()
custom_rules = selector.get_custom_validator_rules()

# Get summary
summary = selector.get_summary()
```

### 5. TemplateGenerator (Multi-Model Pipelines)

Generate pipelines for multiple models at once.

```python
from pydantic_ldp_generator import ModelDiscovery, ValidatorAnalyzer, TemplateGenerator, Config

config = Config(source_paths=["./domain"])
discovery = ModelDiscovery(config)
analyzer = ValidatorAnalyzer(config)
template_gen = TemplateGenerator(config)

# Discover all models
models, validators = discovery.scan_repository()

# Analyze all models
schemas = {}
for model_name, model_info in models.items():
    schemas[model_info.name] = analyzer.analyze_model(model_info, validators)

# Generate complete pipeline system
source_tables = {
    'Customer': 'raw_customers',
    'Order': 'raw_orders'
}
generated_files = template_gen.generate_pipeline(schemas, source_tables)

# Save to disk
template_gen.save_generated_files(generated_files, './output')
```

---

## Configuration

### Basic Configuration

```python
from pydantic_ldp_generator import Config

config = Config(
    # Where to find Pydantic models and validators
    source_paths=["./domain", "./validators"],
    
    # Patterns to exclude from scanning
    exclude_patterns=["*/tests/*", "*/__pycache__/*"],
    
    # Output directory for generated files
    output_directory="generated_pipelines"
)
```

### AI Configuration

```python
from pydantic_ldp_generator import Config, AIConfig, ValidationConfig

config = Config(
    source_paths=["./domain"],
    validation=ValidationConfig(
        ai=AIConfig(
            enabled=True,                        # Enable AI conversion
            model="databricks-claude-sonnet-4",  # AI model to use
            use_mock_responses=True,             # Use mocks for dev/testing
            fallback_enabled=True,               # Fall back to basic rules if AI fails
            timeout_seconds=30,                  # AI query timeout
            enable_cache=True,                   # Cache AI responses
            cache_size_limit=1000                # Max cached responses
        )
    )
)
```

### Database Configuration

```python
from pydantic_ldp_generator import Config, DatabaseConfig

config = Config(
    source_paths=["./domain"],
    database=DatabaseConfig(
        catalog="main",
        schema="validated_data",
        table_prefix="dlt"
    )
)
```

### Validation Behavior

```python
from pydantic_ldp_generator import Config, ValidationConfig

config = Config(
    source_paths=["./domain"],
    validation=ValidationConfig(
        default_action="drop",        # drop, fail, or warn
        strict_mode=True,
        generate_quarantine=True,     # Generate quarantine pipeline
        generate_monitoring=True      # Generate monitoring pipeline
    )
)
```

### YAML Configuration

Save and load configuration from YAML files:

```python
# Save
config.save_to_file('pipeline_config.yml')

# Load
config = Config.from_file('pipeline_config.yml')
```

Example YAML:

```yaml
source_paths:
  - ./domain
  - ./validators
exclude_patterns:
  - "*/tests/*"
  - "*/__pycache__/*"
database:
  catalog: main
  schema: validated_data
  table_prefix: dlt
validation:
  default_action: drop
  generate_quarantine: true
  ai:
    enabled: true
    model: databricks-claude-sonnet-4
output_directory: generated_pipelines
```

---

## AI-Powered Validation Conversion

The library uses Databricks `ai_query()` to intelligently convert Python validation logic to SQL:

### How It Works

1. **Python Validator** (your code):
```python
def email_validator(email: str) -> str:
    if '@' not in email or '.' not in email.split('@')[1]:
        raise ValueError("Invalid email format")
    if len(email) > 254:
        raise ValueError("Email too long")
    return email.lower()
```

2. **AI Conversion** → **SQL Condition**:
```sql
email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
  AND LENGTH(email) <= 254
```

### Supported Validators

- **Field Validators**: `@field_validator` decorated methods
- **Model Validators**: `@model_validator` for cross-field validation
- **Standalone Validators**: Functions ending with `_validator`
- **Custom Types**: Pydantic custom types (Email, ZipCode, etc.)

### AI Caching

AI responses are cached to improve performance:

```python
from pydantic_ldp_generator import get_ai_cache, clear_ai_cache

# Get cache statistics
cache = get_ai_cache()
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")

# Clear cache when needed
clear_ai_cache()
```

---

## Generated Output Examples

### Main Pipeline

```python
from pyspark import pipelines as dp

def get_customer_rules():
    return {
        "required_field_name": "name IS NOT NULL",
        "basic_type_check_name": "typeof(name) = 'string' AND trim(COALESCE(name, '')) != ''",
        "custom_validator_email": "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
        "model_validator_contact_info": "(phone IS NOT NULL) OR (email IS NOT NULL)"
    }

@dp.table(
    comment="Validated Customer records",
    table_properties={
        "quality": "gold",
        "validation.pattern": "reusable_expectations"
    }
)
@dp.expect_all_or_drop(get_customer_rules())
def customer_validated():
    return dp.read("customer_raw")
```

### Quarantine Pipeline

Records that fail validation are routed to quarantine tables with detailed failure information:

```python
@dp.table(comment="Quarantined Customer records")
def quarantine_customer():
    # Returns failed records with:
    # - is_valid_record: false
    # - failed_validations: comma-separated list of failed rules
    # - quarantine_timestamp: when the record was quarantined
```

### Monitoring Pipeline

Track data quality metrics over time:

```python
@dp.table(comment="Data quality metrics")
def data_quality_metrics():
    # Returns metrics including:
    # - total_records, valid_records, invalid_records
    # - validation_rate
    # - total_rules

@dp.table(comment="Daily quality trends")
def daily_quality_trends():
    # Daily aggregated quality metrics
```

---

## Example: Customer Model

### Source Model

```python
# domain/customer_model.py
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

class Customer(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    zip_code: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()
    
    @model_validator(mode='after')
    def validate_contact_info(self):
        if not self.phone and not self.email:
            raise ValueError('Must provide phone or email')
        return self
```

### Generated DLT Code

```python
from pyspark import pipelines as dp

def get_customer_rules():
    return {
        "required_field_name": "name IS NOT NULL",
        "required_field_email": "email IS NOT NULL",
        "required_field_zip_code": "zip_code IS NOT NULL",
        "field_validator_email": "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
        "model_validator_contact_info": "(phone IS NOT NULL) OR (email IS NOT NULL)",
    }

@dp.table(comment="Validated Customer records")
@dp.expect_all_or_drop(get_customer_rules())
def customer_validated():
    return dp.read("customer_raw")
```

---

## API Reference

### Main Classes

| Class | Description |
|-------|-------------|
| `Generator` | High-level API for single-model DLT generation |
| `ModelSelector` | Interactive model analysis and exploration |
| `ModelDiscovery` | Scans repositories for Pydantic models |
| `ValidatorAnalyzer` | Converts validators to SQL rules |
| `TemplateGenerator` | Generates DLT pipeline templates |
| `Config` | Configuration management |

### Configuration Classes

| Class | Description |
|-------|-------------|
| `AIConfig` | AI query settings |
| `DatabaseConfig` | Databricks catalog/schema settings |
| `ValidationConfig` | Validation behavior settings |
| `TemplateConfig` | Template generation settings |

### Utility Functions

| Function | Description |
|----------|-------------|
| `get_ai_cache()` | Get the global AI cache instance |
| `clear_ai_cache()` | Clear all cached AI responses |
| `load_default_config()` | Load sensible default configuration |

---

## Development & Testing

### Mock Mode

For development without Databricks access:

```python
config = Config(
    source_paths=["./domain"],
    validation=ValidationConfig(
        ai=AIConfig(
            use_mock_responses=True  # Use mock AI responses
        )
    )
)
```

### Running in Databricks

For production deployment:

```python
config = Config(
    source_paths=["./domain"],
    validation=ValidationConfig(
        ai=AIConfig(
            use_mock_responses=False,  # Use real ai_query()
            model="databricks-claude-sonnet-4"
        )
    )
)
```

---

## Best Practices

1. **Organize Validators**: Keep standalone validators in dedicated modules for easy discovery
2. **Use Descriptive Names**: Name validators with `_validator` suffix for automatic detection
3. **Leverage Model Validators**: Use `@model_validator` for complex cross-field validation
4. **Configure AI Appropriately**: Use mock responses during development
5. **Review Generated SQL**: Always review AI-generated SQL conditions before production
6. **Use Quarantine Pipelines**: Enable quarantine to capture and analyze failed records
7. **Monitor Data Quality**: Use generated monitoring pipelines for ongoing quality tracking

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

---

## License

[Your License Here]

---

## Support

For questions, issues, or feature requests, please open an issue on GitHub.

