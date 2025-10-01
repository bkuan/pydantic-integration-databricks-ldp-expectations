"""
Template generation module for creating Databricks Lakeflow Declarative Pipeline templates.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, BaseLoader
import logging

from .discovery import ModelInfo, ValidatorInfo
from .analyzer import ModelValidationSchema, ValidationRule
from .config import Config


class TemplateGenerator:
    """Generates Databricks DLT pipeline templates from pydantic models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self.jinja_env.filters['snake_case'] = self._to_snake_case
        self.jinja_env.filters['pascal_case'] = self._to_pascal_case
        
    def generate_pipeline(self, 
                         schemas: Dict[str, ModelValidationSchema],
                         source_tables: Dict[str, str] = None) -> Dict[str, str]:
        """
        Generate complete DLT pipeline templates.
        
        Args:
            schemas: Dictionary of model validation schemas
            source_tables: Mapping of model names to source table names
            
        Returns:
            Dictionary mapping filename to generated content
        """
        self.logger.info(f"Generating DLT pipelines for {len(schemas)} models")
        
        if source_tables is None:
            source_tables = {}
        
        generated_files = {}
        
        # Generate main pipeline file
        main_pipeline = self._generate_main_pipeline(schemas, source_tables)
        generated_files["main_pipeline.py"] = main_pipeline
        
        # Generate individual model pipelines if requested
        if len(schemas) > 1:
            for model_name, schema in schemas.items():
                model_pipeline = self._generate_model_pipeline(schema, source_tables.get(model_name))
                filename = f"{self._to_snake_case(model_name)}_pipeline.py"
                generated_files[filename] = model_pipeline
        
        # Generate quarantine pipeline
        if self.config.validation.generate_quarantine:
            quarantine_pipeline = self._generate_quarantine_pipeline(schemas)
            generated_files["quarantine_pipeline.py"] = quarantine_pipeline
        
        # Generate monitoring pipeline
        if self.config.validation.generate_monitoring:
            monitoring_pipeline = self._generate_monitoring_pipeline(schemas)
            generated_files["monitoring_pipeline.py"] = monitoring_pipeline
        
        # Generate configuration file
        config_file = self._generate_config_file(schemas, source_tables)
        generated_files["pipeline_config.yml"] = config_file
        
        # Generate README
        readme_file = self._generate_readme(schemas)
        generated_files["README.md"] = readme_file
        
        self.logger.info(f"Generated {len(generated_files)} pipeline files")
        return generated_files
    
    def _generate_main_pipeline(self, schemas: Dict[str, ModelValidationSchema], source_tables: Dict[str, str]) -> str:
        """Generate the main DLT pipeline file."""
        template = self.jinja_env.from_string('''"""
Auto-generated Databricks Lakeflow Declarative Pipeline
Generated from pydantic models on {{ generation_date }}

Models included:
{% for schema in schemas.values() %}
- {{ schema.model_name }} ({{ schema.rules|length }} validation rules)
{% endfor %}
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *

{% if config.template.spark_conf %}
# Spark configuration
{% for key, value in config.template.spark_conf.items() %}
spark.conf.set("{{ key }}", "{{ value }}")
{% endfor %}

{% endif %}
{% for schema in schemas.values() %}

# ===============================================================================
# {{ schema.model_name.upper() }} PIPELINE
# ===============================================================================

{% set source_table = source_tables.get(schema.model_name, 'raw_' + schema.model_name|snake_case) %}
@dlt.table(
    comment="Validated {{ schema.model_name }} records",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true",
        "validation.source_model": "{{ schema.model_path }}",
        "validation.rules_count": "{{ schema.rules|length }}"
    }
)
{% for rule in schema.rules %}
@dlt.expect_or_{{ config.validation.default_action }}("{{ rule.expectation_name }}", "{{ rule.condition }}")
{% endfor %}
def {{ schema.model_name|snake_case }}_validated():
    """
    Validated {{ schema.model_name }} records with {{ schema.rules|length }} validation rules.
    
    Validation rules:
{% for rule in schema.rules %}
    - {{ rule.rule_type }}: {{ rule.description }}
{% endfor %}
    """
    return dlt.read("{{ source_table }}")

{% endfor %}
''')
        
        return template.render(
            schemas=schemas,
            source_tables=source_tables,
            config=self.config,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_model_pipeline(self, schema: ModelValidationSchema, source_table: Optional[str]) -> str:
        """Generate pipeline for a single model."""
        template = self.jinja_env.from_string('''"""
{{ schema.model_name }} Validation Pipeline
Auto-generated from pydantic model: {{ schema.model_path }}
Generated on {{ generation_date }}

Validation Rules: {{ schema.rules|length }}
{% for rule in schema.rules %}
- {{ rule.rule_type }}: {{ rule.description }}
{% endfor %}
"""

import dlt
from pyspark.sql import functions as F

{% set source_table_name = source_table or ('raw_' + schema.model_name|snake_case) %}

# ===============================================================================
# SOURCE DATA
# ===============================================================================

@dlt.table(comment="{{ schema.model_name }} raw data source")
def {{ schema.model_name|snake_case }}_source():
    """Source data for {{ schema.model_name }} validation."""
    return spark.table("{{ source_table_name }}")


# ===============================================================================
# VALIDATED DATA
# ===============================================================================

@dlt.table(
    comment="Validated {{ schema.model_name }} records",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
{% for rule in schema.rules %}
@dlt.expect_or_{{ config.validation.default_action }}("{{ rule.expectation_name }}", "{{ rule.condition }}")
{% endfor %}
def {{ schema.model_name|snake_case }}_validated():
    """{{ schema.model_name }} records that pass all validation rules."""
    return dlt.read("{{ schema.model_name|snake_case }}_source")


# ===============================================================================
# QUARANTINE DATA
# ===============================================================================

@dlt.table(comment="{{ schema.model_name }} records that failed validation")
def {{ schema.model_name|snake_case }}_quarantine():
    """{{ schema.model_name }} records quarantined due to validation failures."""
    base_df = dlt.read("{{ schema.model_name|snake_case }}_source")
    
    # Add validation flags
    validated_df = base_df
{% for rule in schema.rules %}
    validated_df = validated_df.withColumn(
        "valid_{{ rule.expectation_name }}", 
        F.expr("{{ rule.condition }}")
    )
{% endfor %}
    
    # Aggregate validation results
    all_validations = [
{% for rule in schema.rules %}
        F.col("valid_{{ rule.expectation_name }}"),
{% endfor %}
    ]
    
    validated_df = (
        validated_df
        .withColumn("is_valid_record", 
                   F.expr(" AND ".join([col._jc.toString() for col in all_validations])))
        .withColumn("failed_validations",
                   F.concat_ws(", ",
{% for rule in schema.rules %}
                              F.when(~F.col("valid_{{ rule.expectation_name }}"), 
                                   F.lit("{{ rule.expectation_name }}")),
{% endfor %}
                              ))
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )
    
    # Return only invalid records
    return validated_df.filter(~F.col("is_valid_record"))
''')
        
        return template.render(
            schema=schema,
            source_table=source_table,
            config=self.config,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_quarantine_pipeline(self, schemas: Dict[str, ModelValidationSchema]) -> str:
        """Generate quarantine pipeline for all models."""
        template = self.jinja_env.from_string('''"""
Quarantine Pipeline for All Models
Auto-generated quarantine management system
Generated on {{ generation_date }}
"""

import dlt
from pyspark.sql import functions as F

{% for schema in schemas.values() %}

@dlt.table(comment="Quarantined {{ schema.model_name }} records")
def quarantine_{{ schema.model_name|snake_case }}():
    """
    Records from {{ schema.model_name }} that failed validation.
    Provides detailed failure analysis for data quality monitoring.
    """
    base_df = dlt.read("{{ schema.model_name|snake_case }}_source")
    
    # Validation flags for each rule
{% for rule in schema.rules %}
    base_df = base_df.withColumn("valid_{{ rule.expectation_name }}", 
                                F.expr("{{ rule.condition }}"))
{% endfor %}
    
    # Overall validation result
    is_valid = F.expr(" AND ".join([
{% for rule in schema.rules %}
        "valid_{{ rule.expectation_name }}",
{% endfor %}
    ]))
    
    # Failed validation details
    failed_validations = F.concat_ws(", ",
{% for rule in schema.rules %}
        F.when(~F.col("valid_{{ rule.expectation_name }}"), 
               F.lit("{{ rule.rule_type }}: {{ rule.description }}")),
{% endfor %}
    )
    
    return (
        base_df
        .withColumn("is_valid_record", is_valid)
        .withColumn("failed_validations", failed_validations)
        .withColumn("quarantine_timestamp", F.current_timestamp())
        .withColumn("model_name", F.lit("{{ schema.model_name }}"))
        .filter(~F.col("is_valid_record"))
    )

{% endfor %}

@dlt.table(comment="Combined quarantine records from all models")
def quarantine_summary():
    """Combined view of all quarantined records across models."""
    quarantine_dfs = [
{% for schema in schemas.values() %}
        dlt.read("quarantine_{{ schema.model_name|snake_case }}"),
{% endfor %}
    ]
    
    # Union all quarantine tables
    combined_df = quarantine_dfs[0]
    for df in quarantine_dfs[1:]:
        combined_df = combined_df.unionByName(df, allowMissingColumns=True)
    
    return combined_df
''')
        
        return template.render(
            schemas=schemas,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_monitoring_pipeline(self, schemas: Dict[str, ModelValidationSchema]) -> str:
        """Generate monitoring and metrics pipeline."""
        template = self.jinja_env.from_string('''"""
Data Quality Monitoring Pipeline
Auto-generated monitoring and metrics system
Generated on {{ generation_date }}
"""

import dlt
from pyspark.sql import functions as F

@dlt.table(comment="Data quality metrics summary")
def data_quality_metrics():
    """
    Comprehensive data quality metrics across all validated models.
    Provides insights into validation success rates and common failures.
    """
    
    metrics = []
    
{% for schema in schemas.values() %}
    # {{ schema.model_name }} metrics
    {{ schema.model_name|snake_case }}_total = dlt.read("{{ schema.model_name|snake_case }}_source").count()
    {{ schema.model_name|snake_case }}_valid = dlt.read("{{ schema.model_name|snake_case }}_validated").count()
    {{ schema.model_name|snake_case }}_invalid = dlt.read("quarantine_{{ schema.model_name|snake_case }}").count()
    
    metrics.append({
        "model_name": "{{ schema.model_name }}",
        "total_records": {{ schema.model_name|snake_case }}_total,
        "valid_records": {{ schema.model_name|snake_case }}_valid,
        "invalid_records": {{ schema.model_name|snake_case }}_invalid,
        "validation_rate": {{ schema.model_name|snake_case }}_valid / {{ schema.model_name|snake_case }}_total if {{ schema.model_name|snake_case }}_total > 0 else 0.0,
        "total_rules": {{ schema.rules|length }},
        "measurement_timestamp": F.current_timestamp()
    })
    
{% endfor %}
    
    return spark.createDataFrame(metrics)


@dlt.table(comment="Validation failure analysis")  
def failure_analysis():
    """
    Analysis of validation failures by type and frequency.
    Helps identify common data quality issues.
    """
    
    quarantine_df = dlt.read("quarantine_summary")
    
    return (
        quarantine_df
        .select("model_name", "failed_validations", "quarantine_timestamp")
        .withColumn("failure_type", F.split(F.col("failed_validations"), ":")[0])
        .groupBy("model_name", "failure_type")
        .agg(
            F.count("*").alias("failure_count"),
            F.max("quarantine_timestamp").alias("last_failure")
        )
        .orderBy(F.desc("failure_count"))
    )


@dlt.table(comment="Daily data quality trends")
def daily_quality_trends():
    """
    Daily trends in data quality metrics.
    Useful for monitoring data quality over time.
    """
    
    metrics_df = dlt.read("data_quality_metrics")
    
    return (
        metrics_df
        .withColumn("date", F.to_date("measurement_timestamp"))
        .groupBy("date", "model_name")
        .agg(
            F.avg("validation_rate").alias("avg_validation_rate"),
            F.sum("total_records").alias("daily_total_records"),
            F.sum("valid_records").alias("daily_valid_records"),
            F.sum("invalid_records").alias("daily_invalid_records")
        )
        .orderBy("date", "model_name")
    )
''')
        
        return template.render(
            schemas=schemas,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _generate_config_file(self, schemas: Dict[str, ModelValidationSchema], source_tables: Dict[str, str]) -> str:
        """Generate pipeline configuration file."""
        config_data = {
            "pipeline_info": {
                "generated_date": datetime.now().isoformat(),
                "models_count": len(schemas),
                "total_validation_rules": sum(len(schema.rules) for schema in schemas.values())
            },
            "models": {
                schema.model_name: {
                    "model_path": schema.model_path,
                    "validation_rules": len(schema.rules),
                    "source_table": source_tables.get(schema.model_name, f"raw_{self._to_snake_case(schema.model_name)}"),
                    "dependencies": list(schema.dependencies)
                }
                for schema in schemas.values()
            },
            "validation_config": {
                "default_action": self.config.validation.default_action,
                "strict_mode": self.config.validation.strict_mode,
                "generate_quarantine": self.config.validation.generate_quarantine,
                "generate_monitoring": self.config.validation.generate_monitoring
            },
            "database_config": {
                "catalog": self.config.database.catalog,
                "schema": self.config.database.schema,
                "table_prefix": self.config.database.table_prefix
            }
        }
        
        import yaml
        return yaml.dump(config_data, default_flow_style=False, indent=2)
    
    def _generate_readme(self, schemas: Dict[str, ModelValidationSchema]) -> str:
        """Generate README documentation."""
        template = self.jinja_env.from_string('''# Auto-Generated Databricks DLT Pipeline

Generated on {{ generation_date }}

## Overview

This pipeline was automatically generated from {{ schemas|length }} pydantic models with a total of {{ total_rules }} validation rules.

## Models and Validation Rules

{% for schema in schemas.values() %}
### {{ schema.model_name }}

- **Source Model**: `{{ schema.model_path }}`
- **Validation Rules**: {{ schema.rules|length }}
- **Dependencies**: {{ schema.dependencies|length }}

{% if schema.rules %}
#### Validation Rules:
{% for rule in schema.rules %}
- **{{ rule.rule_type|title }}**: {{ rule.description }}
  - Condition: `{{ rule.condition }}`
  - Action: {{ rule.action }}
{% endfor %}
{% endif %}

{% endfor %}

## Pipeline Structure

### Main Components

1. **Source Tables**: Raw data ingestion
2. **Validation Tables**: Data with applied expectations  
3. **Quarantine Tables**: Failed validation records
4. **Monitoring Tables**: Data quality metrics

### Generated Files

- `main_pipeline.py` - Main DLT pipeline with all models
- `quarantine_pipeline.py` - Quarantine management system
- `monitoring_pipeline.py` - Data quality monitoring
- `pipeline_config.yml` - Configuration metadata

## Usage

### Deployment

```bash
# Deploy using Databricks Asset Bundles
databricks bundle deploy

# Run the pipeline
databricks bundle run
```

### Monitoring

Access the following tables for monitoring:

- `data_quality_metrics` - Overall quality metrics
- `failure_analysis` - Breakdown of validation failures  
- `daily_quality_trends` - Quality trends over time

## Configuration

Pipeline behavior can be configured through:

- **Validation Action**: {{ config.validation.default_action }}
- **Strict Mode**: {{ config.validation.strict_mode }}
- **Generate Quarantine**: {{ config.validation.generate_quarantine }}
- **Generate Monitoring**: {{ config.validation.generate_monitoring }}

## Support

This pipeline was generated by the Pydantic DLT Generator library.
For issues or customization, refer to the original pydantic models.
''')
        
        total_rules = sum(len(schema.rules) for schema in schemas.values())
        
        return template.render(
            schemas=schemas,
            config=self.config,
            total_rules=total_rules,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def save_generated_files(self, generated_files: Dict[str, str], output_directory: str = None) -> str:
        """Save generated files to disk."""
        if output_directory is None:
            output_directory = self.config.output_directory
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for filename, content in generated_files.items():
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Generated: {file_path}")
        
        self.logger.info(f"All files saved to: {output_path}")
        return str(output_path)
    
    def _to_snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return ''.join(word.capitalize() for word in name.split('_'))

