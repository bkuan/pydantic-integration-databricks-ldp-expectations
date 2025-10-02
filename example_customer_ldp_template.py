"""
Auto-generated Databricks Lakeflow Declarative Pipeline
Generated from pydantic models on 2025-10-01 02:20:16

Models included:
- Customer (20 validation rules)
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *


# ===============================================================================
# CUSTOMER PIPELINE
# ===============================================================================

@dlt.table(
    comment="Validated Customer records",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true",
        "validation.source_model": "/.Workspace.Users.bernie.kuan@databricks.com.pydantic_demo.pydantic_repo.domain.customer_model.Customer",
        "validation.rules_count": "20"
    }
)
@dlt.expect_or_drop("required_customer_id", "customer_id IS NOT NULL")
@dlt.expect_or_drop("type_check_customer_id", "customer_id IS NOT NULL AND customer_id RLIKE '^-?[0-9]+$'")
@dlt.expect_or_drop("required_email", "email IS NOT NULL")
@dlt.expect_or_drop("required_name", "name IS NOT NULL")
@dlt.expect_or_drop("required_sex", "sex IS NOT NULL")
@dlt.expect_or_drop("required_signup_date", "signup_date IS NOT NULL")
@dlt.expect_or_drop("required_street", "street IS NOT NULL")
@dlt.expect_or_drop("type_check_street", "street IS NOT NULL AND trim(street) != ''")
@dlt.expect_or_drop("required_city", "city IS NOT NULL")
@dlt.expect_or_drop("type_check_city", "city IS NOT NULL AND trim(city) != ''")
@dlt.expect_or_drop("required_state", "state IS NOT NULL")
@dlt.expect_or_drop("type_check_state", "state IS NOT NULL AND trim(state) != ''")
@dlt.expect_or_drop("required_zip", "zip IS NOT NULL")
@dlt.expect_or_drop("ai_custom_email_1", "email RLIKE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'")
@dlt.expect_or_drop("ai_custom_email_2", "LENGTH(email) <= 254")
@dlt.expect_or_drop("ai_custom_email_3", "LENGTH(SPLIT(email, '@')[0]) <= 64")
@dlt.expect_or_drop("ai_custom_email_4", "email NOT RLIKE '@(example|test|invalid|localhost)\.(com|org|net)'")
@dlt.expect_or_drop("ai_custom_sex_1", "sex IN ('M', 'F')")
@dlt.expect_or_drop("ai_custom_signup_date_1", "signup_date <= CURRENT_DATE()")
@dlt.expect_or_drop("ai_custom_zip_1", "zip RLIKE '^[0-9]{5}(-[0-9]{4})?$'")
def customer_validated():
    """
    Validated Customer records with 20 validation rules.
    
    Validation rules:
    - required: customer_id is required
    - type_check: customer_id must be a valid integer
    - required: email is required
    - required: name is required
    - required: sex is required
    - required: signup_date is required
    - required: street is required
    - type_check: street must be a non-empty string
    - required: city is required
    - type_check: city must be a non-empty string
    - required: state is required
    - type_check: state must be a non-empty string
    - required: zip is required
    - ai_custom_validator: Valid email format
    - ai_custom_validator: Email length within RFC 5321 limits
    - ai_custom_validator: Local part within limits
    - ai_custom_validator: Exclude test domains
    - ai_custom_validator: Must be M or F
    - ai_custom_validator: Date must be in the past
    - ai_custom_validator: Valid 5-digit or ZIP+4 format
    """
    return dlt.read("customer_table")

