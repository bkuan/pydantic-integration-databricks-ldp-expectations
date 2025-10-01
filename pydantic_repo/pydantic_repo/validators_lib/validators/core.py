import re
from datetime import date

def email_validator(v: str) -> str:
    """
    Custom email validator with business-specific rules.
    More restrictive than basic regex - ensures proper domain structure.
    """
    if not isinstance(v, str):
        raise ValueError("email must be a string")
    
    # Basic format check
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, v):
        raise ValueError("email format is invalid")
    
    # Business rules
    if len(v) > 254:  # RFC 5321 limit
        raise ValueError("email is too long (max 254 characters)")
        
    local, domain = v.split('@')
    if len(local) > 64:  # RFC 5321 limit for local part
        raise ValueError("email local part is too long (max 64 characters)")
        
    # Reject common test/invalid domains
    invalid_domains = {'example.com', 'test.com', 'invalid.com', 'localhost'}
    if domain.lower() in invalid_domains:
        raise ValueError(f"email domain '{domain}' is not allowed")
    
    # Must have at least 2 parts in domain (e.g., domain.com)
    domain_parts = domain.split('.')
    if len(domain_parts) < 2 or any(len(part) == 0 for part in domain_parts):
        raise ValueError("email domain must have valid structure")
    
    return v.lower()  # Normalize to lowercase

def zipcode_validator(v: str) -> str:
    if not re.fullmatch(r"\d{5}(-\d{4})?", v):
        raise ValueError("zipcode must be 5 digits or ZIP+4")
    return v

def sex_validator(v: str) -> str:
    if v not in {"M", "F"}:
        raise ValueError("sex must be 'M' or 'F'")
    return v

def past_date_validator(v: date) -> date:
    if v > date.today():
        raise ValueError("date cannot be in the future")
    return v
