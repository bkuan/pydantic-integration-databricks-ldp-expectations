import re
from datetime import date, datetime

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

def past_date_validator(v: str) -> str:
    d = datetime.strptime(v, "%Y-%m-%d").date()

    # Validate it is not in the future
    if d > date.today():
        raise ValueError("date cannot be in the future")
    return v

def customer_contact_completeness_validator(customer_data: dict) -> dict:
    """
    Complex validator demonstrating 2-3 depth dependencies using existing Customer fields.
    
    Uses: email, phone, street, city, state, zip, sex
    
    Dependencies (2-3 depth):
    - email_validator (depth 1)
    - zipcode_validator (depth 1) 
    - address_completeness_validator (depth 2)
    - contact_method_validator (depth 2)
    - geographic_service_validator (depth 3)
    """
    
    # Call existing validators (creates dependencies)
    if customer_data.get('email'):
        customer_data['email'] = email_validator(customer_data['email'])
    
    if customer_data.get('zip'):
        customer_data['zip'] = zipcode_validator(customer_data['zip'])
    
    # Call complex validators (creates depth 2-3 dependencies)
    customer_data = address_completeness_validator(customer_data)
    customer_data = contact_method_validator(customer_data)
    customer_data = geographic_service_validator(customer_data)
    
    # Complex business logic for AI conversion
    # Rule: Customers must have at least 2 contact methods
    contact_methods = 0
    
    if customer_data.get('email'):
        contact_methods += 1
    
    if customer_data.get('phone'):
        contact_methods += 1
        
    # Complete address counts as contact method
    if all(customer_data.get(f) for f in ['street', 'city', 'state', 'zip']):
        contact_methods += 1
    
    if contact_methods < 2:
        raise ValueError("Customer must have at least 2 contact methods (email, phone, or complete address)")
    
    # Cross-field validation with sex and contact preferences
    if customer_data.get('sex') == 'M':
        # Male customers: prefer email for digital communications
        if not customer_data.get('email'):
            raise ValueError("Email strongly recommended for male customers")
    
    return customer_data

def address_completeness_validator(customer_data: dict) -> dict:
    """
    Address completeness validator (depth 2).
    
    Dependencies:
    - zipcode_validator (depth 1)
    - city_state_consistency_validator (depth 2)
    """
    
    address_fields = ['street', 'city', 'state', 'zip']
    provided_fields = [f for f in address_fields if customer_data.get(f)]
    
    # If any address field provided, require complete address
    if provided_fields and len(provided_fields) < len(address_fields):
        missing = [f for f in address_fields if not customer_data.get(f)]
        raise ValueError(f"Incomplete address - missing: {', '.join(missing)}")
    
    # If complete address provided, validate components
    if len(provided_fields) == len(address_fields):
        # Call zipcode validator (creates dependency)
        customer_data['zip'] = zipcode_validator(customer_data['zip'])
        
        # Call city-state consistency validator (creates depth 2 dependency)
        customer_data = city_state_consistency_validator(customer_data)
    
    return customer_data

def contact_method_validator(customer_data: dict) -> dict:
    """
    Contact method validator (depth 2).
    
    Dependencies:
    - email_validator (depth 1) 
    - phone_format_validator (depth 1)
    """
    
    # Validate email if provided (creates dependency)
    if customer_data.get('email'):
        customer_data['email'] = email_validator(customer_data['email'])
    
    # Validate phone if provided (creates dependency) 
    if customer_data.get('phone'):
        customer_data['phone'] = phone_format_validator(customer_data['phone'])
    
    # Business rule: At least one primary contact method required
    has_email = bool(customer_data.get('email'))
    has_phone = bool(customer_data.get('phone'))
    
    if not (has_email or has_phone):
        raise ValueError("Must provide either email or phone for primary contact")
    
    return customer_data

def geographic_service_validator(customer_data: dict) -> dict:
    """
    Geographic service validator (depth 3, called by main validator).
    
    Dependencies:
    - address_completeness_validator (depth 2)
    - zipcode_validator (depth 1)
    """
    
    # Call address validator first (creates deeper dependency chain)
    customer_data = address_completeness_validator(customer_data)
    
    # Geographic business rules
    restricted_states = ['HI', 'AK']  # Mock service restrictions
    if customer_data.get('state') in restricted_states:
        # Remote states require phone contact
        if not customer_data.get('phone'):
            raise ValueError(f"Phone required for customers in {customer_data.get('state')}")
    
    # ZIP code based service rules
    premium_zip_prefixes = ['90', '10', '94']  # Mock premium service areas
    customer_zip = customer_data.get('zip', '')
    
    if any(customer_zip.startswith(prefix) for prefix in premium_zip_prefixes):
        # Premium areas require complete contact info
        if not (customer_data.get('email') and customer_data.get('phone')):
            raise ValueError(f"Premium service area {customer_zip[:2]}XXX requires both email and phone")
    
    return customer_data

def city_state_consistency_validator(customer_data: dict) -> dict:
    """
    City-state consistency validator (depth 2).
    
    Validates that city and state combinations are realistic.
    Called by address_completeness_validator.
    """
    
    # Mock city-state consistency rules
    city_state_rules = {
        'los angeles': 'CA',
        'new york': 'NY',
        'miami': 'FL', 
        'austin': 'TX',
        'seattle': 'WA',
        'chicago': 'IL'
    }
    
    city = customer_data.get('city', '').lower().strip()
    state = customer_data.get('state', '').strip()
    
    if city in city_state_rules:
        expected_state = city_state_rules[city]
        if state != expected_state:
            raise ValueError(f"City '{customer_data.get('city')}' is not in state '{state}' (expected '{expected_state}')")
    
    # ZIP-state consistency check
    zip_state_prefixes = {
        '9': ['CA', 'NV'],  # West Coast
        '1': ['NY', 'NJ', 'CT'],  # Northeast  
        '3': ['FL', 'GA', 'SC'],  # Southeast
        '7': ['TX', 'OK'],  # South Central
    }
    
    customer_zip = customer_data.get('zip', '')
    if customer_zip and state:
        zip_prefix = customer_zip[0]
        if zip_prefix in zip_state_prefixes:
            valid_states = zip_state_prefixes[zip_prefix]
            if state not in valid_states:
                raise ValueError(f"ZIP code {customer_zip} (region {zip_prefix}) doesn't match state {state}")
    
    return customer_data

def phone_format_validator(phone: str) -> str:
    """
    Phone format validator (depth 1).
    
    Validates and formats US phone numbers.
    """
    if not phone:
        return phone
        
    # Remove all non-digit characters
    clean_phone = re.sub(r'[^\d]', '', phone)
    
    # Must be exactly 10 digits (US format)
    if len(clean_phone) != 10:
        raise ValueError("Phone must be 10 digits (US format)")
    
    # Cannot start with 0 or 1 (invalid US area codes)
    if clean_phone[0] in ['0', '1']:
        raise ValueError("Phone number cannot start with 0 or 1")
    
    # Format as (XXX) XXX-XXXX
    formatted = f"({clean_phone[:3]}) {clean_phone[3:6]}-{clean_phone[6:]}"
    return formatted
