from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional
from pydantic_repo.validators_lib.validators.types import ZipCode, Sex, PastDate, Email, Name

# Import all validator functions at the top
from pydantic_repo.validators_lib.validators.core import (
    customer_contact_completeness_validator,
    address_completeness_validator,
    contact_method_validator,
    geographic_service_validator,
    city_state_consistency_validator,
    phone_format_validator
)

class Customer(BaseModel):
    customer_id: int
    email: Email
    name: Name
    sex: Sex
    signup_date: PastDate
    street: str
    city: str = Field(pattern=r'^[a-zA-Z\s]+$')  # Only letters and spaces
    state: str
    zip: ZipCode
    phone: Optional[str] = None  # Optional with default None

    def model_dump_safe(self):
        """Convert model to dict with string values - avoids TypeError in validators."""
        return {
            'customer_id': self.customer_id,
            'email': str(self.email) if self.email else None,
            'name': str(self.name) if self.name else None,
            'sex': str(self.sex) if self.sex else None,
            'signup_date': str(self.signup_date) if self.signup_date else None,
            'street': str(self.street) if self.street else None,
            'city': str(self.city) if self.city else None,
            'state': str(self.state) if self.state else None,
            'zip': str(self.zip) if self.zip else None,
            'phone': str(self.phone) if self.phone else None
        }


    # Field-level validators (complement type-based and model-level validation)
    
    @field_validator('customer_id')
    @classmethod
    def validate_customer_id_positive(cls, v):
        """Ensure customer ID is positive."""
        if v <= 0:
            raise ValueError('Customer ID must be positive')
        return v
    
    @field_validator('street')
    @classmethod
    def validate_street_format(cls, v):
        """Validate street address format and content."""
        if not v or not v.strip():
            raise ValueError('Street address cannot be empty')
        
        # Check for reasonable street address patterns
        v = v.strip()
        if len(v) < 5:
            raise ValueError('Street address too short')
        
        # Must contain at least one digit (house number)
        if not any(c.isdigit() for c in v):
            raise ValueError('Street address must contain a house number')
            
        return v
    
    @field_validator('state')
    @classmethod
    def validate_state_code(cls, v):
        """Validate US state code format."""
        if not v:
            raise ValueError('State cannot be empty')
            
        v = v.strip().upper()
        
        # Basic format check - 2 letters
        if len(v) != 2 or not v.isalpha():
            raise ValueError('State must be a 2-letter code (e.g., CA, NY, TX)')
        
        # Optional: validate against actual state codes
        valid_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        }
        
        if v not in valid_states:
            raise ValueError(f'Invalid state code: {v}. Must be a valid US state.')
            
        return v
    

    # Model-level validators (validate relationships between fields)
    
    @model_validator(mode="after")
    def validate_contact_completeness(self):
        """Validate customer contact completeness using complex business logic."""
        customer_dict = self.model_dump_safe()
        customer_contact_completeness_validator(customer_dict)
        return self
        
    @model_validator(mode="after")
    def validate_address_completeness(self):
        """Validate address completeness across multiple fields."""
        customer_dict = self.model_dump_safe()
        address_completeness_validator(customer_dict)
        return self
        
    @model_validator(mode="after")
    def validate_contact_method(self):
        """Validate appropriate contact methods are available."""
        customer_dict = self.model_dump_safe()
        contact_method_validator(customer_dict)
        return self
        
    @model_validator(mode="after")
    def validate_geographic_service(self):
        """Validate geographic service coverage and consistency."""
        customer_dict = self.model_dump_safe()
        geographic_service_validator(customer_dict)
        return self
        
    @model_validator(mode="after")
    def validate_city_state_consistency(self):
        """Validate city and state geographical consistency."""
        customer_dict = self.model_dump_safe()
        city_state_consistency_validator(customer_dict)
        return self
        
    @model_validator(mode="after")
    def validate_phone_format_model(self):
        """Validate phone number format when provided."""
        if self.phone:  # Only validate if phone is provided
            validated_phone = phone_format_validator(str(self.phone))
            # Update the phone field with validated/formatted value
            object.__setattr__(self, 'phone', validated_phone)
        return self

class Address(BaseModel):

    street: str
    city: str
    state: str
    zip: ZipCode
    
    @classmethod
    def from_customer(cls, customer: Customer):
        return cls(
            street=customer.street,
            city=customer.city,
            state=customer.state,
            zip=customer.zip
        )
