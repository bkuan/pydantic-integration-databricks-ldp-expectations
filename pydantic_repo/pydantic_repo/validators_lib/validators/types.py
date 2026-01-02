from typing_extensions import Annotated
from pydantic import AfterValidator, Field
from .core import email_validator, zipcode_validator, sex_validator, past_date_validator

Email    = Annotated[str, AfterValidator(email_validator)]
ZipCode  = Annotated[str, AfterValidator(zipcode_validator)]
Sex      = Annotated[str, AfterValidator(sex_validator)]
PastDate = Annotated[str, AfterValidator(past_date_validator)]  
Name     = Annotated[str, Field(min_length=1, max_length=80)]
