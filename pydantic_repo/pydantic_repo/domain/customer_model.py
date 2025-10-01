from pydantic import BaseModel
from pydantic_repo.validators_lib.validators.types import ZipCode, Sex, PastDate, Email, Name

class Customer(BaseModel):
    customer_id: int
    email: Email
    name: Name
    sex: Sex
    signup_date: PastDate
    street: str
    city: str
    state: str
    zip: ZipCode

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
