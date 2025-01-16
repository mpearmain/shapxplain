"""
Schemas module for input/output validation using Pydantic.
"""
from pydantic import BaseModel

class SHAPInputSchema(BaseModel):
    shap_values: dict
    data_point: dict
    template: str

class SHAPOutputSchema(BaseModel):
    explanations: list
    recommendations: list
