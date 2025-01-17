"""
SHAPXplain - A package for combining SHAP values with LLM explanations.
"""

from shapxplain.explainers import ShapLLMExplainer
from shapxplain.schemas import (
    SHAPFeatureContribution,
    SHAPExplanationRequest,
    SHAPExplanationResponse,
    SHAPBatchExplanationResponse,
    ContributionDirection,
    SignificanceLevel,
)

__all__ = [
    'ShapLLMExplainer',
    'SHAPFeatureContribution',
    'SHAPExplanationRequest',
    'SHAPExplanationResponse',
    'SHAPBatchExplanationResponse',
    'ContributionDirection',
    'SignificanceLevel',
]

__version__ = '0.1.0'