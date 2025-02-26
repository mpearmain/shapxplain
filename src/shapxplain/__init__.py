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
from shapxplain.utils import setup_logger, logger

__all__ = [
    "ShapLLMExplainer",
    "SHAPFeatureContribution",
    "SHAPExplanationRequest",
    "SHAPExplanationResponse",
    "SHAPBatchExplanationResponse",
    "ContributionDirection",
    "SignificanceLevel",
    "setup_logger",
    "logger",
]

__version__ = "0.1.0"

# Initialize the default logger
setup_logger()
