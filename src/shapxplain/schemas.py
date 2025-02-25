"""
Core schema definitions for SHAP-LLM explanations.

This module defines the core data structures used throughout the SHAPXplain package.
It provides Pydantic models for validating and structuring SHAP values, feature
contributions, and LLM-generated explanations.

The module implements a comprehensive type system for SHAP explanations, ensuring
type safety and validation while maintaining clear documentation and examples.

Key components:
- SHAPFeatureContribution: Individual feature contribution with SHAP values
- SHAPExplanationRequest: Input structure for LLM explanation generation
- SHAPExplanationResponse: Structured LLM-generated explanations
- SHAPBatchRequest/Response: Handling multiple explanations efficiently
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ContributionDirection(str, Enum):
    """
    Classification of a feature's impact direction based on SHAP value.

    This enum categorizes how each feature contributes to the model's prediction
    based on its SHAP value. The direction helps in understanding whether a feature
    pushes the prediction higher or lower from the base value.

    Values:
        INCREASE: Positive SHAP value, pushes prediction higher than the base value
        DECREASE: Negative SHAP value, pushes prediction lower than the base value
        NEUTRAL: Zero or near-zero SHAP value, minimal impact on prediction
    """

    INCREASE = "increase"
    DECREASE = "decrease"
    NEUTRAL = "neutral"


class SignificanceLevel(str, Enum):
    """
    Categorization of feature importance based on normalized SHAP value magnitude.

    This enum provides a qualitative assessment of a feature's importance based on
    its normalized SHAP value magnitude. The categorization helps in quickly
    identifying which features have the strongest impact on the model's predictions.

    Values:
        HIGH: Major contribution to the prediction (typically |SHAP| > 2*threshold)
        MEDIUM: Moderate contribution (typically threshold <= |SHAP| < 2*threshold)
        LOW: Minor contribution (typically |SHAP| < threshold)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SHAPFeatureContribution(BaseModel):
    """
    Structured representation of a single feature's SHAP-based contribution.

    This model captures both the numerical SHAP value and its interpreted meaning
    for a specific feature in the context of a model prediction. It combines raw
    SHAP values with qualitative assessments to make the feature's impact more
    interpretable.

    Attributes:
        feature_name: Name/identifier of the feature being explained
        shap_value: Calculated SHAP value indicating feature importance and direction
        original_value: The actual value of this feature in the input data
        contribution_direction: Whether this feature increased/decreased the prediction
        significance: Categorized importance level based on SHAP value magnitude
    """

    feature_name: str = Field(description="Name or identifier of the feature")
    shap_value: float = Field(description="SHAP value quantifying feature importance")
    original_value: Any = Field(description="Original feature value from input data")
    contribution_direction: ContributionDirection = Field(
        description="Direction of feature's impact on prediction"
    )
    significance: SignificanceLevel = Field(
        description="Categorized level of feature importance"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "feature_name": "age",
                "shap_value": 0.75,
                "original_value": 45,
                "contribution_direction": "increase",
                "significance": "high",
            }
        },
    )


class SHAPExplanationRequest(BaseModel):
    """
    Input structure for requesting an LLM-based explanation of SHAP values.

    This model encapsulates all necessary information for generating a comprehensive
    explanation of a model's prediction using SHAP values and LLM interpretation.
    It combines model metadata, predictions, and feature contributions to provide
    context for the explanation.

    Attributes:
        model_type: Type/name of the ML model being explained (e.g., "RandomForest")
        prediction: The model's numerical prediction (probability for classification)
        prediction_class: For classification tasks, the predicted class label
        features: List of feature contributions with their SHAP values
        context: Optional additional context to enhance explanation quality
    """

    model_type: str = Field(description="Type/architecture of ML model being explained")
    prediction: float = Field(description="Model's numerical prediction or probability")
    prediction_class: Optional[str] = Field(
        None, description="Predicted class label for classification tasks"
    )
    features: List[SHAPFeatureContribution] = Field(
        description="List of feature contributions with SHAP values"
    )
    context: Dict[str, Any]

    model_config = ConfigDict(use_enum_values=True)

    def model_post_init(self, __context: Any) -> None:
        if self.context is None:
            self.context = {}


class SHAPExplanationResponse(BaseModel):
    """
    Structured output from LLM-based SHAP value interpretation.

    This model captures the LLM's analysis of SHAP values in a structured format,
    providing multiple levels of explanation and actionable insights. The response
    includes both high-level summaries and detailed analyses of feature interactions.

    Attributes:
        summary: Brief overview of key prediction drivers
        detailed_explanation: In-depth analysis of feature contributions
        recommendations: Actionable insights based on the analysis
        confidence_level: LLM's confidence in the explanation
        feature_interactions: Optional analysis of feature interactions
    """

    summary: str = Field(description="Concise summary of key prediction drivers")
    detailed_explanation: str = Field(
        description="Comprehensive analysis of SHAP values and their implications"
    )
    recommendations: List[str] = Field(
        description="Actionable insights derived from the analysis"
    )
    confidence_level: SignificanceLevel = Field(
        description="LLM's confidence in the explanation"
    )
    feature_interactions: Dict[str, str]
    features: List[SHAPFeatureContribution] = Field(  # Add this field
        description="Features that contributed to this explanation"
    )

    model_config = ConfigDict(use_enum_values=True)

    def model_post_init(self, __context: Any) -> None:
        if self.feature_interactions is None:
            self.feature_interactions = {}


class SHAPBatchExplanationResponse(BaseModel):
    """
    Consolidated response for batch SHAP explanations.

    This model provides a structured way to handle multiple SHAP explanations
    simultaneously, including aggregate statistics and cross-request insights.
    It's particularly useful for analyzing patterns across multiple predictions
    or comparing explanations across different instances.

    Attributes:
        responses: Individual explanations for each request in the batch
        summary_statistics: Aggregated metrics and insights across all explanations
        batch_insights: Cross-request patterns and observations
    """

    responses: List[SHAPExplanationResponse] = Field(
        description="List of individual SHAP explanations"
    )
    summary_statistics: Dict[str, Any]
    batch_insights: List[str]

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "responses": [],
                "summary_statistics": {
                    "total_processed": 100,
                    "high_confidence": 75,
                    "medium_confidence": 20,
                    "low_confidence": 5,
                },
                "batch_insights": [
                    "Feature 'age' was consistently important",
                    "Strong interaction between 'income' and 'education'",
                ],
            }
        },
    )

    def model_post_init(self, __context: Any) -> None:
        if self.summary_statistics is None:
            self.summary_statistics = {}
        if self.batch_insights is None:
            self.batch_insights = []
