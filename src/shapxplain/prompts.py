"""
Prompts module for generating SHAP-to-LLM templates and system prompts.

This module provides templates and generators for creating structured prompts
that help LLMs interpret and explain SHAP values effectively.
"""

from typing import Dict, Any, List
from shapxplain.schemas import SHAPFeatureContribution, SignificanceLevel

# System prompt for SHAP explanations
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specializing in machine learning explanations and interpretability.
Your primary task is to help users understand the predictions made by machine learning models
by interpreting SHAP (SHapley Additive exPlanations) values.

Remember to:
1. Focus on the most significant features first
2. Explain in clear, non-technical terms
3. Provide specific, actionable insights when possible
4. Consider feature interactions where relevant
5. Always tie explanations back to the actual prediction
"""


# Utility functions for formatting
def format_feature_contributions(features: List[SHAPFeatureContribution]) -> str:
    """
    Format a list of SHAP feature contributions for prompt generation.

    Args:
        features (List[SHAPFeatureContribution]): List of feature contributions.

    Returns:
        str: Formatted string of feature contributions.
    """
    return "\n".join(
        f"- {feature.feature_name}: SHAP Value = {feature.shap_value:.3f} "
        f"(Original Value = {feature.original_value}, "
        f"Impact = {feature.significance} {feature.contribution_direction})"
        for feature in features
    )


def format_context(context: Dict[str, Any]) -> str:
    """
    Format the context dictionary for prompt generation.

    Args:
        context (Dict[str, Any]): Context information.

    Returns:
        str: Formatted string of context information.
    """
    return "\n".join(f"- {key}: {value}" for key, value in context.items()) if context else "No additional context."


# Templates
SINGLE_PREDICTION_EXPLANATION_TEMPLATE = """
Model Information:
- Type: {model_type}
- Prediction: {prediction}
{class_info}

Feature Contributions (ordered by importance):
{feature_contributions}

Context:
{context}

Please provide:
1. A concise summary of the key factors driving this prediction
2. A detailed explanation of how significant features contribute
3. Any notable feature interactions or patterns
4. Specific, actionable recommendations based on this analysis
5. Any potential caveats or areas of uncertainty

Your response should be structured as:
- Summary: Brief overview of key drivers
- Detailed Analysis: Feature-by-feature breakdown
- Recommendations: Actionable insights
- Confidence Level: Your confidence in this explanation
"""

BATCH_INSIGHT_TEMPLATE = """
Analyze the following batch of predictions:

Number of Cases: {num_cases}
Model Type: {model_type}
Prediction Range: {pred_range}

Common Patterns:
{common_patterns}

Please identify:
1. Overall trends in feature importance
2. Any consistent patterns across predictions
3. Notable outliers or unusual cases
4. General recommendations based on the batch
"""


# Prompt Generators
def generate_explanation_prompt(
        model_type: str,
        prediction: float,
        features: List[SHAPFeatureContribution],
        prediction_class: str = None,
        context: Dict[str, Any] = None,
) -> str:
    """
    Generate a detailed prompt for explaining a single prediction.

    Args:
        model_type (str): The type of model being explained.
        prediction (float): The model's prediction value.
        features (List[SHAPFeatureContribution]): List of feature contributions.
        prediction_class (str): Optional class label for classification tasks.
        context (Dict[str, Any]): Additional contextual information.

    Returns:
        str: Formatted prompt string ready for LLM.
    """
    class_info = f"- Predicted Class: {prediction_class}" if prediction_class else ""
    feature_contributions = format_feature_contributions(features)
    context_str = format_context(context)

    return SINGLE_PREDICTION_EXPLANATION_TEMPLATE.format(
        model_type=model_type,
        prediction=prediction,
        class_info=class_info,
        feature_contributions=feature_contributions,
        context=context_str,
    )


def generate_batch_insight_prompt(
        model_type: str,
        predictions: List[float],
        common_features: List[str],
        confidence_summary: Dict[SignificanceLevel, int],
) -> str:
    """
    Generate a prompt for analyzing batch predictions.

    Args:
        model_type (str): The type of model being explained.
        predictions (List[float]): List of prediction values.
        common_features (List[str]): List of frequently important features.
        confidence_summary (Dict[SignificanceLevel, int]): Summary of confidence levels.

    Returns:
        str: Formatted prompt string for batch analysis.
    """
    pred_range = f"{min(predictions):.2f} to {max(predictions):.2f}"
    common_patterns = "\n".join(f"- {feature}" for feature in common_features)
    confidence_summary_str = "\n".join(
        f"- {level.value.title()} Confidence: {count}" for level, count in confidence_summary.items()
    )

    return BATCH_INSIGHT_TEMPLATE.format(
        num_cases=len(predictions),
        model_type=model_type,
        pred_range=pred_range,
        common_patterns=common_patterns,
    ) + f"\nConfidence Summary:\n{confidence_summary_str}"
