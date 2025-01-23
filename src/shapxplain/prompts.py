"""
Prompts module for generating SHAP-to-LLM templates and system prompts.

This module provides templates and generators for creating structured prompts
that help LLMs interpret and explain SHAP values effectively.
"""

from typing import Dict, Any, List
from shapxplain.schemas import SHAPFeatureContribution, SignificanceLevel

# System prompt for SHAP explanations
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specializing in explaining machine learning predictions in clear, practical terms.
Your role is to help users understand why a model made specific predictions and what actions they can take
based on this understanding.

Follow these principles in your explanations:
1. Use natural language without technical jargon - never mention terms like "SHAP values" or "coefficients"
2. Focus on the practical meaning and real-world implications of model decisions
3. Provide concrete, actionable insights that users can implement
4. Consider the relationships between different factors
5. Ground all explanations in the context of the specific use case
6. Frame recommendations in terms of practical steps that can improve outcomes
7. When discussing feature importance, explain why certain factors matter in relatable terms
8. Consider both individual factors and how they work together
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
Context:
The {model_type} analyzed this case and predicted: {prediction} {class_info}

Key Factors Analyzed:
{feature_contributions}

Additional Context:
{context}

Based on this information, provide a comprehensive analysis in the following format:

1. A clear summary of the main factors driving this prediction
2. A natural explanation of how different factors work together with no mention of the technical model or SHAP values
3. Practical recommendations that could influence future outcomes
4. Any important considerations or limitations to keep in mind

Return your analysis as a JSON object with these exact fields
Return only the JSON object without any other text or formatting.:
{{
    "summary": "A clear, jargon-free overview of the key factors",
    "detailed_explanation": "A natural explanation of how different factors work together and their practical implications for an educated but not expert human",
    "recommendations": [
        "Specific, actionable step 1",
        "Specific, actionable step 2"
    ],
    "confidence_level": "high",  // Must be: high, medium, or low
    "feature_interactions": {{
        "factor combination 1": "How these factors work together in practical terms"
    }},
    "features": [
        {{
            "feature_name": "factor_name",
            "shap_value": 0.5,
            "original_value": 10,
            "contribution_direction": "increase",  // Must be: increase, decrease, or neutral
            "significance": "high"  // Must be: high, medium, or low
        }}
    ]
}}
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
