"""
Prompts module for generating SHAP-to-LLM templates and system prompts.

This module provides templates and generators for creating structured prompts
that help LLMs interpret and explain SHAP values effectively. It includes
optimised prompts for both single predictions and batch analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
from shapxplain.schemas import SHAPFeatureContribution, SignificanceLevel

# System prompt for SHAP explanations
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specialising in explaining machine learning predictions in clear, practical terms.
Your role is to help users understand why a model made specific predictions and what actions they can take
based on this understanding.

Follow these principles in your explanations:
1. Use natural language without technical jargon - never mention terms like "SHAP values" or "coefficients"
2. Focus on the practical meaning and real-world implications of model decisions
3. Provide concrete, actionable insights that users can implement
4. Consider the relationships between different factors and how they interact
5. Ground all explanations in the context of the specific use case
6. Frame recommendations in terms of practical steps that can improve outcomes
7. When discussing feature importance, explain why certain factors matter in relatable terms
8. Consider both individual factors and how they work together

For feature interactions:
- Identify pairs or groups of features that work together
- Explain how these combinations affect the prediction in practical terms
- Highlight when factors reinforce or counteract each other

For confidence levels:
- Use "high" when the explanation is strongly supported by the data
- Use "medium" when the explanation has reasonable support but some uncertainty
- Use "low" when multiple interpretations are possible or the impact is unclear

For recommendations:
- Focus on actionable steps rather than general principles
- Consider the feasibility and practicality of recommendations
- Tailor recommendations to the specific prediction and use case
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
    if not features:
        return "No significant features identified."

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
    if not context:
        return "No additional context."

    formatted_items = []
    for key, value in context.items():
        if isinstance(value, dict):
            # Handle nested dictionaries like feature descriptions
            nested_items = "\n  ".join(f"- {k}: {v}" for k, v in value.items())
            formatted_items.append(f"- {key}:\n  {nested_items}")
        else:
            formatted_items.append(f"- {key}: {value}")

    return "\n".join(formatted_items)


def format_common_features(
    common_features: List[Tuple[str, int]], total_cases: int
) -> str:
    """
    Format the list of common features for batch insight prompts.

    Args:
        common_features (List[Tuple[str, int]]): List of feature names and their counts.
        total_cases (int): Total number of cases analyzed.

    Returns:
        str: Formatted string of common features with occurrence percentages.
    """
    if not common_features:
        return "No consistent feature patterns identified across cases."

    return "\n".join(
        f"- {feature}: appears in {count}/{total_cases} cases ({count / total_cases:.0%})"
        for feature, count in common_features
    )


# Templates
SINGLE_PREDICTION_EXPLANATION_TEMPLATE = """
Context:
The {model_type} analyzed this case and predicted: {prediction} {class_info}

Key Factors Analyzed:
{feature_contributions}

Additional Context:
{context}

Based on this information, provide a comprehensive analysis that would help a non-technical stakeholder understand this prediction and take appropriate action. Your explanation should be:

1. Clear and jargon-free - avoid technical terms like SHAP, coefficients, or algorithms
2. Focused on practical implications and actionable insights
3. Specific to this particular case and prediction
4. Balanced in considering both positive and negative factors

Return your analysis as a JSON object with these exact fields
**IMPORTANT**: For the field `"contribution_direction"` within the `"features"` array, you **MUST** choose **EXACTLY ONE** value from the following limited list of three options in lowercase**: `"increase"`, `"decrease"`, or `"neutral"`. Do not use any synonyms, variations, or other phrases. Use only these exact lowercase strings.
Return only the JSON object without any other text or formatting:
{{
    "summary": "A clear, jargon-free overview of the 2-3 key factors driving this prediction",
    
    "detailed_explanation": "A natural explanation of how different factors work together and their practical implications. Focus on causality where possible and avoid technical terminology. Explain as if to an educated person without ML knowledge.",
    
    "recommendations": [
        "Specific, actionable step 1 that could influence future outcomes",
        "Specific, actionable step 2 that addresses the most important factors"
    ],
    
    "confidence_level": "high",  // Must be: high, medium, or low based on the clarity and consistency of the factors
    
    "feature_interactions": {{
        "factor combination 1": "How these specific factors work together in practical terms and why their combination matters"
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
Analyze the following batch of {num_cases} predictions made by a {model_type} model:

Summary Statistics:
- Prediction Range: {pred_range}
- Confidence Distribution: {confidence_distribution}

Most Common Important Features:
{common_patterns}

Based on this data, provide a comprehensive batch analysis in JSON format with these exact fields.
Return only the JSON object without any other text or formatting:
{{
    "insights": [
        "Overall trend insight focusing on key patterns across predictions",
        "Analysis of feature importance consistency and variations",
        "Identification of any notable outliers or unusual cases",
        "Observation about feature interactions across the batch"
    ],
    "batch_recommendations": [
        "Specific, actionable recommendation based on the entire batch",
        "Recommendation addressing the most consistent feature patterns"
    ],
    "confidence_assessment": "An evaluation of the overall prediction reliability",
    "key_feature_groups": {{
        "group_name_1": "Description of how these features consistently work together across cases",
        "group_name_2": "Description of another feature interaction pattern"
    }}
}}
"""


# Prompt Generators
def generate_explanation_prompt(
    model_type: str,
    prediction: float,
    features: List[SHAPFeatureContribution],
    prediction_class: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a detailed prompt for explaining a single prediction.

    Args:
        model_type (str): The type of model being explained.
        prediction (float): The model's prediction value.
        features (List[SHAPFeatureContribution]): List of feature contributions.
        prediction_class (str, optional): Class label for classification tasks.
        context (Dict[str, Any], optional): Additional contextual information.

    Returns:
        str: Formatted prompt string ready for LLM.
    """
    class_info = f"- Predicted Class: {prediction_class}" if prediction_class else ""
    feature_contributions = format_feature_contributions(features)
    context_str = format_context(context or {})

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
    common_features: List[Tuple[str, int]],
    confidence_summary: Dict[SignificanceLevel, int],
    confidence_distribution: Optional[str] = None,
) -> str:
    """
    Generate a prompt for analyzing batch predictions.

    Args:
        model_type (str): The type of model being explained.
        predictions (List[float]): List of prediction values.
        common_features (List[Tuple[str, int]]): List of frequently important features with counts.
        confidence_summary (Dict[SignificanceLevel, int]): Summary of confidence levels.
        confidence_distribution (str, optional): Formatted confidence distribution string.

    Returns:
        str: Formatted prompt string for batch analysis.
    """
    pred_range = f"{min(predictions):.2f} to {max(predictions):.2f}"
    common_patterns = format_common_features(common_features, len(predictions))

    if confidence_distribution is None:
        confidence_distribution = ", ".join(
            f"{level.value.title()}: {count}"
            for level, count in confidence_summary.items()
        )

    return BATCH_INSIGHT_TEMPLATE.format(
        num_cases=len(predictions),
        model_type=model_type,
        pred_range=pred_range,
        common_patterns=common_patterns,
        confidence_distribution=confidence_distribution,
    )
