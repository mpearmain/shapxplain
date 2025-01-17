import pytest
from shapxplain.prompts import (
    generate_explanation_prompt,
    generate_batch_insight_prompt,
    format_feature_contributions,
)
from shapxplain.schemas import (
    SHAPFeatureContribution,
    SignificanceLevel,
    ContributionDirection,
)


def test_generate_explanation_prompt():
    """Test generating a single prediction explanation prompt."""
    features = [
        SHAPFeatureContribution(
            feature_name="feature_1",
            shap_value=0.5,
            original_value=10,
            contribution_direction=ContributionDirection.INCREASE,
            significance=SignificanceLevel.HIGH,
        ),
        SHAPFeatureContribution(
            feature_name="feature_2",
            shap_value=-0.2,
            original_value=20,
            contribution_direction=ContributionDirection.DECREASE,
            significance=SignificanceLevel.MEDIUM,
        ),
    ]

    prompt = generate_explanation_prompt(
        model_type="MockModel",
        prediction=0.9,
        features=features,
        prediction_class="positive",
        context={"context_key": "context_value"},
    )

    assert "MockModel" in prompt
    assert "feature_1" in prompt
    assert "context_key" in prompt


def test_format_feature_contributions():
    """Test formatting feature contributions."""
    features = [
        SHAPFeatureContribution(
            feature_name="feature_1",
            shap_value=0.5,
            original_value=10,
            contribution_direction=ContributionDirection.INCREASE,
            significance=SignificanceLevel.HIGH,
        )
    ]

    formatted = format_feature_contributions(features)
    assert "feature_1" in formatted
    assert "high" in formatted  # Changed from "HIGH" to "high"


def test_generate_batch_insight_prompt():
    """Test generating a batch insight prompt."""
    prompt = generate_batch_insight_prompt(
        model_type="MockModel",
        predictions=[0.9, 0.85],
        common_features=["feature_1", "feature_2"],
        confidence_summary={
            SignificanceLevel.HIGH: 1,
            SignificanceLevel.MEDIUM: 0,
            SignificanceLevel.LOW: 1,
        },
    )

    assert "MockModel" in prompt
    assert "feature_1" in prompt
    assert "High Confidence: 1" in prompt  # Fixed assertion
