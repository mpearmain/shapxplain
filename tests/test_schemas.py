import pytest
from shapxplain.explainers import ShapLLMExplainer
from shapxplain.schemas import (
    SHAPFeatureContribution,
    SHAPExplanationRequest,
    SHAPExplanationResponse,
    ContributionDirection,
    SignificanceLevel,
)


@pytest.fixture
def mock_explainer(mock_agent):
    """Fixture to provide a ShapLLMExplainer with a mock agent."""
    mock_model = object()  # Use a placeholder or mock model
    return ShapLLMExplainer(
        model=mock_model,
        llm_agent=mock_agent,
        feature_names=["feature_1", "feature_2", "feature_3"],
        significance_threshold=0.1,
    )


def test_shap_feature_contribution():
    """Test SHAPFeatureContribution schema validation."""
    feature = SHAPFeatureContribution(
        feature_name="feature_1",
        shap_value=0.5,
        original_value=10,
        contribution_direction=ContributionDirection.INCREASE,
        significance=SignificanceLevel.HIGH,
    )

    assert feature.feature_name == "feature_1"
    assert feature.shap_value == 0.5


def test_shap_explanation_request():
    """Test SHAPExplanationRequest schema validation."""
    features = [
        SHAPFeatureContribution(
            feature_name="feature_1",
            shap_value=0.5,
            original_value=10,
            contribution_direction=ContributionDirection.INCREASE,
            significance=SignificanceLevel.HIGH,
        )
    ]

    request = SHAPExplanationRequest(
        model_type="MockModel",
        prediction=0.9,
        features=features,
        context={"key": "value"},
    )

    assert request.model_type == "MockModel"
    assert request.features[0].feature_name == "feature_1"


def test_shap_explanation_response():
    """Test SHAPExplanationResponse schema validation."""
    response = SHAPExplanationResponse(
        summary="Test summary",
        detailed_explanation="Test detailed analysis",
        recommendations=["Recommendation 1"],
        confidence_level=SignificanceLevel.HIGH,
        feature_interactions={},
        features=[
            SHAPFeatureContribution(
                feature_name="feature_1",
                shap_value=0.5,
                original_value=10,
                contribution_direction=ContributionDirection.INCREASE,
                significance=SignificanceLevel.HIGH,
            )
        ]
    )

    assert response.summary == "Test summary"
    assert response.features[0].feature_name == "feature_1"
