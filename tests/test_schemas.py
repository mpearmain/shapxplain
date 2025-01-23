import pytest
from pydantic import ValidationError
from shapxplain.schemas import (
    SHAPFeatureContribution,
    ContributionDirection,
    SignificanceLevel,
)


@pytest.fixture
def sample_feature_contribution():
    """Fixture for a valid feature contribution."""
    return {
        "feature_name": "feature_1",
        "shap_value": 0.5,
        "original_value": 10,
        "contribution_direction": "increase",
        "significance": "high"
    }


@pytest.fixture
def sample_features(sample_feature_contribution):
    """Fixture for a list of feature contributions."""
    return [
        SHAPFeatureContribution(**sample_feature_contribution),
        SHAPFeatureContribution(
            feature_name="feature_2",
            shap_value=-0.2,
            original_value=20,
            contribution_direction=ContributionDirection.DECREASE,
            significance=SignificanceLevel.MEDIUM,
        )
    ]


def test_contribution_direction_enum():
    """Test ContributionDirection enum values."""
    assert ContributionDirection.INCREASE == "increase"
    assert ContributionDirection.DECREASE == "decrease"
    assert ContributionDirection.NEUTRAL == "neutral"

    # Test invalid value
    with pytest.raises(ValueError):
        ContributionDirection("invalid")


def test_significance_level_enum():
    """Test SignificanceLevel enum values."""
    assert SignificanceLevel.HIGH == "high"
    assert SignificanceLevel.MEDIUM == "medium"
    assert SignificanceLevel.LOW == "low"

    # Test invalid value
    with pytest.raises(ValueError):
        SignificanceLevel("invalid")


def test_shap_feature_contribution_validation(sample_feature_contribution):
    """Test SHAPFeatureContribution schema validation."""
    # Test valid creation
    feature = SHAPFeatureContribution(**sample_feature_contribution)
    assert feature.feature_name == "feature_1"
    assert feature.shap_value == 0.5

    # Test invalid shap_value
    with pytest.raises(ValidationError):
        invalid_data = sample_feature_contribution.copy()
        invalid_data["shap_value"] = "not a number"
        SHAPFeatureContribution(**invalid_data)

    # Test invalid contribution_direction
    with pytest.raises(ValidationError):
        invalid_data = sample_feature_contribution.copy()
        invalid_data["contribution_direction"] = "invalid"
        SHAPFeatureContribution(**invalid_data)


def test_shap_explanation_request(sample_features):
    """Test SHAPExplanationRequest schema validation."""
    # Test vali
