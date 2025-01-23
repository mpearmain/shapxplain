import pytest
from shapxplain.prompts import (
    generate_explanation_prompt,
    generate_batch_insight_prompt,
    format_feature_contributions,
    format_context,
    DEFAULT_SYSTEM_PROMPT
)
from shapxplain.schemas import (
    SHAPFeatureContribution,
    SignificanceLevel,
    ContributionDirection,
)


@pytest.fixture
def sample_features():
    """Fixture providing sample feature contributions."""
    return [
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


def test_generate_explanation_prompt(sample_features):
    """Test generating a single prediction explanation prompt."""
    prompt = generate_explanation_prompt(
        model_type="MockModel",
        prediction=0.9,
        features=sample_features,
        prediction_class="positive",
        context={"context_key": "context_value"},
    )

    # Check required components
    assert "MockModel" in prompt
    assert "feature_1" in prompt
    assert "context_key" in prompt
    assert "0.9" in prompt
    assert "positive" in prompt


def test_generate_explanation_prompt_no_class(sample_features):
    """Test prompt generation without prediction class."""
    prompt = generate_explanation_prompt(
        model_type="MockModel",
        prediction=0.9,
        features=sample_features,
        context={}
    )

    assert "Predicted Class" not in prompt


def test_format_feature_contributions(sample_features):
    """Test formatting feature contributions."""
    formatted = format_feature_contributions(sample_features)

    assert "feature_1" in formatted
    assert "feature_2" in formatted
    assert "0.5" in formatted
    assert "-0.2" in formatted
    assert "high" in formatted.lower()
    assert "medium" in formatted.lower()


def test_format_empty_feature_list():
    """Test formatting empty feature list."""
    formatted = format_feature_contributions([])
    assert formatted == ""


def test_format_context():
    """Test context formatting."""
    context = {
        "dataset": "test_dataset",
        "model_version": "1.0",
        "timestamp": "2024-01-23"
    }

    formatted = format_context(context)

    assert "dataset" in formatted
    assert "test_dataset" in formatted
    assert "model_version" in formatted
    assert "1.0" in formatted


def test_format_empty_context():
    """Test formatting empty context."""
    assert format_context({}) == "No additional context."


def test_generate_batch_insight_prompt():
    """Test generating a batch insight prompt."""
    prompt = generate_batch_insight_prompt(
        model_type="MockModel",
        predictions=[0.9, 0.85, 0.95],
        common_features=["feature_1", "feature_2"],
        confidence_summary={
            SignificanceLevel.HIGH: 2,
            SignificanceLevel.MEDIUM: 1,
            SignificanceLevel.LOW: 0,
        },
    )

    assert "MockModel" in prompt
    assert "feature_1" in prompt
    assert "feature_2" in prompt
    assert "High Confidence: 2" in prompt
    assert "Medium Confidence: 1" in prompt
    assert "Number of Cases: 3" in prompt


def test_batch_insight_prompt_no_common_features():
    """Test batch insight prompt with no common features."""
    prompt = generate_batch_insight_prompt(
        model_type="MockModel",
        predictions=[0.9],
        common_features=[],
        confidence_summary={SignificanceLevel.HIGH: 1}
    )

    assert "Number of Cases: 1" in prompt


def test_default_system_prompt():
    """Test the default system prompt content."""
    assert "explaining machine learning predictions" in DEFAULT_SYSTEM_PROMPT
    assert "clear, practical terms" in DEFAULT_SYSTEM_PROMPT
    assert "understanding" in DEFAULT_SYSTEM_PROMPT.lower()
    assert len(DEFAULT_SYSTEM_PROMPT.split("\n")) > 5  # Should be multi-line

    # Test the important principles are included
    principles = [
        "natural language",
        "practical meaning",
        "actionable insights",
        "relationships between",
        "specific use case"
    ]
    for principle in principles:
        assert principle in DEFAULT_SYSTEM_PROMPT.lower()


def test_prompt_template_escaping():
    """Test proper escaping in prompt templates."""
    prompt = generate_explanation_prompt(
        model_type="Model{with}special[chars]",
        prediction=0.9,
        features=[],
        context={"key": "value{with}brackets"}
    )

    assert "Model{with}special[chars]" in prompt
