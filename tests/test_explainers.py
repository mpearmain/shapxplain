import pytest
from unittest.mock import MagicMock
from shapxplain.explainers import ShapLLMExplainer
from shapxplain.schemas import SHAPExplanationResponse

@pytest.fixture
def mock_llm_response():
    """Fixture for mock LLM response"""
    return {
        "summary": "Test summary",
        "detailed_explanation": "Test explanation",
        "recommendations": ["Test recommendation"],
        "confidence_level": "high",
        "feature_interactions": {},
        "features": [
            {
                "feature_name": "feature_1",
                "shap_value": 0.5,
                "original_value": 10,
                "contribution_direction": "increase",
                "significance": "high"
            }
        ]
    }

@pytest.fixture
def mock_agent(mock_llm_response):
    """Fixture to provide a mock LLM agent."""
    mock_agent = MagicMock()
    mock_agent.run_sync.return_value.data = mock_llm_response
    return mock_agent

@pytest.fixture
def mock_explainer(mock_agent):
    """Fixture to provide a ShapLLMExplainer with a mock agent."""
    mock_model = MagicMock()
    mock_model.__class__.__name__ = "MockModel"
    return ShapLLMExplainer(
        model=mock_model,
        llm_agent=mock_agent,
        feature_names=["feature_1", "feature_2", "feature_3"],
        significance_threshold=0.1,
    )

def test_explain_single(mock_explainer):
    """Test single explanation generation."""
    shap_values = [0.5, -0.2, 0.1]
    data_point = [10, 20, 30]
    prediction = 0.9

    response = mock_explainer.explain(
        shap_values=shap_values, 
        data_point=data_point, 
        prediction=prediction
    )

    assert isinstance(response, SHAPExplanationResponse)
    assert response.summary == "Test summary"
    assert len(response.features) == 1