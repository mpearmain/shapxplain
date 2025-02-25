import pytest
import numpy as np
import json
from dataclasses import dataclass
import asyncio
from unittest.mock import MagicMock, AsyncMock
from shapxplain.explainers import ShapLLMExplainer
from shapxplain.schemas import (
    SHAPExplanationResponse,
    SHAPBatchExplanationResponse,
    ContributionDirection,
    SignificanceLevel,
)


@dataclass
class MockLLMResponse:
    """Mock LLM agent response"""

    data: str


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
                "significance": "high",
            }
        ],
    }


@pytest.fixture
def mock_agent(mock_llm_response):
    """Fixture to provide a mock LLM agent with both sync and async methods."""
    mock_agent = MagicMock()
    # Set up sync method
    mock_agent.run_sync.return_value = MockLLMResponse(
        data=json.dumps(mock_llm_response)
    )

    # Set up async method
    async_mock = AsyncMock()
    async_mock.return_value = MockLLMResponse(data=json.dumps(mock_llm_response))
    mock_agent.run = async_mock

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
        max_retries=2,
        retry_delay=0.01,  # Short delay for faster tests
    )


def test_explain_single(mock_explainer):
    """Test single explanation generation."""
    shap_values = [0.5, -0.2, 0.1]
    data_point = [10, 20, 30]
    prediction = 0.9

    response = mock_explainer.explain(
        shap_values=shap_values, data_point=data_point, prediction=prediction
    )

    assert isinstance(response, SHAPExplanationResponse)
    assert response.summary == "Test summary"
    assert len(response.features) == 1


def test_explain_with_numpy_arrays(mock_explainer):
    """Test explanation generation with numpy arrays."""
    shap_values = np.array([0.5, -0.2, 0.1])
    data_point = np.array([10, 20, 30])
    prediction = 0.9

    response = mock_explainer.explain(
        shap_values=shap_values, data_point=data_point, prediction=prediction
    )

    assert isinstance(response, SHAPExplanationResponse)
    assert response.summary == "Test summary"


def test_initialization_validation():
    """Test validation during ShapLLMExplainer initialization."""
    with pytest.raises(ValueError, match="Model cannot be None"):
        ShapLLMExplainer(model=None)

    with pytest.raises(ValueError, match="significance_threshold must be positive"):
        ShapLLMExplainer(model=MagicMock(), significance_threshold=0)

    with pytest.raises(ValueError, match="All feature names must be strings"):
        ShapLLMExplainer(model=MagicMock(), feature_names=[1, 2, 3])

    # Test new validation rules
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        ShapLLMExplainer(model=MagicMock(), max_retries=-1)

    with pytest.raises(ValueError, match="retry_delay must be positive"):
        ShapLLMExplainer(model=MagicMock(), retry_delay=0)


def test_process_shap_values(mock_explainer):
    """Test SHAP value processing logic."""
    shap_values = [1.0, 0.5, 0.1]
    data_point = [10, 20, 30]

    contributions = mock_explainer._process_shap_values(shap_values, data_point)

    assert len(contributions) == 3
    assert contributions[0].shap_value == 1.0
    assert contributions[0].feature_name == "feature_1"
    assert contributions[0].significance == SignificanceLevel.HIGH


def test_determine_direction():
    """Test contribution direction determination."""
    explainer = ShapLLMExplainer(model=MagicMock())

    assert explainer._determine_direction(0.5) == ContributionDirection.INCREASE
    assert explainer._determine_direction(-0.5) == ContributionDirection.DECREASE
    assert explainer._determine_direction(0.0) == ContributionDirection.NEUTRAL


def test_input_validation(mock_explainer):
    """Test input validation for explain method."""
    with pytest.raises(ValueError, match="Length mismatch"):
        # Test mismatched lengths
        mock_explainer.explain(
            shap_values=[0.5, 0.1], data_point=[1, 2, 3], prediction=0.9
        )


def test_significance_levels():
    """Test significance level calculation."""
    explainer = ShapLLMExplainer(model=MagicMock(), significance_threshold=0.5)

    assert (
        explainer._determine_significance(1.5) == SignificanceLevel.HIGH
    )  # > 2*threshold
    assert (
        explainer._determine_significance(0.7) == SignificanceLevel.MEDIUM
    )  # > threshold
    assert (
        explainer._determine_significance(0.2) == SignificanceLevel.LOW
    )  # < threshold


@pytest.mark.parametrize(
    "error_type", [json.JSONDecodeError("msg", "doc", 0), Exception("Generic error")]
)
def test_query_llm_error_handling(error_type, mock_explainer):
    """Test error handling in LLM querying with different error types."""
    mock_explainer.llm_agent.run_sync.side_effect = error_type

    with pytest.raises(RuntimeError):
        mock_explainer._query_llm_impl("test prompt")


def test_batch_processing(mock_explainer):
    """Test batch explanation functionality."""
    shap_values_batch = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]
    data_points = [[10, 20, 30], [40, 50, 60]]
    predictions = [0.9, 0.8]

    response = mock_explainer.explain_batch(
        shap_values_batch=shap_values_batch,
        data_points=data_points,
        predictions=predictions,
    )

    assert isinstance(response, SHAPBatchExplanationResponse)
    assert len(response.responses) == 2
    assert "total_processed" in response.summary_statistics


def test_batch_validation(mock_explainer):
    """Test input validation for batch processing."""
    with pytest.raises(ValueError, match="Length mismatch in batch inputs"):
        mock_explainer.explain_batch(
            shap_values_batch=[[0.5]], data_points=[[1], [2]], predictions=[0.9]
        )


def test_confidence_summary(mock_explainer):
    """Test confidence summary calculation."""
    shap_values_batch = np.array([[1.0, 0.5], [0.05, 0.02]])
    summary = mock_explainer._calculate_confidence_summary(shap_values_batch)

    assert summary[SignificanceLevel.HIGH] == 1
    assert summary[SignificanceLevel.LOW] == 1


def test_cache_behavior(mock_explainer):
    """Test LLM query caching."""
    prompt = "test prompt"

    # First call
    response1 = mock_explainer._query_llm(prompt)

    # Second call (should use cache)
    response2 = mock_explainer._query_llm(prompt)

    assert response1 == response2
    # We call _query_llm_impl under the hood, which calls run_sync
    assert mock_explainer.llm_agent.run_sync.call_count == 1


def test_context_handling(mock_explainer):
    """Test handling of additional context."""
    context = {
        "dataset": "test_data",
        "feature_descriptions": {"f1": "Feature 1 description"},
    }

    mock_explainer.explain(
        shap_values=[0.5, -0.2, 0.1],
        data_point=[10, 20, 30],
        prediction=0.9,
        additional_context=context,
    )

    # Verify context was properly passed to LLM
    mock_explainer.llm_agent.run_sync.assert_called_once()
    call_args = mock_explainer.llm_agent.run_sync.call_args[0][0]
    assert "test_data" in call_args
    assert "Feature 1 description" in call_args


def test_clean_json_response():
    """Test JSON response cleaning."""
    # Test Markdown code block
    markdown_json = """```json
{
    "key": "value"
}
```"""
    cleaned = ShapLLMExplainer._clean_json_response(markdown_json)
    assert cleaned == '{\n    "key": "value"\n}'

    # Test clean JSON
    clean_json = '{"key": "value"}'
    cleaned = ShapLLMExplainer._clean_json_response(clean_json)
    assert cleaned == '{"key": "value"}'

    # Test with language marker
    with_lang = """```json
{
    "key": "value"
}
```"""
    cleaned = ShapLLMExplainer._clean_json_response(with_lang)
    assert cleaned == '{\n    "key": "value"\n}'

    # Test empty string
    cleaned = ShapLLMExplainer._clean_json_response("")
    assert cleaned == ""


@pytest.mark.asyncio
async def test_explain_async(mock_explainer):
    """Test asynchronous explanation generation."""
    shap_values = [0.5, -0.2, 0.1]
    data_point = [10, 20, 30]
    prediction = 0.9

    response = await mock_explainer.explain_async(
        shap_values=shap_values, data_point=data_point, prediction=prediction
    )

    assert isinstance(response, SHAPExplanationResponse)
    assert response.summary == "Test summary"
    mock_explainer.llm_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_explain_batch_async(mock_explainer):
    """Test asynchronous batch processing."""
    shap_values_batch = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]]
    data_points = [[10, 20, 30], [40, 50, 60]]
    predictions = [0.9, 0.8]

    response = await mock_explainer.explain_batch_async(
        shap_values_batch=shap_values_batch,
        data_points=data_points,
        predictions=predictions,
    )

    assert isinstance(response, SHAPBatchExplanationResponse)
    assert len(response.responses) == 2
    # Should call the async run method exactly twice (once for each data point)
    assert mock_explainer.llm_agent.run.call_count == 2


@pytest.mark.asyncio
async def test_retry_logic_async(mock_explainer):
    """Test retry logic in async LLM querying."""
    # Make the first call fail with JSONDecodeError
    bad_response = MockLLMResponse(data="not valid json")
    good_response = MockLLMResponse(data=json.dumps({"summary": "Retry succeeded"}))

    mock_explainer.llm_agent.run.side_effect = [
        bad_response,  # First call fails
        good_response,  # Second call succeeds
    ]

    result = await mock_explainer._query_llm_async("test prompt")

    assert result["summary"] == "Retry succeeded"
    assert mock_explainer.llm_agent.run.call_count == 2


@pytest.mark.asyncio
async def test_max_retries_exceeded_async(mock_explainer):
    """Test exception raised when max retries are exceeded."""
    # Make all calls fail
    bad_response = MockLLMResponse(data="not valid json")

    # Set up side effect to return bad response for all calls
    mock_explainer.llm_agent.run.side_effect = [bad_response] * (
        mock_explainer.max_retries + 1
    )

    with pytest.raises(RuntimeError, match="LLM response is not valid JSON after"):
        await mock_explainer._query_llm_async("test prompt")

    # Should try exactly max_retries + 1 times
    assert mock_explainer.llm_agent.run.call_count == mock_explainer.max_retries + 1


@pytest.mark.asyncio
async def test_parallel_processing(mock_explainer):
    """Test that batch async processing runs in parallel."""
    # Reset the mock to clear any previous calls
    mock_explainer.llm_agent.run.reset_mock()

    shap_values_batch = [[0.5, -0.2, 0.1]] * 5  # 5 identical sets
    data_points = [[10, 20, 30]] * 5
    predictions = [0.9] * 5

    # Add a small delay to simulate real LLM call
    async def delayed_response(*args, **kwargs):
        await asyncio.sleep(0.05)
        return MockLLMResponse(
            data=json.dumps(
                {
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
                            "significance": "high",
                        }
                    ],
                }
            )
        )

    mock_explainer.llm_agent.run.side_effect = delayed_response

    # The sequential version would take 5 * 0.05 = 0.25 seconds
    # The parallel version should take just over 0.05 seconds
    start_time = asyncio.get_event_loop().time()

    response = await mock_explainer.explain_batch_async(
        shap_values_batch=shap_values_batch,
        data_points=data_points,
        predictions=predictions,
    )

    elapsed_time = asyncio.get_event_loop().time() - start_time

    assert isinstance(response, SHAPBatchExplanationResponse)
    assert len(response.responses) == 5
    assert mock_explainer.llm_agent.run.call_count == 5

    # Should be closer to 0.05 than 0.25 if running in parallel
    # Allow some buffer for test overhead
    assert elapsed_time < 0.15  # Much less than the 0.25s it would take sequentially


def test_sync_vs_async_parity(mock_explainer):
    """Test that sync and async methods produce equivalent results."""
    shap_values = [0.5, -0.2, 0.1]
    data_point = [10, 20, 30]
    prediction = 0.9

    # Run sync version
    sync_response = mock_explainer.explain(
        shap_values=shap_values, data_point=data_point, prediction=prediction
    )

    # Reset mock call count
    mock_explainer.llm_agent.run_sync.reset_mock()
    mock_explainer.llm_agent.run.reset_mock()

    # Run async version
    async def run_async():
        return await mock_explainer.explain_async(
            shap_values=shap_values, data_point=data_point, prediction=prediction
        )

    async_response = asyncio.run(run_async())

    # Compare responses
    assert sync_response.summary == async_response.summary
    assert sync_response.detailed_explanation == async_response.detailed_explanation
    assert sync_response.recommendations == async_response.recommendations
    assert sync_response.confidence_level == async_response.confidence_level
