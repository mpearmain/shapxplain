# SHAPXplain

**SHAPXplain** combines SHAP (SHapley Additive exPlanations) with Large Language Models (LLMs) to provide natural language explanations of machine learning model predictions. The package helps bridge the gap between technical SHAP values and human-understandable insights.

## Features

- **Natural Language Explanations**: Convert complex SHAP values into clear, actionable explanations using LLMs
- **Flexible LLM Integration**: Works with any LLM via the pydantic-ai interface
- **Structured Outputs**: Get standardized explanation formats including summaries, detailed analysis, and recommendations
- **Batch Processing**: Handle multiple predictions efficiently
- **Confidence Levels**: Understand the reliability of explanations
- **Feature Interaction Analysis**: Identify and explain how features work together

## Installation

```bash
pip install shapxplain
```

## Quick Start

Here's a complete example using the Iris dataset:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from shap import TreeExplainer
from shapxplain import ShapLLMExplainer
from pydantic_ai import Agent

# Load data and train model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Generate SHAP values
explainer = TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Create LLM agent and SHAPXplain explainer
llm_agent = Agent(model="openai:gpt-4")  # Or your preferred LLM
llm_explainer = ShapLLMExplainer(
    model=model,
    llm_agent=llm_agent,
    feature_names=data.feature_names,
    significance_threshold=0.1
)

# Explain a single prediction
data_point = X[0]
prediction_probs = model.predict_proba(data_point.reshape(1, -1))[0]
predicted_class_idx = model.predict(data_point.reshape(1, -1))[0]
prediction_class = data.target_names[predicted_class_idx]

# Get class-specific SHAP values
class_shap_values = shap_values[predicted_class_idx][0]

# Generate explanation
explanation = llm_explainer.explain(
    shap_values=class_shap_values,
    data_point=data_point,
    prediction=prediction_probs[predicted_class_idx],
    prediction_class=prediction_class,
    additional_context={
        "dataset": "Iris",
        "feature_descriptions": {
            "sepal length": "Length of the sepal in cm",
            "sepal width": "Width of the sepal in cm",
            "petal length": "Length of the petal in cm",
            "petal width": "Width of the petal in cm"
        }
    }
)

# Access different parts of the explanation
print("Summary:", explanation.summary)
print("\nDetailed Explanation:", explanation.detailed_explanation)
print("\nRecommendations:", explanation.recommendations)
print("\nConfidence Level:", explanation.confidence_level)
```

## Explanation Structure

The package provides structured explanations with the following components:

```python
class SHAPExplanationResponse:
    summary: str  # Brief overview of key drivers
    detailed_explanation: str  # Comprehensive analysis
    recommendations: List[str]  # Actionable insights
    confidence_level: str  # high/medium/low
    feature_interactions: Dict[str, str]  # How features work together
    features: List[SHAPFeatureContribution]  # Detailed feature impacts
```

## Batch Processing

Process multiple predictions efficiently:

```python
batch_response = llm_explainer.explain_batch(
    shap_values_batch=shap_values,
    data_points=X,
    predictions=predictions,
    batch_size=5  # Optional: control batch size
)

# Access batch results
for response in batch_response.responses:
    print(response.summary)

# Get batch insights
print("Batch Insights:", batch_response.batch_insights)
print("Summary Statistics:", batch_response.summary_statistics)
```

## Advanced Usage

### Custom LLM Configuration

```python
# Use a different LLM model
llm_agent = Agent(
    model="anthropic:claude-v2",
    system_prompt="You are an expert in explaining machine learning predictions..."
)

llm_explainer = ShapLLMExplainer(
    model=model,
    llm_agent=llm_agent,
    feature_names=feature_names,
    significance_threshold=0.15  # Adjust significance threshold
)
```

### Additional Context

Provide domain-specific context for better explanations:

```python
explanation = llm_explainer.explain(
    shap_values=class_shap_values,
    data_point=data_point,
    prediction=prediction,
    additional_context={
        "domain": "medical_diagnosis",
        "feature_descriptions": feature_descriptions,
        "reference_ranges": reference_ranges,
        "measurement_units": units
    }
)
```
