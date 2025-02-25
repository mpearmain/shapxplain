# SHAPXplain

SHAPXplain combines SHAP (SHapley Additive exPlanations) with Large Language Models (LLMs) to provide natural language explanations of machine learning model predictions. The package helps bridge the gap between technical SHAP values and human-understandable insights.

## Features

- **Natural Language Explanations**: Convert complex SHAP values into clear, actionable explanations using LLMs.
- **Flexible LLM Integration**: Works with any LLM via the `pydantic-ai` interface.
- **Structured Outputs**: Get standardized explanation formats including summaries, detailed analysis, and recommendations.
- **Asynchronous API**: Process explanations in parallel with async/await support.
- **Robust Error Handling**: Built-in retry logic with exponential backoff for API reliability.
- **Batch Processing**: Handle multiple predictions efficiently with both sync and async methods.
- **Confidence Levels**: Understand the reliability of explanations.
- **Feature Interaction Analysis**: Identify and explain how features work together.
- **Data Contracts**: Provide domain-specific context to enhance explanation quality.

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
llm_agent = Agent(model="openai:gpt-4o")  # Or your preferred LLM
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
    prediction_class=prediction_class
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

### Synchronous Batch Processing

```python
batch_response = llm_explainer.explain_batch(
    shap_values_batch=shap_values,
    data_points=X,
    predictions=predictions,
    batch_size=5,  # Optional: control batch size
    additional_context={
        "dataset": "Iris",
        "feature_descriptions": {...}
    }
)

# Access batch results
for response in batch_response.responses:
    print(response.summary)

# Get batch insights
print("Batch Insights:", batch_response.batch_insights)
print("Summary Statistics:", batch_response.summary_statistics)
```

### Asynchronous Batch Processing

For significantly improved performance with large batches:

```python
import asyncio

async def process_batch():
    batch_response = await llm_explainer.explain_batch_async(
        shap_values_batch=shap_values,
        data_points=X,
        predictions=predictions,
        additional_context={
            "dataset": "Iris",
            "feature_descriptions": {...}
        }
    )
    
    # Process results asynchronously
    return batch_response

# Run the async function
batch_results = asyncio.run(process_batch())
```

## Advanced Usage

### Data Contracts for Enhanced Explanations
One of SHAPXplain's most powerful features is the ability to provide domain-specific context through the 
`additional_context` parameter, effectively creating a "data contract" that guides the LLM:

```python
explanation = llm_explainer.explain(
    shap_values=class_shap_values,
    data_point=data_point,
    prediction=prediction,
    additional_context={
        "domain": "medical_diagnosis",
        "feature_descriptions": {
            "glucose": "Blood glucose level in mg/dL. Normal range: 70-99 mg/dL fasting",
            "blood_pressure": "Systolic blood pressure in mmHg. Normal range: <120 mmHg",
            "bmi": "Body Mass Index. Normal range: 18.5-24.9"
        },
        "reference_ranges": {
            "glucose": {"low": "<70", "normal": "70-99", "prediabetes": "100-125", "diabetes": ">126"},
            "blood_pressure": {"normal": "<120", "elevated": "120-129", "stage1": "130-139", "stage2": ">=140"}
        },
        "measurement_units": {
            "glucose": "mg/dL",
            "blood_pressure": "mmHg",
            "bmi": "kg/m²"
        },
        "patient_context": "65-year-old male with family history of type 2 diabetes"
    }
)
```

## Error Handling

```python
try:
    explanation = llm_explainer.explain(
        shap_values=class_shap_values,
        data_point=data_point,
        prediction=prediction
    )
except ValueError as e:
    print(f"Input validation error: {e}")
except RuntimeError as e:
    print(f"LLM query error: {e}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

