# SHAPXplain

**SHAPXplain** is a Python package that enhances machine learning interpretability by combining SHAP-based feature
attributions with the explanatory power of Large Language Models (LLMs). With SHAPXplain, you can go beyond numerical
attributions to generate natural language explanations, helping domain experts and stakeholders better understand model
predictions.

## Features

- **SHAP Value Integration**: Compute SHAP values to quantify feature importance.
- **LLM-Driven Explanations**: Use LLMs like OpenAI GPT to generate detailed, human-readable explanations.
- **Extensible Framework**: Supports various machine learning models and SHAP implementations.
- **Customisable Insights**: Tailor explanations using prompt engineering and domain-specific language.
- **End-to-End Workflow**: Seamlessly integrate SHAP, LLMs, and visualisation in a unified pipeline.

---

## Why SHAPXplain?

Machine learning models often act as "black boxes," producing predictions without clear explanations. SHAP values
provide a robust foundation for feature importance, but translating these values into actionable insights can be
challenging for non-technical audiences. **SHAPXplain bridges this gap** by leveraging the natural language capabilities
of LLMs to interpret SHAP values and communicate their meaning effectively.

---

## Installation

Install SHAPXplain using `pip`:

```bash
pip install shapxplain
```

# Quick Start

## 1. Train a Model and Generate SHAP Values

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from shap import TreeExplainer

# Load data and train model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

# Generate SHAP values
explainer = TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

### 2. Generate LLM-Driven Explanations

```python 
from shapxplain.core.main import ShapLLMExplainer

# Instantiate SHAPXplain with OpenAI GPT or other supported LLM
llm_explainer = ShapLLMExplainer(
    model=model,
    shap_explainer=explainer,
    tokenizer="openai",  # Tokenizer for the LLM (e.g., OpenAI GPT)
    prompt_template="Explain the impact of {feature} on the prediction:"
)

# Generate explanations for a specific data point
data_point = X[0]
explanation = llm_explainer.explain(
    data=data_point,
    shap_values=shap_values[0]
)

print(explanation)
```

### 3. Visualise Explanations

```python
llm_explainer.visualise(data_point, shap_values[0])
```

### Example Output

### Example Output

Suppose you're analysing a RandomForest model predicting species in the Iris dataset. For a specific data point:

#### SHAP Values:

| Feature      | SHAP Value | Impact on Prediction |
|--------------|------------|----------------------|
| Petal Length | +0.75      | Strong Positive      |
| Sepal Width  | -0.20      | Weak Negative        |
| Sepal Length | +0.05      | Marginal Positive    |
| Petal Width  | +0.10      | Moderate Positive    |

#### LLM Explanation:

- "The **Petal Length** strongly increases the likelihood of this sample belonging to the target class because longer
  petals are highly indicative of this species."
- "The **Sepal Width** slightly reduces the likelihood, likely due to overlapping characteristics with other species."
- "Overall, the prediction is primarily driven by Petal Length and moderately influenced by Petal Width."

---

### Advanced Usage

#### Custom Prompt Templates

You can design your own prompt templates to tailor the explanation style:

```python
prompt_template = """
You are an AI assistant specialising in machine learning. 
Given the SHAP values below, explain their impact on the prediction in simple terms.
Feature: {feature}, SHAP Value: {shap_value}
"""

llm_explainer.set_prompt_template(prompt_template)
```

```python
prompt_template = """
You are an AI assistant specialising in machine learning. 
Given the SHAP values below, explain their impact on the prediction in simple terms.
Feature: {feature}, SHAP Value: {shap_value}
"""
llm_explainer.set_prompt_template(prompt_template)
```

### Using Different LLMs

SHAPXplain supports multiple LLMs:

```python
llm_explainer.set_tokenizer("huggingface")  # Use HuggingFace transformers
```

Batch Explanations
Generate explanations for multiple data points:

```python
batch_explanations = llm_explainer.explain_batch(data=X, shap_values=shap_values)
```