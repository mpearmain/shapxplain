"""
SHAPXplain Explainers: Integrating SHAP values with LLMs for explainability.
"""

import shap
from shapxplain.prompts import generate_prompt
from shapxplain.llm_client import query_llm

class ShapLLMExplainer:
    def __init__(self, model, api_key, prompt_template=None):
        self.model = model
        self.api_key = api_key
        self.prompt_template = prompt_template or "Explain the impact of {feature} on the prediction."

    def explain(self, data_point, shap_values):
        """
        Generate SHAP values and query the LLM for natural language explanations.
        """
        prompt = generate_prompt(shap_values, data_point, self.prompt_template)
        response = query_llm(prompt, self.api_key)
        return response

    def visualise(self, data_point, shap_values):
        """
        Visualise SHAP values (e.g., using Plotly or Matplotlib).
        """
        shap.waterfall_plot(shap.Explanation(values=shap_values, base_values=self.model.predict_proba(data_point)))
