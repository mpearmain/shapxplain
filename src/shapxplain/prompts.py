"""
Prompts module for generating SHAP-to-LLM templates.
"""
def generate_prompt(shap_values, data_point, template):
    """
    Generate a natural language prompt based on SHAP values and a template.
    """
    prompt = template.format(
        shap_values=shap_values,
        data_point=data_point
    )
    return prompt
