Welcome to SHAPXplain's documentation!
===================================

.. image:: https://img.shields.io/pypi/v/shapxplain.svg
    :target: https://pypi.org/project/shapxplain/
    :alt: PyPI version

.. image:: https://img.shields.io/github/workflow/status/michaelpearmain/shapxplain/CI
    :target: https://github.com/michaelpearmain/shapxplain/actions
    :alt: Build Status

SHAPXplain combines SHAP (SHapley Additive exPlanations) with Large Language Models (LLMs) to provide natural language explanations of machine learning model predictions.

Features
--------

- **Natural Language Explanations**: Convert complex SHAP values into clear, actionable explanations using LLMs.
- **Flexible LLM Integration**: Works with any LLM via the ``pydantic-ai`` interface.
- **Structured Outputs**: Get standardized explanation formats including summaries, detailed analysis, and recommendations.
- **Asynchronous API**: Process explanations in parallel with async/await support.
- **Robust Error Handling**: Built-in retry logic with exponential backoff for API reliability.
- **Batch Processing**: Handle multiple predictions efficiently with both sync and async methods.
- **Confidence Levels**: Understand the reliability of explanations.
- **Feature Interaction Analysis**: Identify and explain how features work together.
- **Data Contracts**: Provide domain-specific context to enhance explanation quality.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`