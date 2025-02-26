Quickstart
==========

This guide will help you get started with SHAPXplain quickly.

Basic Usage
----------

Here's a complete example using the Iris dataset:

.. code-block:: python

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

Using Logging
------------

SHAPXplain includes a built-in logging system:

.. code-block:: python

    import logging
    from shapxplain import setup_logger

    # Set up logger with custom log level
    logger = setup_logger(level=logging.DEBUG)

    # Now the logger will output detailed debug information
    # Set to WARNING, ERROR, etc. as needed in production

Asynchronous Processing
---------------------

For improved performance, especially with batch processing, you can use the async API:

.. code-block:: python

    import asyncio

    async def process_batch():
        batch_response = await llm_explainer.explain_batch_async(
            shap_values_batch=shap_values,
            data_points=X,
            predictions=predictions
        )
        return batch_response

    # Run the async function
    batch_results = asyncio.run(process_batch())