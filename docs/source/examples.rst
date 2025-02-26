Examples
========

This section provides detailed examples of using SHAPXplain for different scenarios.

Data Contracts for Enhanced Explanations
---------------------------------------

One of SHAPXplain's most powerful features is the ability to provide domain-specific context through the 
``additional_context`` parameter, effectively creating a "data contract" that guides the LLM:

.. code-block:: python

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

Batch Processing
---------------

For handling multiple predictions efficiently:

.. code-block:: python

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

Asynchronous Batch Processing
----------------------------

For significantly improved performance with large batches:

.. code-block:: python

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

Error Handling
-------------

SHAPXplain includes robust error handling:

.. code-block:: python

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

Logging
-------

Configure logging for better visibility:

.. code-block:: python

    import logging
    from shapxplain import setup_logger

    # Debug level for development
    logger = setup_logger(level=logging.DEBUG)

    # Info level for production
    # logger = setup_logger(level=logging.INFO)

    # Custom format
    # logger = setup_logger(log_format="%(asctime)s - %(levelname)s - %(message)s")