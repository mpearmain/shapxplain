Installation
============

SHAPXplain can be installed via pip:

.. code-block:: bash

    pip install shapxplain

Requirements
-----------

SHAPXplain requires Python 3.12 or later and has the following core dependencies:

- ``pydantic-ai``: Interface with LLMs
- ``shap``: Calculate SHAP values
- ``scikit-learn``: Support for model interaction
- ``numba``: Accelerated computations
- ``plotly``: Visualization

Development Installation
-----------------------

For development, you can clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/michaelpearmain/shapxplain.git
    cd shapxplain
    poetry install

This installs all the required dependencies, including the development dependencies.

API Key Setup
------------

SHAPXplain uses LLMs through the pydantic-ai interface. Depending on your chosen LLM provider, 
you'll need to set up the appropriate API key:

- For OpenAI: ``OPENAI_API_KEY``
- For Anthropic: ``ANTHROPIC_API_KEY``
- For other providers, refer to the pydantic-ai documentation

You can set these directly in your environment variables or use a ``.env`` file:

.. code-block:: python

    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()