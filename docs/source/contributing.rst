Contributing
============

Contributions to SHAPXplain are welcome! This page provides guidelines for contributing to the project.

Setting Up Development Environment
--------------------------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/yourusername/shapxplain.git
       cd shapxplain

3. Install development dependencies:

   .. code-block:: bash

       poetry install

4. Set up pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Development Guidelines
--------------------

Code Style
~~~~~~~~~

- Use Black for code formatting
- Use isort for import sorting
- Use type hints throughout the codebase
- Follow PEP 8 guidelines

Testing
~~~~~~~

- Write tests for all new features and bug fixes
- Maintain high test coverage
- Run tests before submitting a pull request:

  .. code-block:: bash

      poetry run pytest

Documentation
~~~~~~~~~~~~

- Write comprehensive docstrings for all public modules, classes, and functions
- Update documentation when adding or changing features
- Build docs locally to verify your changes:

  .. code-block:: bash

      cd docs
      make html

Pull Request Process
------------------

1. Create a new branch for your feature:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes and commit with descriptive messages
3. Push your branch and create a pull request
4. Ensure the CI pipeline passes
5. Wait for review and address any feedback

Release Process
-------------

1. Update version in both pyproject.toml and __init__.py
2. Update changelog.rst with the new version and changes
3. Create a new GitHub release with detailed release notes
4. CI will automatically publish to PyPI