# SHAPXplain Codebase Guidelines

## Build Commands
- Install: `poetry install`
- Install with docs: `poetry install --with docs`
- Install with all extras: `poetry install --with dev,docs,notebook`
- Run tests: `poetry run pytest`
- Run specific test: `poetry run pytest tests/test_file.py::test_function`
- Run tests with coverage: `poetry run pytest --cov=shapxplain`
- Lint: `poetry run ruff check .`
- Fix lint issues: `poetry run ruff check --fix .`
- Format: `poetry run black .`
- Build docs: `cd docs && poetry run make html`

## Code Style
- **Imports**: Use isort for organization, stdlib first, then third-party, then local
- **Formatting**: Black with default settings (88 char line length)
- **Types**: Use type hints consistently for all function arguments and return values
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
- **Error Handling**: Use explicit exception types with descriptive messages
- **Testing**: Every module should have corresponding tests, use pytest fixtures
- **Async**: Support both sync and async implementations for public API functions

## Project Structure
- `src/shapxplain/`: Core package code
  - `__init__.py`: Package exports and version
  - `explainers.py`: Main ShapLLMExplainer implementation
  - `schemas.py`: Data models using Pydantic
  - `prompts.py`: LLM prompts and templates
  - `utils.py`: Helper functions and logging
- `tests/`: Test files mirroring package structure
- `examples/notebooks/`: Jupyter notebooks with usage examples
- `docs/`: Sphinx documentation
- `.github/workflows/`: CI/CD GitHub Actions

## CI/CD
- GitHub Actions workflows are configured for testing and publishing
- To publish a new version to PyPI:
  1. Update version in `src/shapxplain/__init__.py` and `pyproject.toml`
  2. Create a GitHub Release or use the GitHub Actions manual workflow
  3. The CI will automatically handle testing and publishing

## Documentation
- Built with Sphinx and hosted on ReadTheDocs
- To build locally: `cd docs && poetry run make html`
- Generated docs are in `docs/build/html/`