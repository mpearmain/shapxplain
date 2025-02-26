# SHAPXplain Codebase Guidelines

## Build Commands
- Install: `poetry install`
- Run tests: `poetry run pytest`
- Run specific test: `poetry run pytest tests/test_file.py::test_function`
- Run tests with coverage: `poetry run pytest --cov=shapxplain`
- Lint: `poetry run ruff check .`
- Format: `poetry run black .`
- Sort imports: `poetry run isort .`

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
- `tests/`: Test files mirroring package structure
- `examples/notebooks/`: Jupyter notebooks with usage examples