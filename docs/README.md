# SHAPXplain Documentation

This directory contains the Sphinx documentation for the SHAPXplain project.

## Building the Documentation

To build the documentation:

1. Install the documentation dependencies:

```bash
poetry install --with docs
```

2. Build the HTML documentation:

```bash
cd docs
make html
```

3. View the documentation:

The built HTML documentation will be in `docs/build/html/index.html`.

## Documentation Structure

- `source/index.rst`: Main documentation index
- `source/api.rst`: API reference
- `source/installation.rst`: Installation instructions
- `source/quickstart.rst`: Quick start guide
- `source/examples.rst`: Detailed examples
- `source/contributing.rst`: Contribution guidelines
- `source/changelog.rst`: Project changelog

## Updating Documentation

When updating the documentation:

1. Make changes to the relevant RST files
2. Build the documentation to verify your changes look correct
3. Commit your changes

## Publishing Documentation

The documentation is automatically built and published to Read the Docs when changes are pushed to the main branch.