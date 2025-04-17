# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Install dev dependencies: `pip install -e ".[dev]"`
- Run all tests: `pytest`
- Run a single test: `pytest test/test_file.py::test_function`
- Check types: `mypy src/`
- Format code: `black src/ test/`
- Sort imports: `isort src/ test/`

## Code Style Guidelines
- Use Python 3.8+ features
- Imports: Group standard library, third-party, and local imports
- Formatting: Follow Black style guide (line length 88)
- Types: Use typing annotations for all function parameters and return values
- Naming: snake_case for variables/functions, PascalCase for classes
- Docstrings: Use Google style docstrings with Args/Returns sections
- Error handling: Use explicit exception types, add helpful error messages
- Async: All bot interaction methods should be async
- Testing: Write tests for all new functionality