# Contributing to LAKER-XSA

Thank you for considering contributing to LAKER-XSA! This document outlines the process for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/laker-xsa.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linting: `pylint src/laker_xsa/` and `mypy src/laker_xsa/`
5. Format code: `black src/laker_xsa/ tests/`
6. Commit your changes: `git commit -m "Add your message here"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write Google-style docstrings
- Keep functions focused and small (< 50 lines preferred)
- Write tests for new functionality

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=laker_xsa tests/

# Run specific test file
pytest tests/test_attention.py
```

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any relevant error messages or stack traces

## Pull Request Guidelines

- Keep PRs focused on a single concern
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Write a clear PR description

## Questions?

Feel free to open an issue for any questions or discussions about the project.
