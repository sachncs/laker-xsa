# Contributing to LAKER-XSA

Thank you for considering contributing to LAKER-XSA! This document outlines the process for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/laker-xsa.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Run the tests to verify everything works: `pytest tests/ -v`

Or use the setup script:

```bash
bash setup.sh
```

## Branch Naming

Use descriptive branch names with a prefix:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |
| `perf/` | Performance improvements |

Example: `feat/add-flash-attention-support`

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, missing semicolons, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (dependencies, CI, etc.) |
| `ci` | Changes to CI configuration |

### Examples

```
feat(attention): add FlashAttention integration
fix(solver): resolve PCG divergence on ill-conditioned kernels
docs: update installation instructions for Python 3.12
test(laker): add edge case tests for empty sequences
chore: update pytest to 8.0
```

## Development Workflow

1. Create a branch: `git checkout -b feat/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run linting: `pylint laker_xsa/ --rcfile=pyproject.toml`
5. Run type checking: `mypy laker_xsa/ --ignore-missing-imports`
6. Format code: `black laker_xsa/ tests/`
7. Commit your changes using conventional commit format
8. Push to your fork: `git push origin feat/your-feature-name`
9. Open a Pull Request

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use type hints for all function signatures
- Write [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep functions focused and small (< 50 lines preferred)
- Maximum line length: 88 characters (enforced by black)
- Use `from __future__ import annotations` in all Python files

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=laker_xsa --cov-report=html

# Run specific test file
pytest tests/test_attention.py -v

# Run tests matching a pattern
pytest tests/ -k "test_laker" -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Name test functions `test_<what_it_tests>`
- Use descriptive test names that explain the expected behavior
- Test both the happy path and edge cases
- Mock external dependencies when appropriate

## Pull Request Guidelines

- Keep PRs focused on a single concern
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Write a clear PR description explaining what and why
- Reference related issues using `Fixes #123` or `Relates to #123`
- Ensure all CI checks pass before requesting review

## Reporting Issues

When reporting issues, please include:

- Python version (`python --version`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any relevant error messages or stack traces

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

Feel free to open an issue for any questions or discussions about the project.
