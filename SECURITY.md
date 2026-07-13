# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within LAKER-XSA, please send an email to **sachncs@gmail.com**. All security vulnerabilities will be promptly addressed.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### What to include

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Assessment**: We will assess the vulnerability and determine its impact within 5 business days.
- **Fix**: We will work on a fix and aim to release a patch within 14 days of confirmation.
- **Disclosure**: We will coordinate with you on the timing of public disclosure.

## Security Best Practices

When using LAKER-XSA in production:

- Keep dependencies up to date (`pip install --upgrade laker-xsa`)
- Use virtual environments to isolate installations
- Do not expose model checkpoints or training data in public repositories
- Review configuration parameters before deploying to production
- Monitor for unusual memory or compute usage patterns

## Dependency Security

We use [Dependabot](https://github.com/dependabot) to monitor dependencies for known vulnerabilities. Dependabot pull requests are reviewed and merged after CI passes.

## Scope

This security policy applies to the LAKER-XSA package distributed via PyPI and the source code in the official GitHub repository at [github.com/sachncs/laker-xsa](https://github.com/sachncs/laker-xsa).

It does not apply to:

- Third-party packages or services
- Custom deployments or forks
- Usage beyond the intended scope of the library
