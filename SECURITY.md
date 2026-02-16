# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by:

1. **Do NOT** open a public issue
2. Email the maintainer directly (see contact information in README.md)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You can expect:
- Acknowledgment within 48 hours
- Assessment and response within 7 days
- Coordinated disclosure once a fix is available

## Security Best Practices for Users

When using this template:

1. **Environment Files**: Never commit `.env` files containing sensitive credentials
2. **Data Privacy**: Review all data files before committing to ensure no personal/sensitive information
3. **Dependencies**: Keep dependencies updated by running `uv lock --upgrade` regularly
4. **API Keys**: Use environment variables or secure vaults for any API keys or secrets
5. **Claude Settings**: The `.claude/settings.json` file may contain local paths - review before sharing

## Known Limitations

This is an educational and research template. For production use:
- Implement proper input validation
- Add authentication/authorization if building web services
- Review data handling practices for compliance with relevant regulations (GDPR, HIPAA, etc.)
- Conduct security audits appropriate to your use case
