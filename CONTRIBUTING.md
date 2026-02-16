# Contributing to Chemometrics Vibe Coding

Thank you for your interest in contributing! This repository serves both as a teaching resource and a production chemometrics analysis workflow.

## Ways to Contribute

- 🐛 Report bugs or issues
- 💡 Suggest enhancements or new features
- 📚 Improve documentation
- 🧪 Add tests or improve test coverage
- 🔬 Contribute new analysis examples or skills
- 🎓 Share how you've used this template

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Quarto](https://quarto.org/) (for documentation)
- Git
- (Optional) [Claude Code CLI](https://github.com/anthropics/claude-code) for AI-assisted development

### Setting Up Your Development Environment

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/chemometrics_vibe_coding.git
cd chemometrics_vibe_coding

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest

# Optional: Render the Quarto book
cd reports
uv run quarto render
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow the guidelines in [CLAUDE.md](CLAUDE.md):

**Code Quality:**
- Add type hints to all functions
- Follow PEP 8 and PEP 484
- Keep functions focused and small
- Line length: 120 characters maximum
- NO imports inside functions

**Testing:**
- Write tests for new features
- Add regression tests for bug fixes
- Run tests: `uv run --frozen pytest`
- Use function-based tests (not `Test` classes)

**Code Formatting:**
```bash
# Format code
uv run --frozen ruff format .

# Check for issues
uv run --frozen ruff check .

# Fix auto-fixable issues
uv run --frozen ruff check . --fix

# Type checking
uv run --frozen pyright
```

### 3. Commit Your Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (formatting, missing semicolons, etc.)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples:**
```bash
git commit -m "feat: add PLS-DA permutation test function"
git commit -m "fix: correct Pareto scaling normalization"
git commit -m "docs: update preprocessing tutorial with new workflow"
```

### 4. Push and Create a Pull Request

```bash
git push -u origin feature/your-feature-name
```

Then create a pull request on GitHub. The PR template will guide you through the checklist.

## Code Style Guidelines

### Python

```python
# Good - type hints, clear naming, docstring
def calculate_vip_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2
) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        n_components: Number of PLS components

    Returns:
        VIP scores for each feature (n_features,)
    """
    # Implementation here
    pass

# Bad - no types, unclear naming
def calc(x, y):
    pass
```

### Quarto Documents

```python
#| label: fig-pca-scores
#| fig-cap: "PCA scores plot showing sample separation"
#| fig-subcap:
#|   - "PC1 vs PC2"
#|   - "PC2 vs PC3"

# Use itables for interactive tables
from itables import show
show(df, paging=False, searching=False, ordering=True)

# Use plotly for interactive plots
import plotly.express as px
fig = px.scatter(df, x='PC1', y='PC2', color='group')
fig.show()

# Vertical layouts only (no layout-ncol)
```

See [CLAUDE.md](CLAUDE.md) for full Quarto guidelines.

## Testing Guidelines

### Writing Tests

```python
# tests/test_preprocessing.py
import numpy as np
from src.preprocessing import pqn_normalize

def test_pqn_normalize_preserves_median():
    """PQN normalization should preserve the median of reference sample."""
    X = np.random.rand(10, 100)
    X_norm = pqn_normalize(X, reference_idx=0)

    # Reference sample should have median close to 1
    assert np.abs(np.median(X_norm[0, :]) - 1.0) < 0.01

def test_pqn_normalize_handles_zeros():
    """PQN normalization should handle zero values correctly."""
    X = np.array([[1, 0, 2], [3, 0, 4]])
    X_norm = pqn_normalize(X)

    assert not np.any(np.isnan(X_norm))
    assert not np.any(np.isinf(X_norm))
```

### Running Tests

```bash
# Run all tests
uv run --frozen pytest

# Run specific test file
uv run --frozen pytest tests/test_preprocessing.py

# Run with coverage
uv run --frozen pytest --cov=src

# Run with verbose output
uv run --frozen pytest -v
```

## Documentation Guidelines

### Code Documentation

- Use docstrings for all public functions
- Include parameter types and return types
- Provide examples for complex functions
- Explain the "why" not just the "what"

### Quarto Reports

- Start each chapter with learning objectives
- Explain the scientific reasoning behind each step
- Include interpretation of results
- Use callouts for important notes

```markdown
::: {.callout-note}
PQN normalization assumes that the majority of features do not change between samples.
:::

::: {.callout-warning}
Small sample sizes (n < 30) may lead to overfitting. Use LOSO-CV for validation.
:::
```

## Contributing New Skills

Skills are domain-specific knowledge modules in `.claude/skills/`. To create a new skill:

```bash
# Use the skill-creator skill
uv run python .claude/skills/skill-creator/scripts/init_skill.py my-new-skill

# Edit the generated SKILL.md
# Add reference files as needed
# Test with Claude Code CLI
```

See [skill-creator SKILL.md](.claude/skills/skill-creator/SKILL.md) for detailed guidance.

## Pull Request Process

1. **Ensure all checks pass:**
   - Tests pass locally
   - Code is formatted (ruff)
   - No linting errors
   - Type checking passes

2. **Update documentation:**
   - Add docstrings to new functions
   - Update relevant Quarto chapters
   - Update README if adding features

3. **Fill out the PR template:**
   - Describe what changed and why
   - Link related issues
   - Complete the checklist

4. **Wait for CI/CD:**
   - All workflow checks must pass
   - Tests run on Python 3.9, 3.10, 3.11, 3.12
   - Quarto book builds successfully

5. **Respond to reviews:**
   - Address reviewer comments
   - Push new commits to the same branch
   - Mark conversations as resolved

6. **Merge:**
   - Maintainer will merge once approved
   - Delete your feature branch after merge

## Reporting Issues

### Bug Reports

Include:
- Python version: `python --version`
- uv version: `uv --version`
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Use case and motivation
- Proposed implementation (if any)
- Examples from other tools
- Impact on existing functionality

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to the maintainer.

## Questions?

- **Documentation:** Start with [README.md](README.md) and [CLAUDE.md](CLAUDE.md)
- **Issues:** Check [existing issues](https://github.com/AlbanOtt/chemometrics_vibe_coding/issues)
- **Contact:** See maintainer contact information in [README.md](README.md)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Thank you for contributing to making chemometrics analysis more accessible!** 🎓🔬
