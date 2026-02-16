# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Aim of this repository

A vibe coding optimized repository template for Chemometrics.

### Principles

- **Plan before doing**: Always design your approach before implementation
- **Follow data science best practices**: Reproducibility, documentation, version control

Use beads for issue tracking and git for version control. Follow the commit message guidelines and PR best practices.
See AGENTS.md for more details.

### Chemometrics Project Process

Follow the standard 7-step workflow (problem definition through deployment). See the `chemometrics-shared` skill: `references/workflow.md` for the full process.

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add <package>`
   - Running tools: `uv run <tool>`
   - Upgrading: `uv lock --upgrade-package <package>`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Python code must follow PEP 8 and PEP 484
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 120 chars maximum
   - FORBIDDEN: imports inside functions

3. Testing Requirements
   - Framework: `uv run --frozen pytest`
   - Async testing: use anyio, not asyncio
   - Do not use `Test` prefixed classes, use functions
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

## Commit messages

Follow https://www.conventionalcommits.org/en/v1.0.0/

In brief, the commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Python Tools

## Quarto Authoring Guidelines

1. **Tables**
   - Use itables for all data tables (interactive, sortable)
   - Import: `from itables import show`
   - Display: `show(df, paging=False, searching=False, info=False, ordering=True)`

2. **Figure Layouts**
   - Use vertical (top/down) layouts for multi-panel figures
   - Vertical arrangement improves readability and mobile compatibility
   - FORBIDDEN: Horizontal (left/right) layouts using `layout-ncol`

3. **Subfigure Captions**
   - Use `fig-subcap` directive for proper subcaption naming
   - Example:
     ```yaml
     #| fig-subcap:
     #|   - "First panel description"
     #|   - "Second panel description"
     ```

4. **Interactive Plots**
   - Use plotly for complex interactive visualizations (e.g., PCA or PLS score plots)
   - Import: `import plotly.express as px` or `from plotly.subplots import make_subplots`
   - Display: `fig.show()`

5. **Subplot Handling**
   - Create separate plot calls instead of `plt.subplots(n, m)` with n*m > 1
   - Quarto automatically arranges multiple plots vertically when using `fig-subcap`
   - Preferred pattern:
     ```python
     #| fig-subcap:
     #|   - "First plot"
     #|   - "Second plot"

     fig, ax = plt.subplots(figsize=(10, 6))
     plot_function_1(data, ax=ax)
     plt.show()

     fig, ax = plt.subplots(figsize=(10, 6))
     plot_function_2(data, ax=ax)
     plt.show()
     ```
   - FORBIDDEN: `fig, axes = plt.subplots(2, 1, ...)` with manual subplot indexing

## Code Formatting

1. Ruff
   - Format: `uv run --frozen ruff format .`
   - Check: `uv run --frozen ruff check .`
   - Fix: `uv run --frozen ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: try to use a single line

2. Type Checking
   - Tool: `uv run --frozen pyright`
   - Requirements:
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass

3. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Prettier (YAML/JSON), Ruff (Python)
   - Ruff updates:
     - Check PyPI versions
     - Update config rev
     - Commit config first

## Error Resolution

- **Fix order:** Formatting → Type errors → Linting
- **Line length:** Break with parentheses (strings), multi-line (calls), split (imports)
- **Type errors:** Check Optional types, add narrowing, verify signatures
- **Before commits:** Run formatters, check git status, keep changes minimal

## Exception Handling

- Use `logger.exception("Failed")` (never `logger.error()` with exception, never include `{e}` in message)
- Catch specific exceptions: `OSError`/`PermissionError` for files, `json.JSONDecodeError` for JSON
- **FORBIDDEN** `except Exception:` — unless in top-level handlers
