# Vibe Coding for Chemometrics

A vibe coding template with a complete LC-MS metabolomics analysis and workshop materials, built entirely with Claude Code CLI.

## Overview

This repository serves two purposes: a **teaching resource** for AI-assisted scientific analysis and a **production artifact** — a complete metabolomics study you can read, reproduce, and extend.

The case study analyzes diphenhydramine pharmacokinetics in plasma and skin using LC-MS metabolomics data from 7 subjects across 6 timepoints. The analysis covers the full chemometrics workflow from raw data preprocessing through proxy biomarker discovery, implemented as a 7-chapter Quarto book backed by tested Python modules.

Everything here was built with Claude Code CLI guided by domain-specific skills and project configuration (`CLAUDE.md`). The key workshop message: vibe coding is not about writing code faster — it's a different way of working where you describe intent and the AI handles implementation within guardrails you define.

## What's Inside

### The Analysis

- **6 Python modules** (`src/`): preprocessing, exploratory data analysis, targeted drug detection, multivariate modeling (PCA/PLS-DA), proxy biomarker discovery, and sampling location comparison
- **7-chapter Quarto book** (`reports/`): self-contained HTML tutorials walking through the full analysis with interactive plots and interpretation
- **Test suite** (`tests/`): automated tests for the analysis modules
- **Real LC-MS dataset**: preprocessed GNPS quantification data from a diphenhydramine PK study

### The Workshop

- **Reveal.js presentation** (`presentation/`): self-contained HTML slide deck covering vibe coding methodology, live demos, and practical guidance
- **7 Claude Code skills** (`.claude/skills/`): domain expertise that guides the AI during analysis — from metabolomics processing to validation strategies
- **Project configuration** (`CLAUDE.md`, `AGENTS.md`): examples of how to frame AI-assisted workflows with coding standards, testing requirements, and domain conventions
- **Example prompts** (`prompts/`): workshop objectives and analysis prompts

## Repository Structure

```
chemometrics_vibe_coding/
├── src/                                # Analysis modules
│   ├── preprocessing.py                # Data loading, blank filtering, PQN normalization
│   ├── eda.py                          # Intensity distributions, PCA QC, outlier detection
│   ├── drug_detection.py               # m/z search, PK curves, plasma-skin comparison
│   ├── multivariate.py                 # PCA trajectories, PLS-DA, VIP scores, S-plot
│   ├── biomarkers.py                   # Correlation screening, PLS regression, LOSO CV
│   └── location_comparison.py          # Forearm vs forehead feature matching and evaluation
├── tests/                              # Automated tests
│   ├── test_eda.py
│   ├── test_multivariate.py
│   ├── test_biomarkers.py
│   └── test_location_comparison.py
├── reports/                            # Quarto book (7 chapters)
│   ├── _quarto.yml                     # Book configuration
│   ├── index.qmd                       # Welcome and study design
│   ├── preprocessing_tutorial.qmd      # Data preprocessing
│   ├── eda_tutorial.qmd                # Exploratory data analysis
│   ├── drug_detection_tutorial.qmd     # Diphenhydramine detection and PK
│   ├── multivariate_tutorial.qmd       # Multivariate modeling
│   ├── biomarkers_tutorial.qmd         # Proxy biomarker discovery
│   └── location_comparison_tutorial.qmd # Sampling location comparison
├── presentation/                       # Workshop slides
│   ├── workshop_presentation.qmd       # Quarto reveal.js source
│   └── workshop_presentation.html      # Rendered presentation
├── .claude/
│   └── skills/                         # Claude Code domain skills
│       ├── chemometrics-shared/        # Shared foundations
│       ├── chemometrics-ms-metabolomics/ # LC-MS processing
│       ├── chemometrics-ml-selection/  # ML method selection
│       ├── chemometrics-validation/    # Model validation
│       ├── chemometrics-hybrid-modeling/ # Physics-informed ML
│       ├── quarto-authoring/           # Report authoring
│       └── skill-creator/              # Creating new skills
├── data/                               # Preprocessed LC-MS data
│   ├── preprocessing_result.pkl        # Forearm + forehead datasets
│   └── DATASETS_AND_PROMPTS.md         # Dataset catalog
├── prompts/                            # Workshop prompts and objectives
├── assets/                             # Images and reference papers
├── CLAUDE.md                           # Project configuration for Claude Code
├── AGENTS.md                           # Agent workflow instructions
├── pyproject.toml                      # Python project config (uv)
└── LICENSE                             # MIT License
```

## Quick Start

### View the outputs (no installation needed)

**📚 Read the analysis online**: The complete metabolomics tutorial is available at:
**https://albanott.github.io/chemometrics_vibe_coding/**

No installation, no setup, just click and read. All interactive plots and tables work directly in your browser.

**🎤 View the presentation**: Download `presentation/workshop_presentation.html` and open in a browser.

### Reproduce or extend the analysis

Requires Python 3.9+, [uv](https://docs.astral.sh/uv/), and [Quarto](https://quarto.org/docs/get-started/).

```bash
# Clone the repository (~100MB, optimized for classroom use)
git clone https://github.com/AlbanOtt/chemometrics_vibe_coding.git
cd chemometrics_vibe_coding

# Install dependencies
uv sync

# Run tests
uv run pytest

# Build the book locally (optional)
cd reports
uv run quarto render

# Or preview with live reload
uv run quarto preview
```

**Note for students**: The repository is lightweight (~100MB) because built HTML files are automatically deployed to GitHub Pages rather than included in git. You can view the analysis online or build it locally.

## Claude Code Skills

The `.claude/skills/` directory contains 7 domain skills that guide AI behavior during analysis:

| Skill | Purpose |
|-------|---------|
| `chemometrics-shared` | Cross-validation, metrics, overfitting prevention |
| `chemometrics-ms-metabolomics` | LC-MS data processing and analysis workflows |
| `chemometrics-ml-selection` | Choosing appropriate ML methods for the data |
| `chemometrics-validation` | Validation strategies, performance reporting |
| `chemometrics-hybrid-modeling` | Combining mechanistic models with ML |
| `quarto-authoring` | Scientific report and book authoring |
| `skill-creator` | Creating new domain-specific skills |

Skills act as persistent domain expertise: they are loaded into the AI's context when relevant, ensuring best practices are followed without manual prompting.

## Citation

This repository accompanies a pre-congress training session at [Chimiométrie 2026](https://chemom2026.sciencesconf.org/resource/page/id/2) (February 17, 2026).

```bibtex
@misc{ott2026vibecoding,
  title={Vibe coding: an intuitive and creative approach to exploratory data
         analysis and modeling using {AI}-assisted programming},
  author={Ott, Alban},
  year={2026},
  howpublished={\url{https://github.com/AlbanOtt/chemometrics_vibe_coding}},
  note={Pre-congress training, Chimiom\'{e}trie 2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

**Alban Ott** — [GitHub](https://github.com/AlbanOtt) | [LinkedIn](https://www.linkedin.com/in/alban-ott-25222396/)
