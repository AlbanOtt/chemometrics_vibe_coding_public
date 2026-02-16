I'm preparing a conference in English.
It can last 2 to 3 hours.
Here's the abstract that was accepted.

## Abstract 

>Vibe coding: an intuitive and creative approach to exploratory data analysis and modeling using AI-assisted programming.
>
>Facilitator: Alban Ott (L'Oréal R&I)
>
>I will introduce the concept of vibe coding: an intuitive and creative approach to exploratory data analysis and modeling using AI-assisted programming. Vibe coding is a workflow where scientists interact fluidly with AI tools, iteratively translating their scientific questions and hypotheses into code in a conversational, collaborative manner. It emphasizes a playful, feedback-driven exploration of data. Vibe coding leverages tools like GitHub Copilot, Gemini, and Cursor to enable Python scripting, making it accessible even for those with minimal programming experience.  
>
>This methodology offers a gentle yet powerful entry point into Python for chemometricians, allowing them to focus on insight and discovery rather than syntax. I'll demonstrate its potential with datasets from either spectroscopy, chromatography, or metabolomics. Vibe coding fosters rapid pattern recognition, dimensionality reduction, and anomaly detection, all while lowering the barrier to technical skill.

## Desired Final State


At the end of the workshop, the idea is to have a GitHub repository with:
- `claude.md` correctly configured for chemometrics
- a set of `skill.md` files starting by retrieving existing skills and creating a few specific to chemometrics
    - [Anthropic's skill repository](https://github.com/anthropics/skills)
        1. skill-creator
        2. pdf
        3. xlsx
    - [K-Dense-AI](https://github.com/K-Dense-AI/claude-scientific-skills)
        1. plotly
        2. polars
        3. scientific-writing
        4. scikit-learn
        5. statistical-analysis
        6. statsmodels
    - [Quarto's authoring skill](https://github.com/posit-dev/skills/blob/main/quarto/authoring/SKILL.md)
    - [Trinh_et_al_2021](assets\Trinh_et_al_2021.pdf)
        - The idea here is truly to show that, based on a scientific article, we can create skills that will help us follow the authors' recommendations more easily
        - analyse the PDF and build skills corresponding to the authors' recommendations
        - use the skill-creator skill to build high-quality skills
- A data folder with a few standard datasets and associated prompts that enable producing typical analyses
- Ideally, we'll have one or two articles from guest speakers and demonstrate that if we reuse data from one of their articles, we can redo the analyses using vibe coding: https://chemom2026.sciencesconf.org/resource/page/id/3
- Beyond the data, we'll produce one or more prompts aimed at obtaining for each dataset:
    - An analysis done in Quarto and Python
    - The most striking results in a Quarto presentation
    - A draft scientific article
- A Quarto presentation corresponding to the presentation I'll deliver at the workshop

## Presentation

### Context

You're helping me design the materials for a 3-hour conference/workshop on vibe coding as a new paradigm for chemometricians.
The audience: chemometricians and PhD students who already know multivariate analysis and Python, but are discovering AI-assisted coding.

Our objectives:
Explain the shift in mindset: moving from "I write the code" to "I design the analysis plan and scientific critique".
Contrast my recommendation: guardrailed vibe coding vs naive vibe coding and using stable pipelines. 
Show how prompts, CLAUDE.md and SKILL.md can encode the "ways of working" of the laboratory for chemometric workflows.
Use only Python in examples with Quarto.

Your tasks in this project:
Help me refine the workshop structure (sections, timing, learning objectives).
Propose slide outlines and exercise ideas before generating full code or long texts.
When I ask for code, generate clear and idiomatic Python, focused on chemometrics (spectral data, PCA, PLS, validation), with concise comments.
Always respect the distinction between: pipelines (stable, industrialised), naive vibe coding (risky), and framed vibe coding (guided by prompts and skills).
Before responding, start by restating in a few sentences your understanding of the workshop's purpose and ask any clarification questions you need.

### Proposed Plan

1. Introduction: Why Vibe Coding for Chemometricians?
2. From Writing Code to Designing Analysis Plans
3. Pipelines vs Vibe Coding
   1. Industrial Chemometric Pipelines in Python  
   2. Naive Vibe Coding (YOLO Mode)  
   3. Framed Vibe Coding with Guardrails
4. The Self-Driving Car Analogy
   1. Vibe Coding: seatbelt unfastened, hands in the air
   2. Framed Vibe Coding: hands on the wheel, seatbelt fastened
   3. Pipelines: car on rails
5. Vibe Coding Pitfalls
    1. First dataset: observe mistakes made in naive vibe coding
    2. Discussion: how to avoid these pitfalls?
6. Building Guardrails for Vibe Coding
    1. Introduction to CLAUDE.md and SKILL.md
    2. gather existing generic skills
    3. gather scientific skills
    4. create chemometrics-specific skills based on literature
7. Framed Vibe Coding in Practice
    1. redo the same dataset analysis with framed vibe coding
    2. compare results, time, errors
8. Framed Vibe Coding for More Complex Techniques
    1. devcontainers
    2. GitHub Actions
9. Wrap-up and Next Steps

### Building the Final GitHub Repository

The objective is to start by building the final GitHub repository that represents the desired end state of the workshop.
I want to create regular "checkpoints", between 3 and 10 at most. From these, participants can restart if they're lost, if their AI went in the wrong direction, or for other reasons.

