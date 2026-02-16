---
name: chemometrics-hybrid-modeling
description: Guide for combining mechanistic models with machine learning (hybrid modeling) in chemometrics and chemical engineering. Covers physics-informed ML, residual modeling, model augmentation, and constraint incorporation for improved predictions and interpretability.
license: MIT
metadata:
  skill-author: Alban Ott
  based-on: Trinh et al. 2021 - Machine Learning in Chemical Product Engineering
---

# Chemometrics Hybrid Modeling

Hybrid modeling combines mechanistic (first-principles) models with machine learning.
Use physics/chemistry knowledge where available; use ML to learn what is unknown or too complex.

## Why Hybrid Models?

| Aspect              | Pure Mechanistic         | Pure Data-Driven          | Hybrid                        |
|---------------------|--------------------------|---------------------------|-------------------------------|
| Interpretability    | High                     | Low (black box)           | Moderate-High                 |
| Extrapolation       | Good within physics      | Poor                      | Better than pure ML           |
| Data requirements   | Low                      | High                      | Moderate                      |
| Flexibility         | Limited to known physics | Learns any pattern        | Physics + data flexibility    |
| Physical validity   | Guaranteed               | May violate laws          | Constrained by design         |
| Development effort  | High (needs domain)      | Low (needs data)          | Moderate                      |

## When to Use This Skill

Use hybrid modeling when:

- You have partial mechanistic knowledge of the system
- Pure mechanistic models are inaccurate (missing phenomena)
- Pure ML models violate physical laws
- Need interpretable predictions that respect physics
- Want to extrapolate beyond training data safely
- Have limited data but know underlying physics
- Modeling chemical processes, reactions, or thermodynamics
- Dealing with Beer-Lambert law deviations in spectroscopy

## Core Hybrid Modeling Approaches

| # | Approach                      | Formula / Idea                                | Best For                                |
|---|-------------------------------|-----------------------------------------------|-----------------------------------------|
| 1 | Residual Modeling (Serial)    | `y = y_mech + ML(x, residual)`                | Decent mech. model with systematic bias |
| 2 | Parallel Hybrid (Ensemble)    | `y = w1*y_mech + w2*y_ML`                     | Both models have merits; uncertain form |
| 3 | Physics-Informed NN (PINNs)   | Physics laws as loss constraints               | PDE-governed systems (diffusion, flow)  |
| 4 | Mechanistic Features for ML   | Engineer physics features as ML inputs         | Partial domain knowledge available      |
| 5 | Constrained Optimization      | ML predictions post-processed for feasibility  | ML violates known inequality bounds     |

**Residual Modeling**: `y_pred = y_mechanistic + ML(x, residual)`. Simplest hybrid -- start here.
Details: [references/approaches.md](references/approaches.md)

**Parallel Hybrid**: `y_pred = w1 * y_mech + w2 * y_ML`. Weighted ensemble of both worlds.
Details: [references/approaches.md](references/approaches.md)

**Physics-Informed NN**: Add physics loss terms (non-negativity, mass balance, PDEs) to training.
Details: [references/approaches.md](references/approaches.md)

**Mechanistic Features**: Compute Arrhenius rates, dimensionless numbers, etc. as ML inputs.
Details: [references/approaches.md](references/approaches.md)

**Constrained Optimization**: Post-process ML predictions with NMF, NNLS, or scipy constraints.
Details: [references/approaches.md](references/approaches.md)

## When to Use What

| Situation                                     | Recommended Approach           |
|-----------------------------------------------|--------------------------------|
| Good mech. model, systematic residuals        | 1 - Residual Modeling          |
| Two decent models, want best of both          | 2 - Parallel Hybrid            |
| PDEs / differential equations govern system   | 3 - Physics-Informed NN        |
| Know relevant dimensionless numbers / rates   | 4 - Mechanistic Features       |
| ML predictions violate physical constraints   | 5 - Constrained Optimization   |
| Not sure where to start                       | 1 - Residual Modeling (simplest)|

## Application Examples

Full worked examples with code comparing pure ML, pure mechanistic, and hybrid approaches.
Details: [references/application-examples.md](references/application-examples.md)

- **NIR Spectroscopy**: Beer-Lambert deviations corrected via residual modeling
- **Chemical Reactor**: Arrhenius kinetics augmented with NN correction
- **Spectral Unmixing**: PLS with mass balance enforcement (normalization + non-negativity)

## Best Practices, Pitfalls, and Advanced Topics

Guidance on validation, interpretation, extrapolation testing, common mistakes, transfer learning, and multi-fidelity modeling.
Details: [references/approaches.md](references/approaches.md)

Key points:
- Always compare pure mechanistic, pure ML, and hybrid (choose hybrid only if it wins)
- Validate physics constraints on predictions (non-negativity, mass balance, range)
- Interpret residual importance to find where physics breaks down
- Test extrapolation performance -- hybrid should degrade gracefully
- Avoid model mismatch (validate mechanistic component first, R2 > 0)
- Balance `lambda_physics` to avoid over-constraining

## See Also

- ML method selection: [../chemometrics-ml-selection/SKILL.md](../chemometrics-ml-selection/SKILL.md)
- Validation strategies: [../chemometrics-shared/references/validation-strategies.md](../chemometrics-shared/references/validation-strategies.md)
- Performance metrics: [../chemometrics-shared/references/performance-metrics.md](../chemometrics-shared/references/performance-metrics.md)

## References

- **Trinh et al. (2021).** Machine Learning in Chemical Product Engineering. *Processes*, 9(8), 1456.
- **von Stosch et al. (2014).** Hybrid semi-parametric modeling in process systems engineering. *Computers & Chemical Engineering*, 60, 86-101.
- **Psichogios & Ungar (1992).** A hybrid neural network-first principles approach to process modeling. *AIChE Journal*, 38(10), 1499-1511.
- **Raissi et al. (2019).** Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
