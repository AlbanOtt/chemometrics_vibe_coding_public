# Using Checkpoints in the Workshop

This guide explains how to use Git checkpoints (tags/releases) to restart workshop exercises from specific points if you get stuck or want to skip ahead.

## Overview

The workshop includes 4 checkpoints for the hands-on exercise, each representing a completed stage of the analysis:

| Checkpoint | Description | What's Completed | What's Next |
|------------|-------------|------------------|-------------|
| **checkpoint-1** | Data Exploration | Data loaded, visualized, outliers checked | Apply preprocessing |
| **checkpoint-2** | Preprocessing | SNV/MSC applied, data ready | Build models |
| **checkpoint-3** | Model Training | PLS/SVM/RF trained, CV done | Final evaluation |
| **checkpoint-4** | Final Analysis | All evaluation, interpretation complete | Report writing |

## Why Use Checkpoints?

- **Get unstuck:** If you encounter errors or get lost, restart from the last checkpoint
- **Time management:** Skip ahead if running short on time
- **Compare approaches:** Try different paths from the same starting point
- **Learn incrementally:** Focus on one stage at a time

## Creating Checkpoints (For Instructors)

### Step 1: Complete Each Analysis Stage

Work through the analysis and complete checkpoint 1:

```bash
# Complete data exploration stage
# - Load data
# - Create visualizations
# - Check for outliers
# - Save exploratory plots to results/figures/
```

### Step 2: Commit Your Work

```bash
git add .
git commit -m "Checkpoint 1: Data exploration complete

- Loaded NIR spectra and reference values
- Created spectra overview plot
- Performed PCA for outlier detection
- Identified 2 potential outliers (samples 23, 67)
- Saved exploratory figures to results/figures/"
```

### Step 3: Create Git Tag

```bash
git tag -a checkpoint-1 -m "Checkpoint 1: Data exploration complete"
```

### Step 4: Push Tag to Remote

```bash
git push origin checkpoint-1
```

### Step 5: Repeat for All Checkpoints

```bash
# Checkpoint 2: Preprocessing
git add .
git commit -m "Checkpoint 2: Preprocessing applied"
git tag -a checkpoint-2 -m "Checkpoint 2: Preprocessing applied"
git push origin checkpoint-2

# Checkpoint 3: Model training
git add .
git commit -m "Checkpoint 3: Models trained and validated"
git tag -a checkpoint-3 -m "Checkpoint 3: Models trained"
git push origin checkpoint-3

# Checkpoint 4: Final analysis
git add .
git commit -m "Checkpoint 4: Final evaluation and interpretation"
git tag -a checkpoint-4 -m "Checkpoint 4: Final analysis complete"
git push origin checkpoint-4
```

### Step 6: Create GitHub Releases (Optional)

For better visibility, convert tags to releases on GitHub:

1. Go to your repository on GitHub
2. Click "Releases" in the right sidebar
3. Click "Create a new release"
4. Select the tag (e.g., `checkpoint-1`)
5. Add release title: "Checkpoint 1: Data Exploration"
6. Add description with details about what's included
7. Attach any relevant files (optional)
8. Click "Publish release"

Repeat for all 4 checkpoints.

## Using Checkpoints (For Participants)

### Viewing Available Checkpoints

```bash
# List all tags
git tag

# Output:
# checkpoint-1
# checkpoint-2
# checkpoint-3
# checkpoint-4

# View tag details
git show checkpoint-1
```

### Restoring from a Checkpoint

#### Method 1: Checkout Tag (Detached HEAD)

This creates a detached HEAD state - good for viewing but not for making changes:

```bash
# Restore checkpoint 2
git checkout checkpoint-2

# You're now at the preprocessing stage
# View files, run code, but commits won't be on a branch

# When done, return to main
git checkout main
```

#### Method 2: Create Branch from Checkpoint (Recommended)

This allows you to make changes and continue working:

```bash
# Create a new branch from checkpoint 2
git checkout -b my-analysis checkpoint-2

# Now work on this branch
# Make changes, commit as needed
git add modified_files
git commit -m "Trying alternative preprocessing method"

# If you want to start fresh from checkpoint 2 again
git checkout main
git checkout -b attempt-2 checkpoint-2
```

#### Method 3: Reset to Checkpoint (Destructive)

⚠️ **Warning:** This discards your current work! Only use if you want to start over.

```bash
# Save current work first (optional)
git stash

# Hard reset to checkpoint 2
git reset --hard checkpoint-2

# To recover your stashed work later
git stash pop
```

### Typical Workshop Workflow

#### Scenario 1: Following Along

```bash
# Start fresh
git checkout main
cd examples/wine_authentication

# Work on checkpoint 1...
# If you get stuck or instructor moves ahead:
git checkout checkpoint-1

# Continue to checkpoint 2...
git checkout checkpoint-2

# And so on
```

#### Scenario 2: Running Out of Time

```bash
# You're at checkpoint 1 but session is ending soon
# Jump to checkpoint 3 to see model results
git checkout checkpoint-3

# Now explore the trained models
```

#### Scenario 3: Comparing Approaches

```bash
# Create branch from checkpoint 2 to try PLS
git checkout -b pls-approach checkpoint-2
# ... work on PLS ...

# Create another branch to try SVM
git checkout -b svm-approach checkpoint-2
# ... work on SVM ...

# Compare results between branches
git diff pls-approach svm-approach
```

#### Scenario 4: Recovering from Mistakes

```bash
# You made changes that broke things
# Start fresh from last good checkpoint
git status  # See what you changed
git stash   # Save changes (just in case)
git reset --hard checkpoint-2  # Go back to known good state
```

## Checkpoint Contents

### Checkpoint 1: Data Exploration

**Files created:**
```
results/
└── figures/
    ├── 01_spectra_overview.png       # All spectra overlaid
    ├── 02_pca_scores.png             # PCA scores plot
    └── 03_reference_distribution.png # Histogram of reference values

data_processed/
└── outliers_identified.csv           # List of outlier samples
```

**Code artifacts:**
- Data loading functions
- Exploratory plotting functions
- Outlier detection (Mahalanobis distance)

**Analysis decisions documented:**
- Which samples are outliers (if any)
- Whether to remove outliers or keep
- Initial observations about data quality

### Checkpoint 2: Preprocessing

**Files created:**
```
results/
└── figures/
    ├── 04_preprocessing_comparison.png  # Raw vs SNV vs MSC
    └── 05_preprocessed_spectra.png      # Final preprocessed spectra

data_processed/
├── X_train_raw.csv
├── X_test_raw.csv
├── X_train_snv.csv                      # SNV preprocessed
├── X_test_snv.csv
├── y_train.csv
└── y_test.csv
```

**Code artifacts:**
- SNV and/or MSC preprocessing functions
- Train/test split (80/20, stratified)
- Preprocessing visualization

**Analysis decisions documented:**
- Which preprocessing method chosen (SNV vs MSC)
- Train/test split details (random seed, stratification)
- Any samples removed

### Checkpoint 3: Model Training

**Files created:**
```
results/
├── figures/
│   ├── 06_cv_components.png          # CV scores vs n_components
│   ├── 07_learning_curve.png         # Optional: learning curve
│   └── 08_validation_curve.png       # Optional: validation curve
└── models/
    ├── pls_model.pkl                  # Trained PLS model
    ├── svr_model.pkl                  # Optional: SVM model
    ├── rf_model.pkl                   # Optional: RF model
    ├── scaler.pkl                     # Preprocessing scaler
    └── model_metadata.json            # Hyperparameters, CV scores

reports/
└── cv_results.csv                     # Cross-validation results
```

**Code artifacts:**
- Cross-validation loops
- Hyperparameter tuning (n_components, C, gamma, etc.)
- Model training code
- Model persistence (pickle)

**Analysis decisions documented:**
- Optimal number of PLS components
- Which model(s) selected for final evaluation
- Cross-validation scores

### Checkpoint 4: Final Analysis

**Files created:**
```
results/
├── figures/
│   ├── 09_predicted_vs_actual.png    # Main performance plot
│   ├── 10_loadings_plot.png          # PLS loadings
│   ├── 11_vip_scores.png             # Variable importance
│   ├── 12_residuals.png              # Residual analysis
│   └── 13_confusion_matrix.png       # For classification
└── reports/
    ├── analysis_report.md             # Complete analysis report
    ├── validation_metrics.csv         # All performance metrics
    └── model_interpretation.md        # Interpretation notes
```

**Code artifacts:**
- Test set evaluation
- Performance metrics calculation (RMSEP, R², RPD)
- Visualization functions
- Report generation

**Analysis decisions documented:**
- Final model selection rationale
- Performance assessment (meets requirements?)
- Interpretation of important features
- Recommendations for deployment

## Best Practices

### For Instructors

1. **Test checkpoints before workshop:**
   ```bash
   git checkout checkpoint-1
   # Verify all files are present
   # Test that next steps work
   ```

2. **Document checkpoint contents:**
   - List files created
   - Note any important variables
   - Explain analysis state

3. **Keep checkpoints focused:**
   - Each represents a logical stage
   - Not too granular (overwhelms) or coarse (less useful)

4. **Provide checkpoint guide:**
   - Distribute this CHECKPOINTS.md
   - Show live demo of restoring checkpoint

5. **Include checkpoint prompts:**
   - In each checkpoint directory, add `NEXT_STEPS.md`
   - Guides participants on what to do next

### For Participants

1. **Don't be afraid to use checkpoints:**
   - They're there to help you learn
   - No penalty for using them

2. **Try solving first:**
   - Attempt each stage yourself
   - Use checkpoint only if truly stuck

3. **Create your own branches:**
   ```bash
   git checkout -b my-experiment checkpoint-2
   ```
   - Safe to experiment without losing checkpoint state

4. **Review checkpoint code:**
   - Don't just skip ahead
   - Understand what was done at each stage
   - Compare your approach to checkpoint solution

5. **Ask questions:**
   - If checkpoint code is unclear, ask instructor
   - Checkpoints are learning tools, not just shortcuts

## Troubleshooting

### "I can't find the checkpoints"

```bash
# Fetch all tags from remote
git fetch --all --tags

# List tags
git tag
```

### "Checkout says I have uncommitted changes"

```bash
# Save your work first
git stash

# Then checkout
git checkout checkpoint-2

# Restore your work later (on a new branch)
git checkout -b my-work
git stash pop
```

### "I'm in detached HEAD state"

This is normal when checking out a tag directly. To fix:

```bash
# Create a branch from here
git checkout -b my-analysis

# Or return to main
git checkout main
```

### "The checkpoint doesn't have the files I need"

Contact the instructor - checkpoint may not have been created correctly.

### "I want to start completely fresh"

```bash
# Discard all local changes
git reset --hard HEAD

# Clean untracked files
git clean -fd

# Return to main
git checkout main
```

## Advanced: Creating Dynamic Checkpoints During Workshop

If you want to save your progress as personal checkpoints:

```bash
# Create local tag
git tag my-checkpoint-2a

# Later, restore
git checkout my-checkpoint-2a

# These are local only (not pushed to remote)
```

## Questions?

If you have issues with checkpoints during the workshop:

1. Raise your hand and ask the instructor
2. Check this guide's troubleshooting section
3. Ask a neighbor or teaching assistant
4. Post question in workshop chat/forum

---

**Remember:** Checkpoints are learning tools. Use them to focus on understanding concepts, not just completing steps!
