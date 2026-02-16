# Standard Chemometrics Datasets for Workshop

This document lists publicly available chemometrics datasets suitable for the vibe coding workshop, along with analysis prompts for hands-on exercises and checkpoints.

## Dataset Categories

1. **NIR Spectroscopy** (regression, calibration)
2. **Raman Spectroscopy** (classification, authentication)
3. **Hyperspectral Imaging** (spatial analysis)
4. **Process Monitoring** (time series, MSPC)
5. **Metabolomics** (multivariate analysis)

---

## 1. NIR Spectroscopy Datasets

### 1.1 Corn Dataset (Eigenvector Research)

**Description:** NIR spectra (1100-2498 nm) of 80 corn samples with moisture, oil, protein, and starch content measured by reference methods.

**Source:** Available in R package `pls` or from Eigenvector Research
- R: `data(NIRSoil, package="pls")`
- Python: Download from ChemometricsWithR

**Size:** 80 samples × 700 wavelengths

**Reference values:** Moisture (%), Oil (%), Protein (%), Starch (%)

**Typical use:** PLS regression calibration, preprocessing comparison

**Analysis prompts:**

#### Checkpoint 1: Data Exploration
```
"I have the corn NIR dataset loaded in corn_spectra.csv and corn_reference.csv.
Help me explore this data:
1. Load and visualize the spectra
2. Check for outliers using Mahalanobis distance
3. Show the distribution of reference values
4. Identify if any samples look unusual"
```

#### Checkpoint 2: Preprocessing
```
"Apply different preprocessing methods to the corn NIR spectra:
1. Compare raw vs SNV vs MSC preprocessing
2. Visualize the effect of each preprocessing
3. Which preprocessing method reduces scatter effects most effectively?
Use the chemometrics-preprocessing skill for guidance."
```

#### Checkpoint 3: Model Building
```
"Build a PLS regression model to predict moisture content from NIR spectra:
1. Split data 80/20 (stratified by moisture quartiles)
2. Apply SNV preprocessing
3. Determine optimal number of components via cross-validation
4. Train final model and evaluate on test set
5. Report RMSEP, R², and RPD
Follow the chemometrics-ml-selection and chemometrics-validation skills."
```

#### Checkpoint 4: Model Interpretation
```
"Interpret the PLS moisture prediction model:
1. Create PLS loadings plot (which wavelengths are important?)
2. Calculate VIP scores
3. Plot predicted vs actual with 95% confidence intervals
4. Identify the 3 most influential wavelengths and explain why they matter
   (hint: water absorption bands around 1450 and 1940 nm)"
```

### 1.2 Pharmaceutical Tablets (Simulated)

**Description:** NIR spectra of pharmaceutical tablets with API (active pharmaceutical ingredient) content variation.

**Source:** Can be simulated using Beer-Lambert law with realistic noise

**Size:** 100 samples × 500 wavelengths (1100-2500 nm)

**Reference values:** API content (% w/w), ranging 2-5%

**Typical use:** Quantitative calibration, regulatory validation

**Analysis prompts:**

#### Hands-On Exercise
```
"Analyze pharmaceutical tablet NIR spectra for API content prediction:
1. Explore the data and check for batch effects
2. Build multiple models: PLS, PCR, and Random Forest
3. Compare their performance using 10-fold cross-validation
4. Which model performs best? Why?
5. Validate that the best model meets ICH requirements (RPD > 2.5)
Report results following chemometrics-validation guidelines."
```

---

## 2. Raman Spectroscopy Datasets

### 2.1 Wine Authenticity Dataset

**Description:** Raman spectra of wine samples from different regions/varieties for authentication.

**Source:** UCI Machine Learning Repository or simulate based on literature

**Size:** 178 samples × ~1000 Raman shifts, 3 classes (wine varieties)

**Classes:** Cultivar 1, Cultivar 2, Cultivar 3

**Typical use:** Classification, discriminant analysis

**Analysis prompts:**

#### Checkpoint 1: Exploratory Analysis
```
"I have Raman spectra of wines from 3 different cultivars in wine_raman.csv.
Help me explore this dataset:
1. Load and visualize representative spectra from each class
2. Perform PCA and plot PC1 vs PC2 colored by class
3. Are the classes well-separated?
4. Identify potential outliers"
```

#### Checkpoint 2: Classification Model
```
"Build a classification model for wine authentication:
1. Split data 70/30 with stratified sampling
2. Preprocess: baseline correction + normalization
3. Compare PLS-DA vs SVM vs Random Forest
4. Use 10-fold stratified cross-validation
5. Report sensitivity, specificity, and confusion matrix
6. Which model is most accurate?"
```

#### Checkpoint 3: Feature Interpretation
```
"Interpret which Raman peaks distinguish the wine varieties:
1. Calculate feature importances (or PLS-DA loadings)
2. Identify the top 10 most discriminative Raman shifts
3. Visualize these important regions on mean spectra
4. Can you identify what chemical compounds these peaks correspond to?
   (e.g., phenolics, anthocyanins)"
```

### 2.2 Pharmaceutical Counterfeit Detection

**Description:** Raman spectra of genuine vs counterfeit pharmaceutical products.

**Source:** Can be simulated or obtained from collaborative research

**Size:** 200 samples × 800 Raman shifts, 2 classes (genuine, counterfeit)

**Classes:** Genuine (60%), Counterfeit (40%) - imbalanced

**Typical use:** Binary classification, imbalanced data handling

**Analysis prompts:**

#### Hands-On Exercise (Imbalanced Data)
```
"Detect counterfeit pharmaceuticals using Raman spectroscopy:
1. Load the imbalanced dataset (60% genuine, 40% counterfeit)
2. Use stratified sampling to preserve class ratios
3. Build a classification model focusing on high sensitivity
   (we cannot miss counterfeit products!)
4. Report precision, recall, F1-score, and ROC AUC
5. What is the optimal classification threshold for 99% sensitivity?
Handle the imbalanced data following chemometrics-validation practices."
```

---

## 3. Hyperspectral Imaging Datasets

### 3.1 Food Quality HSI Dataset

**Description:** Hyperspectral images of food products (e.g., meat, fruit) with quality labels.

**Source:** Public HSI datasets from agricultural research institutions

**Size:** 50 images, each ~500×500 pixels × 200 wavelengths

**Labels:** Quality grades (A, B, C) or defect presence

**Typical use:** Spatial quality mapping, image segmentation

**Analysis prompts:**

#### Checkpoint: Spatial Analysis
```
"Analyze hyperspectral images for food quality assessment:
1. Load a hyperspectral image cube
2. Visualize RGB composite and spectral profiles
3. Apply PCA for dimensionality reduction
4. Build a pixel-wise classifier (PLS-DA or SVM)
5. Create a quality map showing predicted grades spatially
6. Calculate percentage of pixels in each quality grade"
```

---

## 4. Process Monitoring Datasets

### 4.1 Batch Process Monitoring

**Description:** Multivariate time series from batch chemical process (temperature, pressure, pH, concentration over time).

**Source:** Simulated batch reactor data or Tennessee Eastman process

**Size:** 30 batches × 100 time points × 10 variables

**Labels:** Normal operation vs fault conditions

**Typical use:** MSPC (Multivariate Statistical Process Control), fault detection

**Analysis prompts:**

#### Process Monitoring Exercise
```
"Monitor a batch chemical process for fault detection:
1. Load batch process data with normal and fault batches
2. Build PCA model on normal operation batches
3. Calculate T² and SPE (Q) control limits
4. Apply to test batches: detect when faults occur
5. Create control charts showing T² and SPE over time
6. Which time points and variables contribute to faults?"
```

---

## 5. Metabolomics Datasets

### 5.1 NMR Metabolomics Dataset

**Description:** ¹H-NMR spectra of biological samples (e.g., urine, serum) with disease/control labels.

**Source:** MetaboLights or simulated based on literature

**Size:** 100 samples × 600 NMR bins, 2 classes (disease, control)

**Classes:** Disease (n=50), Control (n=50)

**Typical use:** Biomarker discovery, classification

**Analysis prompts:**

#### Biomarker Discovery
```
"Identify metabolic biomarkers distinguishing disease from control:
1. Load NMR metabolomics data
2. Preprocess: probabilistic quotient normalization + log transform
3. Build PLS-DA model with 10-fold cross-validation
4. Extract significant bins (VIP > 1)
5. Visualize loadings to identify discriminative regions
6. Validate model permutation testing (is classification real or chance?)
Report following metabolomics best practices."
```

---

## Recommended Dataset Downloads

### Public Repositories

1. **UCI Machine Learning Repository**
   - URL: https://archive.ics.uci.edu/ml/
   - Datasets: Wine, Multiple Features
   - Good for classification

2. **Kaggle Datasets**
   - Search: "spectroscopy", "NIR", "Raman"
   - Many chemistry-related datasets
   - Requires free account

3. **R Package `pls`**
   - Install: `install.packages("pls")`
   - Load: `data(yarn)`, `data(oliveoil)`, `data(gasoline)`
   - Convert to CSV for Python use

4. **ChemometricsWithR**
   - Companion datasets for the book
   - Available online
   - NIR, GC-MS, LC-MS datasets

5. **MetaboLights**
   - URL: https://www.ebi.ac.uk/metabolights/
   - Public metabolomics repository
   - NMR and MS data

### Creating Simulated Datasets

If public datasets are unavailable, create realistic simulations:

```python
import numpy as np
import pandas as pd

def simulate_nir_dataset(n_samples=80, n_wavelengths=500,
                         n_components=3, noise_level=0.01):
    """
    Simulate NIR spectroscopy dataset following Beer-Lambert law
    """
    # Wavelengths (nm)
    wavelengths = np.linspace(1100, 2500, n_wavelengths)

    # True concentrations (random)
    concentrations = np.random.uniform(0.5, 5.0, (n_samples, n_components))

    # Pure component spectra (Gaussian peaks)
    pure_spectra = np.zeros((n_components, n_wavelengths))
    peak_positions = [1450, 1940, 2100]  # Example: water absorption bands
    for i in range(n_components):
        peak = peak_positions[i]
        pure_spectra[i] = np.exp(-0.5 * ((wavelengths - peak) / 50) ** 2)

    # Mixture spectra (Beer-Lambert)
    spectra = concentrations @ pure_spectra

    # Add baseline (polynomial)
    baseline = np.outer(np.random.uniform(0, 0.5, n_samples),
                        wavelengths / 1000)
    spectra += baseline

    # Add noise
    spectra += np.random.normal(0, noise_level, spectra.shape)

    # Ensure non-negative
    spectra = np.maximum(spectra, 0)

    # Save
    df_spectra = pd.DataFrame(spectra, columns=wavelengths)
    df_reference = pd.DataFrame(concentrations,
                                columns=[f'Component_{i+1}'
                                        for i in range(n_components)])

    df_spectra.to_csv('simulated_nir_spectra.csv', index=False)
    df_reference.to_csv('simulated_nir_reference.csv', index=False)

    print(f"Simulated {n_samples} NIR spectra with {n_wavelengths} wavelengths")
    return df_spectra, df_reference

# Create dataset
simulate_nir_dataset()
```

---

## Workshop Datasets Directory Structure

```
data/
├── DATASETS_AND_PROMPTS.md  (this file)
├── corn_nir/
│   ├── README.md
│   ├── corn_spectra.csv
│   ├── corn_reference.csv
│   └── analysis_prompt.txt
├── wine_raman/
│   ├── README.md
│   ├── wine_raman_spectra.csv
│   ├── wine_classes.csv
│   └── analysis_prompt.txt
├── pharma_tablets/
│   ├── README.md
│   ├── tablet_nir_spectra.csv
│   ├── tablet_api_content.csv
│   └── analysis_prompt.txt
└── simulation_scripts/
    ├── simulate_nir.py
    ├── simulate_raman.py
    └── add_realistic_noise.py
```

---

## Dataset Selection Guide for Workshop

### For Introduction/Demo (Live Demo - Section 6)
- **Dataset:** Corn NIR (well-known, simple)
- **Task:** Moisture prediction with PLS
- **Duration:** 15-20 minutes
- **Shows:** Complete workflow from data loading to interpretation

### For Hands-On Exercise (Section 7)
- **Dataset:** Wine Raman or Pharmaceutical Counterfeit
- **Task:** Classification with multiple models
- **Duration:** 30-40 minutes
- **Shows:** Model comparison, validation strategies, class imbalance handling

### For Checkpoints (GitHub Releases)
- **Checkpoint 1:** Data loaded, exploratory plots created
- **Checkpoint 2:** Preprocessing applied, ready for modeling
- **Checkpoint 3:** Models trained, cross-validation complete
- **Checkpoint 4:** Final evaluation, interpretation, report generated

---

## Prompt Engineering Tips

### Good Prompts for Vibe Coding

✅ **Specific and contextualized:**
```
"I have NIR spectra of corn samples (1100-2498 nm, 700 wavelengths, 80 samples)
in corn_spectra.csv with reference moisture values in corn_reference.csv.
Build a PLS calibration model following chemometrics best practices from the
chemometrics-ml-selection and chemometrics-validation skills."
```

✅ **Break complex tasks into steps:**
```
"Step 1: Load the data and create an overview plot showing all spectra.
Step 2: Apply SNV preprocessing.
Step 3: Build PLS model with cross-validation to select components.
Step 4: Evaluate on test set and report RMSEP, R², RPD."
```

✅ **Reference skills explicitly:**
```
"Use the chemometrics-preprocessing skill to choose between SNV and MSC.
Then use chemometrics-validation to set up proper cross-validation."
```

❌ **Too vague:**
```
"Analyze the data"
```

❌ **No context:**
```
"Build a model"
```

---

## Additional Resources

### Books with Datasets

1. **Chemometrics with R** - Ron Wehrens
   - Companion datasets available online
   - NIR, GC-MS, LC-MS examples

2. **Introduction to Multivariate Statistical Analysis in Chemometrics** - Varmuza & Filzmoser
   - Example datasets included
   - Good for metabolomics

3. **Practical Guide to Chemometrics** - Paul Gemperline
   - Industrial case studies
   - Process monitoring examples

### Online Courses with Data

1. **Coursera: Chemometrics**
   - Some datasets available in course materials

2. **Eigenvector Research: PLS Toolbox Demos**
   - Demo datasets available with evaluation version

### Community

1. **r-chemometrics Google Group**
   - Ask for dataset recommendations

2. **Stack Overflow: [chemometrics] tag**
   - Dataset recommendations in questions

---

## Creating Your Own Workshop Dataset

If you have proprietary data suitable for the workshop:

1. **Anonymize:** Remove identifying information
2. **Subset:** Select representative samples (50-100)
3. **Document:** Create README with background, units, analytical methods
4. **Prepare prompts:** Write 4 checkpoint prompts as shown above
5. **Test:** Run through the analysis yourself first

**Minimal dataset requirements:**
- At least 50 samples (for meaningful validation)
- Clear reference values or class labels
- Documented analytical method
- Realistic noise/variability
- Represents a chemometrics application area

---

## Next Steps

1. Download or simulate 2-3 datasets for the workshop
2. Create checkpoint prompts for each dataset
3. Test the prompts with Claude Code CLI
4. Document expected outputs at each checkpoint
5. Create GitHub releases for checkpoint restoration

For questions about datasets, contact the workshop organizer or refer to the public repositories listed above.
