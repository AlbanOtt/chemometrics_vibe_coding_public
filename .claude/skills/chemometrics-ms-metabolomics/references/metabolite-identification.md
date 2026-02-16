# Metabolite Identification

## MSI Annotation Levels

The Metabolomics Standards Initiative (MSI) defines four annotation levels:

| Level | Name | Requirements |
|-------|------|--------------|
| **1** | Identified compound | Match with authentic standard: RT, MS, MS/MS |
| **2** | Putatively annotated | Spectral library match (MS/MS only) |
| **3** | Putatively characterized class | Characteristic fragments, compound class |
| **4** | Unknown | No match, structural information |

## Database Resources

| Database | Focus | URL |
|----------|-------|-----|
| **HMDB** | Human metabolites | hmdb.ca |
| **METLIN** | MS/MS spectra | metlin.scripps.edu |
| **MassBank** | Mass spectra | massbank.eu |
| **LipidMaps** | Lipids | lipidmaps.org |
| **KEGG** | Pathways | kegg.jp |
| **MoNA** | Community spectra | mona.fiehnlab.ucdavis.edu |

## MS/MS Spectral Matching

```python
from scipy.spatial.distance import cosine
import numpy as np

def match_msms_spectrum(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    library_mz: np.ndarray,
    library_intensity: np.ndarray,
    mz_tolerance: float = 0.02,
    min_matched_peaks: int = 3
) -> dict:
    """Match query MS/MS spectrum against library spectrum."""
    # Normalize intensities
    query_norm = query_intensity / query_intensity.max()
    library_norm = library_intensity / library_intensity.max()

    # Match peaks within tolerance
    matched_query = []
    matched_library = []

    for i, q_mz in enumerate(query_mz):
        matches = np.where(np.abs(library_mz - q_mz) < mz_tolerance)[0]
        if len(matches) > 0:
            # Take closest match
            best_match = matches[np.argmin(np.abs(library_mz[matches] - q_mz))]
            matched_query.append(query_norm[i])
            matched_library.append(library_norm[best_match])

    n_matched = len(matched_query)

    if n_matched < min_matched_peaks:
        return {'score': 0.0, 'n_matched': n_matched, 'valid': False}

    # Cosine similarity
    matched_query = np.array(matched_query)
    matched_library = np.array(matched_library)

    score = 1 - cosine(matched_query, matched_library)

    return {
        'score': score,
        'n_matched': n_matched,
        'n_query': len(query_mz),
        'n_library': len(library_mz),
        'valid': True
    }

def assign_annotation_level(
    has_authentic_standard: bool,
    rt_matched: bool,
    ms_matched: bool,
    msms_matched: bool,
    msms_score: float = 0.0
) -> int:
    """Assign MSI annotation level based on available evidence."""
    if has_authentic_standard and rt_matched and ms_matched and msms_matched:
        return 1  # Identified compound
    elif msms_matched and msms_score > 0.7:
        return 2  # Putatively annotated
    elif ms_matched:
        return 3  # Putatively characterized class
    else:
        return 4  # Unknown
```
