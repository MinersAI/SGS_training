# Geospatial Machine Learning Training Notebook

## Overview

This comprehensive training notebook covers essential concepts and methods in geospatial machine learning for mineral exploration, from foundational data handling through advanced supervised prospectivity mapping.

## Learning Objectives

After completing this notebook, you will understand:

- **Data Foundations**: Handling raster and vector data, transformations, scaling, missing data, and spatial bias
- **Interpolation**: IDW and kriging methods for creating continuous surfaces
- **Unsupervised Methods**: PCA for dimensionality reduction and K-means for pattern detection
- **Anomaly Detection**: Multivariate outlier detection using Isolation Forest and ABOD
- **Spectral Analysis**: Mineral index calculation and spectral halo classification
- **Supervised ML**: End-to-end prospectivity mapping with CatBoost, including feature engineering, model interpretation, and output generation

## Contents

### Part 0: Setup (5-10 minutes)
- Environment configuration
- Synthetic data generation with real data fallback

### Part 1: Foundations (5-8 minutes)
1. Analytical Foundations - When to use ML, spatial scale considerations
2. Data Format Considerations - Raster vs vector, continuous vs categorical
3. Transformations and Scaling - Log transforms, standardization, outlier handling
4. Missing Data and Imputation - Strategies for categorical and continuous variables
5. Bias and Data Leakage - Spatial autocorrelation and proper train-test splitting
6. Interpolation - IDW vs kriging

### Part 2: Unsupervised Methods (3-5 minutes)
7. PCA - Dimensionality reduction and variance capture
8. K-Means - Cluster-based pattern identification

### Part 3: Anomaly Detection (3-5 minutes)
9. Multivariate Anomaly Detection Concepts - Global vs local outliers
10. Anomaly Detection Methods - Isolation Forest and ABOD

### Part 4: Spectral Methods (2-3 minutes)
11. Spectral Halo Classification - Mineral indices and alteration mapping

### Part 5: Supervised ML (8-12 minutes)
12. What Is Supervised Prospectivity Mapping
13. Primary Exploration ML Workflow
14. Problem Setup
15. Targets and Training Data
16. Feature Engineering
17. Interpreting and Iterating Results
18. Prospectivity Outputs

**Total estimated execution time: 15-25 minutes**

## Installation

### Quick Start

```bash
# Clone or download this repository
cd SGS_training

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook geospatial_ml_training.ipynb
```

### Using a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook geospatial_ml_training.ipynb
```

## Usage

### Running the Full Notebook

Simply run all cells sequentially from top to bottom. The notebook is designed to execute in under 25 minutes with synthetic data.

### Using Real Data

To use your own data instead of synthetic data:

1. Open the notebook
2. In the "Data Configuration" section, set `USE_REAL_DATA = True`
3. Update the file paths to point to your data files
4. Ensure your data matches the expected format (see comments in notebook)

### Running Individual Sections

Each major section can be run independently after executing the Setup cells (Part 0). This allows you to focus on specific topics without running the entire notebook.

## Data Requirements

### Synthetic Data (Default)
The notebook generates all required data automatically, including:
- 1000-2000 point samples with geochemistry
- 100×100 or 250×250 pixel rasters with spectral bands
- 20-30 synthetic mineral deposits

### Real Data (Optional)
If using your own data, you'll need:
- Point data: CSV with X, Y coordinates and geochemical columns
- Raster data: GeoTIFF files with spectral or geophysical bands
- Deposit locations: CSV with known deposit coordinates

## Key Features

- **Self-Contained**: Works out of the box with synthetic data
- **Educational Focus**: Heavy commenting and step-by-step explanations
- **Professional Visualizations**: Publication-quality figures (300 DPI)
- **Modular Design**: Each section builds on previous concepts
- **Fast Execution**: Optimized for quick iteration and learning
- **Industry-Relevant**: Methods actively used in mineral exploration

## Troubleshooting

### Import Errors
If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues
If the notebook runs out of memory:
- Reduce `n_points` in data generation (default 1000, try 500)
- Reduce `grid_size` for rasters (default 100×100, try 50×50)
- Restart the kernel between major sections

### Slow Execution
If execution is too slow:
- Ensure you're using synthetic data (not large real datasets)
- Reduce sample sizes in the configuration section
- Skip computationally expensive sections (SHAP, kriging)

## References

This notebook demonstrates simplified versions of production methods from:
- Anomaly detection module (`anomaly_detection/`)
- Supervised ML module (`supervised_ML/`)
- Spectral analysis tools (`MinersAI_datatools/`)

## License

This training material is provided for educational purposes.

## Support

For questions or issues with the notebook, please refer to the inline documentation and comments throughout the code.
