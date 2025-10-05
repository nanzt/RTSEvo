# RTSEvo: Retrogressive Thaw Slump (RTS) Evolution Model

## Overview

This repository contains the implementation of a dynamic evolution model for Retrogressive Thaw Slumps (RTS)in permafrost regions, , known as RTSEvo. Existing RTS modeling studies are largely confined to static susceptibility mapping, lacking the capacity to predict their spatiotemporal evolution. To bridge this gap, we developed a new dynamic RTS evolution model that couples three modules: (1) a time-series forecast of regional RTS area, (2) a machine-learning module for pixel-level probability mapping, and (3) a constrained spatial allocation module that simulates RTS expansion by integrating neighborhood effects, stochasticity, and a novel retrogressive erosion factor, representing a significant advancement from traditional static susceptibility assessments.

## Research Paper

The methodology is detailed in the paper "A Retrogressive Thaw Slump Evolution Model", which we submitted to Geoscientific Model Development for consideration of publication.

## System Requirements

### Hardware

-   CUDA-capable GPU (optional but recommended for faster computation)


### Software Dependencies

```
numpy
pandas
matplotlib
scikit-learn
rasterio
gdal (osgeo)
statsmodels
pyDOE
numba
joblib
tqdm

```

## Model Architecture

The framework consists of three core modules:

1.  **RTS Areal Demand Forecasting** - Projects total RTS area using Holt's linear trend method
2.  **Base Occurrence Probability Mapping** - Uses machine learning (Logistic Regression or Random Forest) to calculate pixel-level RTS initiation probability
3.  **Constrained Spatial Allocation** - Simulates RTS expansion through cellular automata with neighborhood effects, retrogressive erosion factors, and stochastic components

## File Descriptions

### Core Model Scripts

#### `LR-EM.py`

Logistic Regression-based Evolution Model for RTS simulation.

**Key Features:**

-   Processes multi-temporal RTS driving datasets (2016-2020)
-   Performs feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
-   Optimizes hyperparameters using Latin Hypercube Sampling
-   Generates RTS occurrence probability maps
-   Simulates RTS evolution using spatial allocation module

**Usage:**

```python
python LR-EM.py

```

**Inputs Required:**

-   Historical RTS raster data (2016-2020)
-   Environmental driving factors (DEM, slope, aspect, climate, geology, vegetation, permafrost characteristics)
-   Initial RTS distribution for simulation year

**Outputs:**

-   Optimized Logistic Regression model
-   RTS occurrence probability maps
-   Simulated RTS distribution rasters

#### `RF-EM.py`

Random Forest-based Evolution Model, similar structure to LR-EM but uses Random Forest classifier.

**Key Differences:**

-   Uses ensemble learning approach
-   Different hyperparameter optimization ranges
-   Generally higher AUC but may be more prone to overfitting

#### `Parameter Calibration.py`

Calibrates spatial allocation module parameters for optimal model performance.

**Purpose:**

-   Optimizes neighborhood size ($\omega$)
-   Calibrates neighborhood weight
-   Tunes stochastic parameters ($\alpha$, $\beta$)

**Method:**

-   Uses 2020 RTS map as calibration reference
-   Employs Latin Hypercube Sampling for efficient parameter space exploration
-   Evaluates performance using Figure of Merit (FoM)

**Usage:**

```python
python "Parameter Calibration.py"

```


#### `RTS areal demand forecasting.py`

Implements Holt's linear trend method for forecasting total RTS area.

**Features:**

-   Trains on 2017-2020 data
-   Validates on 2021-2022 data
-   Provides model evaluation metrics (MAPE, R²)

**Usage:**

```python
python "RTS areal demand forecasting.py"

```

## Workflow

### 1. Data Preparation

Ensure all driving factor rasters are at 10m resolution with consistent projection and extent:

**Required Data Layers:**

-   **Topographic:** DEM, slope, aspect, profile curvature
-   **Climate:** Precipitation (cumulative and maximum), temperature (maximum summer)
-   **Geological:** Distance to faults, lithology, soil texture (multiple depths)
-   **Hydrological:** TWI, distance to water bodies, NDVI
-   **Permafrost:** FDD, TDD, ground ice content, active layer thickness
-   **Anthropogenic:** Distance to railway, Land use/land cover

### 2. Area Demand Forecasting

```python
# Run the forecasting model
python "RTS areal demand forecasting.py"

```

This generates area targets for simulation years (2021, 2022).

### 3. Parameter Calibration

```python
# Calibrate using 2020 as reference
python "Parameter Calibration.py"

```

**Optimal parameters from the study:**

-   **LR-EM:** N=3, $\omega$=0.813, $\alpha$=0.04, $\beta$=0.161
-   **RF-EM:** N=3, $\omega$=0.759, $\alpha$=0.351, $\beta$=0.498

### 4. Model Training and Simulation

#### For Logistic Regression model:

```python
python LR-EM.py

```

#### For Random Forest model:

```python
python RF-EM.py

```

Both scripts will:

1.  Load and preprocess multi-year data (2016-2020)
2.  Perform feature selection
3.  Optimize machine learning model hyperparameters
4.  Generate occurrence probability maps
5.  Run evolution model simulation
6.  Output simulated RTS distributions

## Key Model Components

### Cellular Automata Rules

The spatial allocation uses the following transition probability:

```
P_Total = P_base × U × Erosion × RA × Inertia

```

Where:

-   **P_base:** Base occurrence probability from ML model
-   **U:** Neighborhood effect (density of RTS in surrounding window)
-   **Erosion:** Retrogressive erosion factor (directional growth preference)
-   **RA:** Stochastic factor (random variability)
-   **Inertia:** Adaptive coefficient to match area demand

### Retrogressive Erosion Factor

The model incorporates process-based rules that simulate upslope (headward) retreat:

```python
# Calculates directional weights based on slope aspect
# Favors growth opposite to slope direction
# Critical for capturing RTS morphology

```

## Model Performance

Based on Beiluhe Basin validation (2021-2022):

**LR-EM:**

-   2021 FoM: 12.00%, Kappa: 94.79%
-   2022 FoM: 8.88%, Kappa: 91.51%

**RF-EM:**

-   2021 FoM: 10.77%, Kappa: 94.87%
-   2022 FoM: 8.78%, Kappa: 91.22%

**Note:** Including the retrogressive erosion factor improves FoM by >22%

## GPU Acceleration

The model supports CUDA GPU acceleration for faster computation:

For systems without GPU, the model falls back to optimized CPU vectorized computation.

## Output Files

-   **Probability maps:** GeoTIFF format probability rasters (0-1 scale)
-   **Simulated RTS distributions:** GeoTIFF with values 1 (non-RTS) and 2 (RTS)
-   **Parameter tuning results:** CSV files with performance metrics
-   **Model evaluation metrics:** Console output with FoM, Kappa, Moran's I

## Citation

If you use this code, please cite:

```
Jiwei Xu and Zhuotong Nan, RTSEvo (v1.0): A retrogressive thaw slump evolution model, https://github.com/nanzt/RTSEvo

```

## License

MIT License - See individual script headers for full license text.

## Contact

For questions or issues, please refer to the paper or contact the corresponding author at giscn@msn.com from permalab (https://permalab.science)
