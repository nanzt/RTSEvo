# RTSEvo: Retrogressive Thaw Slump (RTS) Evolution Model

## Overview

This repository contains the implementation of RTSEvo, a dynamic evolution model for Retrogressive Thaw Slumps in permafrost regions. This framework moves beyond traditional static susceptibility assessments by simulating the spatiotemporal expansion of RTSs over time.

RTSEvo couples three modules: (1) a time-series forecast of the total regional RTS area, (2) a machine-learning module for pixel-level occurrence probability, and (3) a constrained spatial allocation module that simulates expansion using neighborhood effects, stochasticity, and a novel, process-based retrogressive erosion factor.

## Research Paper

The methodology, calibration, and validation of this model are detailed in our paper: "RTSEvo v1.0: A Retrogressive Thaw Slump Evolution Model" (Submitted to Geoscientific Model Development).

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

Logistic Regression version of the RTS Evolution Model. The main executable scripts for running the full RTSEvo simulation pipeline. They handle everything from data preprocessing and model training to the final spatial allocation and output generation.

**Key Features:**

-   Processes multi-temporal RTS driving datasets
-   Performs feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
-   Optimizes hyperparameters using Latin Hypercube Sampling
-   Generates RTS occurrence probability maps
-   Simulates RTS evolution using spatial allocation module

**Usage:**

```python
python LR-EM.py

```

**Inputs Required:**

-   Historical RTS raster data
-   Environmental driving factors (DEM, slope, aspect, climate, geology, vegetation, permafrost characteristics)
-   Initial RTS distribution from a previous year to use as the starting point. For example, the study used the observed baseline RTS distribution as the starting point to simulate the feature RTS maps.

**Outputs:**

-   Optimized Logistic Regression model
-   RTS occurrence probability maps
-   Simulated RTS distribution rasters

#### `RF-EM.py`

Random Forest version of the RTS Evolution Model, similar structure to LR-EM but uses Random Forest classifier.

**Key Differences:**

-   Uses ensemble learning approach
-   Different hyperparameter optimization ranges
-   Generally higher AUC but may be more prone to overfitting

#### `Parameter Calibration.py`

A utility script to systematically find the optimal parameters for the spatial allocation module based on a user-defined reference year.

**Purpose:**

-   Optimizes neighborhood size ($\omega$)
-   Calibrates neighborhood weight
-   Tunes stochastic parameters ($\alpha$, $\beta$)

**Method:**

-   Uses a user-defined reference RTS map for calibration
-   Employs Latin Hypercube Sampling for efficient parameter space exploration
-   Evaluates performance using Figure of Merit (FoM)

**Usage:**

```python
python "Parameter Calibration.py"

```


#### `RTS areal demand forecasting.py`

A utility script to forecast the total area of new RTS growth needed for the simulation based on Holt's linear trend method.

**Features:**

-   Trains on historical data
-   Validates on independent historical data
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

This generates the total area of RTS for your simulation years based on the historical trend.

### 3. Parameter Calibration

```python
# Calibrate using reference RTS map 
python "Parameter Calibration.py"

```

This script uses a reference year from the historical data to tune the parameters.

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

1.  Load and preprocess multi-year data
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

We setup three experiments to test model performance. The related data can be accessed via figshare (https://doi.org/10.6084/m9.figshare.30317599). The study area is the Beiluhe basin, Qinghai-Tibet Plateau. 

Based on the independent validation of RTS maps 2021 and 2022 on the study area,

**LR-EM:**

-   2021 FoM: 12.00%, Kappa: 94.79%
-   2022 FoM: 8.88%, Kappa: 91.51%

**RF-EM:**

-   2021 FoM: 10.77%, Kappa: 94.87%
-   2022 FoM: 8.78%, Kappa: 91.22%

We also found including the retrogressive erosion factor in the model improves FoM by up to 29.3%.

## GPU Acceleration

The model supports CUDA GPU acceleration for faster computation:

For systems without GPU, the model falls back to optimized CPU vectorized computation.

## Output Files

-   **Probability maps:** GeoTIFF format probability rasters (0-1 scale)
-   **Simulated RTS distributions:** GeoTIFF with values 1 (non-RTS) and 2 (RTS)
-   **Parameter tuning results:** CSV files with performance metrics
-   **Model evaluation metrics:** Console output with FoM, Kappa, Moran's I

## Citation

If you use this model or code in your research, please cite our paper and this repository:

```
[Paper citation details - To be added upon publication]

Jiwei Xu and Zhuotong Nan, RTSEvo (v1.0): A retrogressive thaw slump evolution model, https://github.com/nanzt/RTSEvo

```

## License

MIT License - See individual script headers for full license text.

## Contact

For questions or issues, please open an issue on this GitHub repository or contact the corresponding author at giscn@msn.com from permalab (https://permalab.science)
