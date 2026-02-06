# RTSEvo v1.0: Retrogressive Thaw Slump Evolution Model

## Overview
RTSEvo is a dynamic evolution model for simulating the spatiotemporal expansion of Retrogressive Thaw Slumps (RTS) in permafrost regions. This framework integrates machine learning with cellular automata to move beyond traditional static susceptibility assessments, enabling regional-scale dynamic simulation and short-term forecasting.
RTSEvo couples three modules: (1) a time-series forecast of the total regional RTS area, (2) a machine-learning module for pixel-level occurrence probability, and (3) a constrained spatial allocation module that simulates expansion using neighborhood effects, stochasticity, and a novel, process-based retrogressive erosion factor.

## Research Paper

The methodology, calibration, and validation of this model are detailed in our paper: "RTSEvo v1.0: A Retrogressive Thaw Slump Evolution Model" (Submitted to Geoscientific Model Development).

## Model Architecture

The framework consists of three core modules:

### 1. RTS Areal Demand Forecasting Module
Projects total RTS area using Holt's linear trend method based on historical time series data.

### 2. Base Occurrence Probability Mapping Module
Calculates pixel-level RTS initiation probability using:
- **Logistic Regression** (`LR-EM.py`)
- **Random Forest** (`RF-EM.py`)

### 3. Constrained Spatial Allocation Module
Simulates RTS expansion through cellular automata with:
- **Neighborhood effect**: Density-dependent growth
- **Retrogressive erosion factor**: Process-based directional preference
- **Stochastic factor**: Random variability
- **Adaptive inertia**: Dynamic constraint to match areal demand

**Mathematical formulation**:
```
P_Total= P_base × U × Erosion × RA × Inertia
```

Where:
- P_base: Base occurrence probability from ML
- U: Neighborhood effect
- Erosion: Retrogressive erosion factor
- RA: Stochastic factor
- Inertia: Adaptive inertia coefficient


## File Descriptions

### Core Model Scripts

#### `LR-EM.py`

Logistic Regression version of the RTS Evolution Model. The main executable scripts for running the full RTSEvo simulation pipeline. They handle everything from data preprocessing and model training to the final spatial allocation and output generation.

**Key Features:**

-   Processes multi-temporal RTS driving datasets
-   Performs feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
-   Optimizes hyperparameters using Latin Hypercube Sampling
-   Generates RTS base occurrence probability maps
-   Simulates RTS evolution using spatial allocation module

**Usage:**

```python
python LR-EM.py
```
```
landuse_2021_predict = ca_simulation_optimized(
            landuse_2020.copy(), prob_2, aspect,
            target_areas=103587.75, max_iterations=100,
            seed=seed, neighborhood_weight=neighborhood_weight, use_gpu=use_gpu
        )
```
- target_areas: The total RTS area for the target simulation year.
- max_iterations: Maximum number of iterations to reach the total area

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

-   Optimizes neighborhood size 
-   Calibrates neighborhood weight
-   Tunes stochastic parameters 

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
#### The datasets are available via https://doi.org/10.6084/m9.figshare.30325243

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

## Model Performance

We setup three experiments to test model performance. The study area is the Beiluhe basin, Qinghai-Tibet Plateau. 

Based on the independent validation of RTS maps 2021 and 2022 on the study area,


| Year | Model   | FoM    | Kappa  | F1 Score | Moran's I |
|------|---------|--------|--------|----------|-----------|
| 2021 | LR-EM   | 12.00% | 94.79% | 95.74%   | 0.616     |
| 2021 | RF-EM   | 10.77% | 94.87% | 95.08%   | 0.613     |
| 2022 | LR-EM   | 8.88%  | 91.51% | 91.89%   | 0.631     |
| 2022 | RF-EM   | 8.78%  | 91.22% | 91.61%   | 0.631     |

We also found including the retrogressive erosion factor in the model improves FoM by up to 29.3%.


## Citation

If you use this model or code in your research, please cite our paper and this repository:

```
[Paper citation details - Xu, J., Zhao, S., Nan, Z., Niu, F., and Zhang, Y.: RTSEvo v1.0: A Retrogressive Thaw Slump Evolution Model, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-5005, 2025.]

Jiwei Xu and Zhuotong Nan, RTSEvo (v1.0): A retrogressive thaw slump evolution model, https://github.com/nanzt/RTSEvo
```

## License

MIT License

## Contact

For questions or issues, please open an issue on this GitHub repository or contact the corresponding author at giscn@msn.com from permalab (https://permalab.science)
