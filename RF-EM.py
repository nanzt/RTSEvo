"""
@Description: RTSEvo v1.0 - Random Forest-based Evolution Model (LR-EM)

    'RF-EM.py' integrates three core components:
    1. Feature Engineering and Machine Learning Module:
       - Recursive Feature Elimination with Cross-Validation (RFECV) for optimal feature selection
       - Random Forest classifier with Latin Hypercube Sampling (LHS) for hyperparameter optimization
       - Multi-period training approach (2016-2020) capturing temporal dynamics of RTS expansion

    2. Base Occurrence Probability Mapping

    3. Constrained Spatial Allocation Module:
       - Cellular Automata (CA) framework for iterative RTS expansion simulation
       - Four dynamic factors integrated in total transition probability:
         a) Base occurrence probability (from Random Forest)
         b) Neighborhood effect (spatial contagion within N×N window)
         c) Retrogressive erosion factor (directional headwall retreat based on slope aspect)
         d) Stochastic factor
       - Adaptive inertia coefficient ensuring target area demand is met

@Date: 2026-2-5

@Website: https://permalab.science
"""
from osgeo import gdal
import rasterio
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pyDOE import lhs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os
import numba
from numba import cuda
import math



# Function to read raster data
def read_raster(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open file：{file_path}")
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()


#Function to write Raster Data
def write_raster1(output_file, data, geo_transform, projection, x_size, y_size):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file, x_size, y_size, 1, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(projection)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.FlushCache()
    out_dataset = None

"""
First, construct the RTS-driven dataset for 2016-2017
"""
# Read RTS-driven data for 2016-2017

# Reading topographic factors: DEM, slope, aspect (flat, north, northeast, east, northwest, west, southeast, southwest, south), and profile curvature.
DEM = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\DEM.tif')
slope = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\slope.tif')
#aspect
flat = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\flat slope.tif')
East = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\East slope.tif')
Northeast = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Northeast slope.tif')
Southeast = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Southeast slope.tif')
North = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\North slope.tif')
West = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\west slope.tif')
Northwest = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Northwest slope.tif')
Southwest = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Southwest slope.tif')
South = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\South slope.tif')

Profile_Curvature = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Profile Curvature.tif')

# Hydrological Vegetation：TWI, Distance to Rivers and Lakes, NDVI
TWI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\TWI.tif')
Distance_Lake = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Lake.tif')
NDVI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\NDVI201609_201708mean.tif')

# Climate factors: Cumulative precipitation , Maximum precipitation, Highest temperature in summer
precipitation_sum = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201609_201708Cumulative Precipitation.tif')
Max_Summer_Precipitation = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201609_201708Maximum Summer Precipitation.tif')
Max_Summer_Temperature = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201609_201708Maximum Summer Temperature.tif')

# Topography
Distance_Faults = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Fault.tif')
# soil_texture
sand0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand0_5.tif')
sand5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand5_15.tif')
sand15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand15_30.tif')
sand30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand30_60.tif')
sand60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand60_100.tif')
sand100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand100_200.tif')
clay0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay0_5.tif')
clay5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay5_15.tif')
clay15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay15_30.tif')
clay30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay30_60.tif')
clay60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay60_100.tif')
clay100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay100_200.tif')
silt0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt0_5.tif')
silt5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt5_15.tif')
silt15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt15_30.tif')
silt30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt30_60.tif')
silt60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt60_100.tif')
silt100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt100_200.tif')
# Lithology: Hard rock、Weak rock 、Semi-Hard rock、 Loose rock
hard_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\hard rock.tif')
weak_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\weak rock.tif')
semi_hard_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\semi-hard rock.tif')
loose_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\loose rock.tif')

# anthropogenic factors: Distance to QTR  LULC(grassland、meadow、water body、wetland、bare land)
Distance_QTR = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Qinghai-Tibet Railway.tif')
# LULC
Bareland = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\bareland.tif')
grassland = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\grassland.tif')
meadow = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\meadow.tif')
water_body = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Water_body.tif')
wetland = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Wetland.tif')

# permafrost characteristics（FDD、TDD、Ground Ice Content、Active layer thickness）
FDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201609_201708FDD.tif')
TDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201609_201708TDD.tif')
Ground_Ice = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Ground ice content.tif')
ALT = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Active layer thickness.tif')

# The Time feature is used to index the driving dataset of RTS expansion between every two years
Time = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2020 Time raster\2016_2017.tif')

#The binary_variable is the RTS expansion raster data for two consecutive years
binary_variable = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2022 RTS Expansion Raster\2016-2017expansion.tif')

rasters = {
    "DEM": DEM, "slope": slope, "flat": flat, "East": East, "Northeast": Northeast,
    "Southeast": Southeast, "North": North, "West": West, "Northwest": Northwest, "Southwest": Southwest, "South": South,
    "Profile_Curvature": Profile_Curvature, "TWI": TWI, "Distance_Lake": Distance_Lake,
    "NDVI": NDVI, "precipitation_sum": precipitation_sum, "Max_Summer_Precipitation": Max_Summer_Precipitation,
    "Max_Summer_Temperature": Max_Summer_Temperature, "Distance_Faults": Distance_Faults,
    "sand0_5": sand0_5, "sand5_15": sand5_15, "sand15_30": sand15_30, "sand30_60": sand30_60, "sand60_100": sand60_100,
    "sand100_200": sand100_200,
    "clay0_5": clay0_5, "clay5_15": clay5_15, "clay15_30": clay15_30, "clay30_60": clay30_60, "clay60_100": clay60_100,
    "clay100_200": clay100_200,
    "silt0_5": silt0_5, "silt5_15": silt5_15, "silt15_30": silt15_30, "silt30_60": silt30_60, "silt60_100": silt60_100,
    "silt100_200": silt100_200,
    "hard_rock": hard_rock, "weak_rock": weak_rock, "semi_hard_rock": semi_hard_rock,
    "loose_rock": loose_rock, "Distance_QTR": Distance_QTR, "Bareland": Bareland, "grassland": grassland, "meadow": meadow,
    "water_body": water_body, "wetland": wetland,
    "FDD": FDD, "TDD": TDD, "Ground_Ice": Ground_Ice, "ALT": ALT,
    "Time": Time,
    "binary_variable": binary_variable
}

# Select the DEM as the reference raster to determine the size of the global mask;
reference_raster = rasters["DEM"]

# Iterate through all raster data to create a global mask;
global_mask = np.zeros_like(reference_raster, dtype=bool)

for key, raster in rasters.items():
    mask = (raster == -9999)
    global_mask |= mask

# Traverse all raster data and replace the values at corresponding positions in the global mask with -9999.
for key, raster in rasters.items():
    raster = raster.astype(np.float32)
    raster[global_mask] = -9999
    rasters[key] = raster

# Update variables
DEM = rasters["DEM"]
slope = rasters["slope"]
flat = rasters["flat"]
East = rasters["East"]
Northeast = rasters["Northeast"]
Southeast = rasters["Southeast"]
North = rasters["North"]
West = rasters["West"]
Northwest = rasters["Northwest"]
Southwest = rasters["Southwest"]
South = rasters["South"]

Profile_Curvature = rasters["Profile_Curvature"]
TWI = rasters["TWI"]
Distance_Lake = rasters["Distance_Lake"]
NDVI = rasters["NDVI"]
precipitation_sum = rasters["precipitation_sum"]
Max_Summer_Precipitation = rasters["Max_Summer_Precipitation"]
Max_Summer_Temperature = rasters["Max_Summer_Temperature"]
Distance_Faults = rasters["Distance_Faults"]

sand0_5 = rasters["sand0_5"]
sand5_15 = rasters["sand5_15"]
sand15_30 = rasters["sand15_30"]
sand30_60 = rasters["sand30_60"]
sand60_100 = rasters["sand60_100"]
sand100_200 = rasters["sand100_200"]

clay0_5 = rasters["clay0_5"]
clay5_15 = rasters["clay5_15"]
clay15_30 = rasters["clay15_30"]
clay30_60 = rasters["clay30_60"]
clay60_100 = rasters["clay60_100"]
clay100_200 = rasters["clay100_200"]

silt0_5 = rasters["silt0_5"]
silt5_15 = rasters["silt5_15"]
silt15_30 = rasters["silt15_30"]
silt30_60 = rasters["silt30_60"]
silt60_100 = rasters["silt60_100"]
silt100_200 = rasters["silt100_200"]

hard_rock = rasters["hard_rock"]
weak_rock = rasters["weak_rock"]
semi_hard_rock = rasters["semi_hard_rock"]
loose_rock = rasters["loose_rock"]
Distance_QTR = rasters["Distance_QTR"]
Bareland = rasters["Bareland"]
grassland = rasters["grassland"]
meadow = rasters["meadow"]
water_body = rasters["water_body"]
wetland = rasters["wetland"]
FDD = rasters["FDD"]
TDD = rasters["TDD"]
Ground_Ice = rasters["Ground_Ice"]
ALT = rasters["ALT"]
Time = rasters["Time"]

n_samples = DEM.shape[0] * DEM.shape[1]

# Convert raster data into a feature matrix
X201617 = np.column_stack([
    DEM.flatten(), slope.flatten(), flat.flatten(), East.flatten(), Northeast.flatten(),
    Southeast.flatten(), North.flatten(), West.flatten(), Northwest.flatten(), Southwest.flatten(), South.flatten(),
    Profile_Curvature.flatten(),
    TWI.flatten(), Distance_Lake.flatten(), NDVI.flatten(),
    precipitation_sum.flatten(), Max_Summer_Precipitation.flatten(), Max_Summer_Temperature.flatten(),
    Distance_Faults.flatten(),
    sand0_5.flatten(),
    sand5_15.flatten(),
    sand15_30.flatten(),
    sand30_60.flatten(),
    sand60_100.flatten(),
    sand100_200.flatten(),
    clay0_5.flatten(),
    clay5_15.flatten(),
    clay15_30.flatten(),
    clay30_60.flatten(),
    clay60_100.flatten(),
    clay100_200.flatten(),
    silt0_5.flatten(),
    silt5_15.flatten(),
    silt15_30.flatten(),
    silt30_60.flatten(),
    silt60_100.flatten(),
    silt100_200.flatten(),
    hard_rock.flatten(),
    weak_rock.flatten(), semi_hard_rock.flatten(), loose_rock.flatten(),
    Distance_QTR.flatten(), Bareland.flatten(), grassland.flatten(), meadow.flatten(), water_body.flatten(),
    wetland.flatten(),
    FDD.flatten(), TDD.flatten(), Ground_Ice.flatten(), ALT.flatten(), Time.flatten()
])

# Convert the dependent variable raster data to the target vector
y201617 = binary_variable.flatten()

# Check if -9999 exists in y
y_mask201617 = (y201617 == -9999) | (y201617 == -1)

# Use Boolean indexing to remove rows containing -9999 in y
X201617 = X201617[~y_mask201617]
y201617 = y201617[~y_mask201617]

# Check if -9999 exists in X，If a row contains -9999, the mask for that row is True
mask = (X201617 == -9999).any(axis=1)

# Use Boolean indexing to remove rows containing -9999 in X
X_clean201617 = X201617[~mask]
y_clean201617 = y201617[~mask]

# print("“After removing rows containing -9999:”")
# print("X201617 shape:", X_clean201617.shape)
# print("y201617 shape:", y_clean201617.shape)
categorical_columns = [
    'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South',
   'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock','Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland', 'Time'
]

all_columns = [
     'DEM', 'Slope', 'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South', 'Profile_Curvature',
    'TWI', 'Distance_Lake', 'NDVI',
    'Precipitation_sum', 'Max_Summer_Precipitation', 'Max_Summer_Temperature',
    'Distance_Faults', 'sand0_5','sand5_15','sand15_30','sand30_60','sand60_100','sand100_200',
    'clay0_5','clay5_15','clay15_30','clay30_60','clay60_100','clay100_200',
    'silt0_5','silt5_15','silt15_30','silt30_60', 'silt60_100','silt100_200', 'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock',
    'Distance_QTR', 'Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland',
    'FDD', 'TDD', 'Ground_Ice', 'ALT',
    'Time'
]

non_categorical_columns = [col for col in all_columns if col not in categorical_columns]
non_categorical_indices = [all_columns.index(col) for col in non_categorical_columns]
for idx in non_categorical_indices:
    col = X_clean201617[:, idx]
    col_min = np.min(col)
    col_max = np.max(col)
    X_clean201617[:, idx] = (col - col_min) / (col_max - col_min)



"""
"""
# Read RTS-driven data for 2017-2018
# Reading topographic factors: DEM, slope, aspect (flat, north, northeast, east, northwest, west, southeast, southwest, south), and profile curvature.
DEM = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\DEM.tif')
slope = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\slope.tif')
#aspect
# flat = read_raster(r'RTS-evolution model driven data\categorical variables\flat slope.tif')
# East = read_raster(r'RTS-evolution model driven data\categorical variables\East slope.tif')
# Northeast = read_raster(r'RTS-evolution model driven data\categorical variables\Northeast slope.tif')
# Southeast = read_raster(r'RTS-evolution model driven data\categorical variables\Southeast slope.tif')
# North = read_raster(r'RTS-evolution model driven data\categorical variables\North slope.tif')
# West = read_raster(r'RTS-evolution model driven data\categorical variables\west slope.tif')
# Northwest = read_raster(r'RTS-evolution model driven data\categorical variables\Northwest slope.tif')
# Southwest = read_raster(r'RTS-evolution model driven data\categorical variables\Southwest slope.tif')
# South = read_raster(r'RTS-evolution model driven data\categorical variables\South slope.tif')

Profile_Curvature = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Profile Curvature.tif')

# Hydrological Vegetation：TWI, Distance to Rivers and Lakes, NDVI
TWI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\TWI.tif')
Distance_Lake = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Lake.tif')
NDVI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\NDVI201709_201808mean.tif')

# Climate factors: Cumulative precipitation , Maximum precipitation, Highest temperature in summer
precipitation_sum = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201709_201808Cumulative Precipitation.tif')
Max_Summer_Precipitation = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201709_201808Maximum Summer Precipitation.tif')
Max_Summer_Temperature = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201709_201808Maximum Summer Temperature.tif')

# Topography
Distance_Faults = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Fault.tif')
# soil_texture
sand0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand0_5.tif')
sand5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand5_15.tif')
sand15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand15_30.tif')
sand30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand30_60.tif')
sand60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand60_100.tif')
sand100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand100_200.tif')
clay0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay0_5.tif')
clay5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay5_15.tif')
clay15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay15_30.tif')
clay30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay30_60.tif')
clay60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay60_100.tif')
clay100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay100_200.tif')
silt0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt0_5.tif')
silt5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt5_15.tif')
silt15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt15_30.tif')
silt30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt30_60.tif')
silt60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt60_100.tif')
silt100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt100_200.tif')
# Lithology: Hard rock、Weak rock 、Semi-Hard rock、 Loose rock
# hard_rock = read_raster(r'RTS-evolution model driven data\categorical variables\hard rock.tif')
# weak_rock = read_raster(r'RTS-evolution model driven data\categorical variables\weak rock.tif')
# semi_hard_rock = read_raster(r'RTS-evolution model driven data\categorical variables\semi-hard rock.tif')
# loose_rock = read_raster(r'RTS-evolution model driven data\categorical variables\loose rock.tif')

# anthropogenic factors: Distance to QTR  LULC(grassland、meadow、water body、wetland、bare land)
Distance_QTR = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Qinghai-Tibet Railway.tif')
# LULC
# Bareland = read_raster(r'RTS-evolution model driven data\categorical variables\bareland.tif')
# grassland = read_raster(r'RTS-evolution model driven data\categorical variables\grassland.tif')
# meadow = read_raster(r'RTS-evolution model driven data\categorical variables\meadow.tif')
# water_body = read_raster(r'RTS-evolution model driven data\categorical variables\Water_body.tif')
# wetland = read_raster(r'RTS-evolution model driven data\categorical variables\Wetland.tif')

# permafrost characteristics（FDD、TDD、Ground Ice Content、Active layer thickness）
FDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201709_201808FDD.tif')
TDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201709_201808TDD.tif')
Ground_Ice = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Ground ice content.tif')
ALT = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Active layer thickness.tif')

# The Time feature is used to index the driving dataset of RTS expansion between every two years
Time = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2020 Time raster\2017_2018.tif')

#The binary_variable is the RTS expansion raster data for two consecutive years
binary_variable = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2022 RTS Expansion Raster\2017-2018expansion.tif')

rasters = {
    "DEM": DEM, "slope": slope, "flat": flat, "East": East, "Northeast": Northeast,
    "Southeast": Southeast, "North": North, "West": West, "Northwest": Northwest, "Southwest": Southwest, "South": South,
    "Profile_Curvature": Profile_Curvature, "TWI": TWI, "Distance_Lake": Distance_Lake,
    "NDVI": NDVI, "precipitation_sum": precipitation_sum, "Max_Summer_Precipitation": Max_Summer_Precipitation,
    "Max_Summer_Temperature": Max_Summer_Temperature, "Distance_Faults": Distance_Faults,
    "sand0_5": sand0_5, "sand5_15": sand5_15, "sand15_30": sand15_30, "sand30_60": sand30_60, "sand60_100": sand60_100,
    "sand100_200": sand100_200,
    "clay0_5": clay0_5, "clay5_15": clay5_15, "clay15_30": clay15_30, "clay30_60": clay30_60, "clay60_100": clay60_100,
    "clay100_200": clay100_200,
    "silt0_5": silt0_5, "silt5_15": silt5_15, "silt15_30": silt15_30, "silt30_60": silt30_60, "silt60_100": silt60_100,
    "silt100_200": silt100_200,
    "hard_rock": hard_rock, "weak_rock": weak_rock, "semi_hard_rock": semi_hard_rock,
    "loose_rock": loose_rock, "Distance_QTR": Distance_QTR, "Bareland": Bareland, "grassland": grassland, "meadow": meadow,
    "water_body": water_body, "wetland": wetland,
    "FDD": FDD, "TDD": TDD, "Ground_Ice": Ground_Ice, "ALT": ALT,
    "Time": Time,
    "binary_variable": binary_variable
}

reference_raster = rasters["DEM"]

global_mask = np.zeros_like(reference_raster, dtype=bool)  

for key, raster in rasters.items():

    mask = (raster == -9999)

    global_mask |= mask

for key, raster in rasters.items():

    raster = raster.astype(np.float32)
    raster[global_mask] = -9999
    rasters[key] = raster

# Update variables
DEM = rasters["DEM"]
slope = rasters["slope"]
flat = rasters["flat"]
East = rasters["East"]
Northeast = rasters["Northeast"]
Southeast = rasters["Southeast"]
North = rasters["North"]
West = rasters["West"]
Northwest = rasters["Northwest"]
Southwest = rasters["Southwest"]
South = rasters["South"]

Profile_Curvature = rasters["Profile_Curvature"]
TWI = rasters["TWI"]
Distance_Lake = rasters["Distance_Lake"]
NDVI = rasters["NDVI"]
precipitation_sum = rasters["precipitation_sum"]
Max_Summer_Precipitation = rasters["Max_Summer_Precipitation"]
Max_Summer_Temperature = rasters["Max_Summer_Temperature"]
Distance_Faults = rasters["Distance_Faults"]

sand0_5 = rasters["sand0_5"]
sand5_15 = rasters["sand5_15"]
sand15_30 = rasters["sand15_30"]
sand30_60 = rasters["sand30_60"]
sand60_100 = rasters["sand60_100"]
sand100_200 = rasters["sand100_200"]

clay0_5 = rasters["clay0_5"]
clay5_15 = rasters["clay5_15"]
clay15_30 = rasters["clay15_30"]
clay30_60 = rasters["clay30_60"]
clay60_100 = rasters["clay60_100"]
clay100_200 = rasters["clay100_200"]

silt0_5 = rasters["silt0_5"]
silt5_15 = rasters["silt5_15"]
silt15_30 = rasters["silt15_30"]
silt30_60 = rasters["silt30_60"]
silt60_100 = rasters["silt60_100"]
silt100_200 = rasters["silt100_200"]

hard_rock = rasters["hard_rock"]
weak_rock = rasters["weak_rock"]
semi_hard_rock = rasters["semi_hard_rock"]
loose_rock = rasters["loose_rock"]
Distance_QTR = rasters["Distance_QTR"]
Bareland = rasters["Bareland"]
grassland = rasters["grassland"]
meadow = rasters["meadow"]
water_body = rasters["water_body"]
wetland = rasters["wetland"]
FDD = rasters["FDD"]
TDD = rasters["TDD"]
Ground_Ice = rasters["Ground_Ice"]
ALT = rasters["ALT"]
Time = rasters["Time"]

n_samples = DEM.shape[0] * DEM.shape[1]

# Convert raster data into a feature matrix
X201718 = np.column_stack([
    DEM.flatten(), slope.flatten(), flat.flatten(), East.flatten(), Northeast.flatten(),
    Southeast.flatten(), North.flatten(), West.flatten(), Northwest.flatten(), Southwest.flatten(), South.flatten(),
    Profile_Curvature.flatten(),
    TWI.flatten(), Distance_Lake.flatten(), NDVI.flatten(),
    precipitation_sum.flatten(), Max_Summer_Precipitation.flatten(), Max_Summer_Temperature.flatten(),
    Distance_Faults.flatten(),
    sand0_5.flatten(),
    sand5_15.flatten(),
    sand15_30.flatten(),
    sand30_60.flatten(),
    sand60_100.flatten(),
    sand100_200.flatten(),
    clay0_5.flatten(),
    clay5_15.flatten(),
    clay15_30.flatten(),
    clay30_60.flatten(),
    clay60_100.flatten(),
    clay100_200.flatten(),
    silt0_5.flatten(),
    silt5_15.flatten(),
    silt15_30.flatten(),
    silt30_60.flatten(),
    silt60_100.flatten(),
    silt100_200.flatten(),
    hard_rock.flatten(),
    weak_rock.flatten(), semi_hard_rock.flatten(), loose_rock.flatten(),
    Distance_QTR.flatten(), Bareland.flatten(), grassland.flatten(), meadow.flatten(), water_body.flatten(),
    wetland.flatten(),
    FDD.flatten(), TDD.flatten(), Ground_Ice.flatten(), ALT.flatten(), Time.flatten()
])

# Convert the dependent variable raster data to the target vector
y201718 = binary_variable.flatten()

# Check if -9999 exists in y
y_mask201718 = (y201718 == -9999) | (y201718 == -1)

# Use Boolean indexing to remove rows containing -9999 in y
X201718 = X201718[~y_mask201718]
y201718 = y201718[~y_mask201718]

# Check if -9999 exists in X，If a row contains -9999, the mask for that row is True
mask = (X201718 == -9999).any(axis=1)

# Use Boolean indexing to remove rows containing -9999 in X
X_clean201718 = X201718[~mask]
y_clean201718 = y201718[~mask]

# print("“After removing rows containing -9999:”")
# print("X201718 shape:", X_clean201718.shape)
# print("y201718 shape:", y_clean201718.shape)
categorical_columns = [
    'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South',
   'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock','Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland', 'Time'
]
all_columns = [
     'DEM', 'Slope', 'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South', 'Profile_Curvature',
    'TWI', 'Distance_Lake', 'NDVI',
    'Precipitation_sum', 'Max_Summer_Precipitation', 'Max_Summer_Temperature',
    'Distance_Faults', 'sand0_5','sand5_15','sand15_30','sand30_60','sand60_100','sand100_200',
    'clay0_5','clay5_15','clay15_30','clay30_60','clay60_100','clay100_200',
    'silt0_5','silt5_15','silt15_30','silt30_60', 'silt60_100','silt100_200', 'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock',
    'Distance_QTR', 'Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland',
    'FDD', 'TDD', 'Ground_Ice', 'ALT',
    'Time'
]
non_categorical_columns = [col for col in all_columns if col not in categorical_columns]
non_categorical_indices = [all_columns.index(col) for col in non_categorical_columns]
for idx in non_categorical_indices:
    col = X_clean201718[:, idx]
    col_min = np.min(col)
    col_max = np.max(col)
    X_clean201718[:, idx] = (col - col_min) / (col_max - col_min)



"""
"""
# Read RTS-driven data for 2018-2019
# Reading topographic factors: DEM, slope, aspect (flat, north, northeast, east, northwest, west, southeast, southwest, south), and profile curvature.

# Hydrological Vegetation：TWI, Distance to Rivers and Lakes, NDVI
NDVI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\NDVI201809_201908mean.tif')

# Climate factors: Cumulative precipitation , Maximum precipitation, Highest temperature in summer
precipitation_sum = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201809_201908Cumulative Precipitation.tif')
Max_Summer_Precipitation = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201809_201908Maximum Summer Precipitation.tif')
Max_Summer_Temperature = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201809_201908Maximum Summer Temperature.tif')

# Topography
# soil_texture
# Lithology: Hard rock、Weak rock 、Semi-Hard rock、 Loose rock
# anthropogenic factors: Distance to QTR  LULC(grassland、meadow、water body、wetland、bare land)
# LULC

# permafrost characteristics（FDD、TDD、Ground Ice Content、Active layer thickness）
FDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201809_201908FDD.tif')
TDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201809_201908TDD.tif')

# The Time feature is used to index the driving dataset of RTS expansion between every two years
Time = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2020 Time raster\2018_2019.tif')

#The binary_variable is the RTS expansion raster data for two consecutive years
binary_variable = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2022 RTS Expansion Raster\2018-2019expansion.tif')

rasters = {
    "DEM": DEM, "slope": slope, "flat": flat, "East": East, "Northeast": Northeast,
    "Southeast": Southeast, "North": North, "West": West, "Northwest": Northwest, "Southwest": Southwest, "South": South,
    "Profile_Curvature": Profile_Curvature, "TWI": TWI, "Distance_Lake": Distance_Lake,
    "NDVI": NDVI, "precipitation_sum": precipitation_sum, "Max_Summer_Precipitation": Max_Summer_Precipitation,
    "Max_Summer_Temperature": Max_Summer_Temperature, "Distance_Faults": Distance_Faults,
    "sand0_5": sand0_5, "sand5_15": sand5_15, "sand15_30": sand15_30, "sand30_60": sand30_60, "sand60_100": sand60_100,
    "sand100_200": sand100_200,
    "clay0_5": clay0_5, "clay5_15": clay5_15, "clay15_30": clay15_30, "clay30_60": clay30_60, "clay60_100": clay60_100,
    "clay100_200": clay100_200,
    "silt0_5": silt0_5, "silt5_15": silt5_15, "silt15_30": silt15_30, "silt30_60": silt30_60, "silt60_100": silt60_100,
    "silt100_200": silt100_200,
    "hard_rock": hard_rock, "weak_rock": weak_rock, "semi_hard_rock": semi_hard_rock,
    "loose_rock": loose_rock, "Distance_QTR": Distance_QTR, "Bareland": Bareland, "grassland": grassland, "meadow": meadow,
    "water_body": water_body, "wetland": wetland,
    "FDD": FDD, "TDD": TDD, "Ground_Ice": Ground_Ice, "ALT": ALT,
    "Time": Time,
    "binary_variable": binary_variable
}

reference_raster = rasters["DEM"]

global_mask = np.zeros_like(reference_raster, dtype=bool)

for key, raster in rasters.items():
    mask = (raster == -9999)
    global_mask |= mask

for key, raster in rasters.items():
    raster = raster.astype(np.float32)
    raster[global_mask] = -9999
    rasters[key] = raster

# Update variables
DEM = rasters["DEM"]
slope = rasters["slope"]
flat = rasters["flat"]
East = rasters["East"]
Northeast = rasters["Northeast"]
Southeast = rasters["Southeast"]
North = rasters["North"]
West = rasters["West"]
Northwest = rasters["Northwest"]
Southwest = rasters["Southwest"]
South = rasters["South"]

Profile_Curvature = rasters["Profile_Curvature"]
TWI = rasters["TWI"]
Distance_Lake = rasters["Distance_Lake"]
NDVI = rasters["NDVI"]
precipitation_sum = rasters["precipitation_sum"]
Max_Summer_Precipitation = rasters["Max_Summer_Precipitation"]
Max_Summer_Temperature = rasters["Max_Summer_Temperature"]
Distance_Faults = rasters["Distance_Faults"]

sand0_5 = rasters["sand0_5"]
sand5_15 = rasters["sand5_15"]
sand15_30 = rasters["sand15_30"]
sand30_60 = rasters["sand30_60"]
sand60_100 = rasters["sand60_100"]
sand100_200 = rasters["sand100_200"]

clay0_5 = rasters["clay0_5"]
clay5_15 = rasters["clay5_15"]
clay15_30 = rasters["clay15_30"]
clay30_60 = rasters["clay30_60"]
clay60_100 = rasters["clay60_100"]
clay100_200 = rasters["clay100_200"]

silt0_5 = rasters["silt0_5"]
silt5_15 = rasters["silt5_15"]
silt15_30 = rasters["silt15_30"]
silt30_60 = rasters["silt30_60"]
silt60_100 = rasters["silt60_100"]
silt100_200 = rasters["silt100_200"]

hard_rock = rasters["hard_rock"]
weak_rock = rasters["weak_rock"]
semi_hard_rock = rasters["semi_hard_rock"]
loose_rock = rasters["loose_rock"]
Distance_QTR = rasters["Distance_QTR"]
Bareland = rasters["Bareland"]
grassland = rasters["grassland"]
meadow = rasters["meadow"]
water_body = rasters["water_body"]
wetland = rasters["wetland"]
FDD = rasters["FDD"]
TDD = rasters["TDD"]
Ground_Ice = rasters["Ground_Ice"]
ALT = rasters["ALT"]
Time = rasters["Time"]

n_samples = DEM.shape[0] * DEM.shape[1]

# Convert raster data into a feature matrix
X201819 = np.column_stack([
    DEM.flatten(), slope.flatten(), flat.flatten(), East.flatten(), Northeast.flatten(),
    Southeast.flatten(), North.flatten(), West.flatten(), Northwest.flatten(), Southwest.flatten(), South.flatten(),
    Profile_Curvature.flatten(),
    TWI.flatten(), Distance_Lake.flatten(), NDVI.flatten(),
    precipitation_sum.flatten(), Max_Summer_Precipitation.flatten(), Max_Summer_Temperature.flatten(),
    Distance_Faults.flatten(),
    sand0_5.flatten(),
    sand5_15.flatten(),
    sand15_30.flatten(),
    sand30_60.flatten(),
    sand60_100.flatten(),
    sand100_200.flatten(),
    clay0_5.flatten(),
    clay5_15.flatten(),
    clay15_30.flatten(),
    clay30_60.flatten(),
    clay60_100.flatten(),
    clay100_200.flatten(),
    silt0_5.flatten(),
    silt5_15.flatten(),
    silt15_30.flatten(),
    silt30_60.flatten(),
    silt60_100.flatten(),
    silt100_200.flatten(),
    hard_rock.flatten(),
    weak_rock.flatten(), semi_hard_rock.flatten(), loose_rock.flatten(),
    Distance_QTR.flatten(), Bareland.flatten(), grassland.flatten(), meadow.flatten(), water_body.flatten(),
    wetland.flatten(),
    FDD.flatten(), TDD.flatten(), Ground_Ice.flatten(), ALT.flatten(), Time.flatten()
])

# Convert the dependent variable raster data to the target vector
y201819 = binary_variable.flatten()

# Check if -9999 exists in y
y_mask201819 = (y201819 == -9999) | (y201819 == -1)

# Use Boolean indexing to remove rows containing -9999 in y
X201819 = X201819[~y_mask201819]
y201819 = y201819[~y_mask201819]

# Check if -9999 exists in X，If a row contains -9999, the mask for that row is True
mask = (X201819 == -9999).any(axis=1)

# Use Boolean indexing to remove rows containing -9999 in X
X_clean201819 = X201819[~mask]
y_clean201819 = y201819[~mask]

# print("“After removing rows containing -9999:”")
# print("X201819 shape:", X_clean201819.shape)
# print("y201819 shape:", y_clean201819.shape)
categorical_columns = [
    'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South',
   'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock','Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland', 'Time'
]

all_columns = [
     'DEM', 'Slope', 'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South', 'Profile_Curvature',
    'TWI', 'Distance_Lake', 'NDVI',
    'Precipitation_sum', 'Max_Summer_Precipitation', 'Max_Summer_Temperature',
    'Distance_Faults', 'sand0_5','sand5_15','sand15_30','sand30_60','sand60_100','sand100_200',
    'clay0_5','clay5_15','clay15_30','clay30_60','clay60_100','clay100_200',
    'silt0_5','silt5_15','silt15_30','silt30_60', 'silt60_100','silt100_200', 'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock',
    'Distance_QTR', 'Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland',
    'FDD', 'TDD', 'Ground_Ice', 'ALT',
    'Time'
]
non_categorical_columns = [col for col in all_columns if col not in categorical_columns]
non_categorical_indices = [all_columns.index(col) for col in non_categorical_columns]
for idx in non_categorical_indices:
    col = X_clean201819[:, idx]
    col_min = np.min(col)
    col_max = np.max(col)
    X_clean201819[:, idx] = (col - col_min) / (col_max - col_min)
"""

"""
# Read RTS-driven data for 2019-2020
# Reading topographic factors: DEM, slope, aspect (flat, north, northeast, east, northwest, west, southeast, southwest, south), and profile curvature.

# Hydrological Vegetation：TWI, Distance to Rivers and Lakes, NDVI
NDVI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\NDVI201909_202008mean.tif')

# Climate factors: Cumulative precipitation , Maximum precipitation, Highest temperature in summer
precipitation_sum = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201909_202008Cumulative Precipitation.tif')
Max_Summer_Precipitation = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201909_202008Maximum Summer Precipitation.tif')
Max_Summer_Temperature = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201909_202008Maximum Summer Temperature.tif')

# Topography
# soil_texture
# Lithology: Hard rock、Weak rock 、Semi-Hard rock、 Loose rock
# anthropogenic factors: Distance to QTR  LULC(grassland、meadow、water body、wetland、bare land)
# LULC

# permafrost characteristics（FDD、TDD、Ground Ice Content、Active layer thickness）
FDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201909_202008FDD.tif')
TDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\201909_202008TDD.tif')

# The Time feature is used to index the driving dataset of RTS expansion between every two years
Time = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2020 Time raster\2019_2020.tif')

#The binary_variable is the RTS expansion raster data for two consecutive years
binary_variable = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\2016-2022 RTS Expansion Raster\2019-2020expansion.tif')

rasters = {
    "DEM": DEM, "slope": slope, "flat": flat, "East": East, "Northeast": Northeast,
    "Southeast": Southeast, "North": North, "West": West, "Northwest": Northwest, "Southwest": Southwest, "South": South,
    "Profile_Curvature": Profile_Curvature, "TWI": TWI, "Distance_Lake": Distance_Lake,
    "NDVI": NDVI, "precipitation_sum": precipitation_sum, "Max_Summer_Precipitation": Max_Summer_Precipitation,
    "Max_Summer_Temperature": Max_Summer_Temperature, "Distance_Faults": Distance_Faults,
    "sand0_5": sand0_5, "sand5_15": sand5_15, "sand15_30": sand15_30, "sand30_60": sand30_60, "sand60_100": sand60_100,
    "sand100_200": sand100_200,
    "clay0_5": clay0_5, "clay5_15": clay5_15, "clay15_30": clay15_30, "clay30_60": clay30_60, "clay60_100": clay60_100,
    "clay100_200": clay100_200,
    "silt0_5": silt0_5, "silt5_15": silt5_15, "silt15_30": silt15_30, "silt30_60": silt30_60, "silt60_100": silt60_100,
    "silt100_200": silt100_200,
    "hard_rock": hard_rock, "weak_rock": weak_rock, "semi_hard_rock": semi_hard_rock,
    "loose_rock": loose_rock, "Distance_QTR": Distance_QTR, "Bareland": Bareland, "grassland": grassland, "meadow": meadow,
    "water_body": water_body, "wetland": wetland,
    "FDD": FDD, "TDD": TDD, "Ground_Ice": Ground_Ice, "ALT": ALT,
    "Time": Time,
    "binary_variable": binary_variable
}

reference_raster = rasters["DEM"]
global_mask = np.zeros_like(reference_raster, dtype=bool)

for key, raster in rasters.items():
    mask = (raster == -9999)
    global_mask |= mask

for key, raster in rasters.items():
    raster = raster.astype(np.float32)
    raster[global_mask] = -9999
    rasters[key] = raster

# Update variables
DEM = rasters["DEM"]
slope = rasters["slope"]
flat = rasters["flat"]
East = rasters["East"]
Northeast = rasters["Northeast"]
Southeast = rasters["Southeast"]
North = rasters["North"]
West = rasters["West"]
Northwest = rasters["Northwest"]
Southwest = rasters["Southwest"]
South = rasters["South"]

Profile_Curvature = rasters["Profile_Curvature"]
TWI = rasters["TWI"]
Distance_Lake = rasters["Distance_Lake"]
NDVI = rasters["NDVI"]
precipitation_sum = rasters["precipitation_sum"]
Max_Summer_Precipitation = rasters["Max_Summer_Precipitation"]
Max_Summer_Temperature = rasters["Max_Summer_Temperature"]
Distance_Faults = rasters["Distance_Faults"]

sand0_5 = rasters["sand0_5"]
sand5_15 = rasters["sand5_15"]
sand15_30 = rasters["sand15_30"]
sand30_60 = rasters["sand30_60"]
sand60_100 = rasters["sand60_100"]
sand100_200 = rasters["sand100_200"]

clay0_5 = rasters["clay0_5"]
clay5_15 = rasters["clay5_15"]
clay15_30 = rasters["clay15_30"]
clay30_60 = rasters["clay30_60"]
clay60_100 = rasters["clay60_100"]
clay100_200 = rasters["clay100_200"]

silt0_5 = rasters["silt0_5"]
silt5_15 = rasters["silt5_15"]
silt15_30 = rasters["silt15_30"]
silt30_60 = rasters["silt30_60"]
silt60_100 = rasters["silt60_100"]
silt100_200 = rasters["silt100_200"]

hard_rock = rasters["hard_rock"]
weak_rock = rasters["weak_rock"]
semi_hard_rock = rasters["semi_hard_rock"]
loose_rock = rasters["loose_rock"]
Distance_QTR = rasters["Distance_QTR"]
Bareland = rasters["Bareland"]
grassland = rasters["grassland"]
meadow = rasters["meadow"]
water_body = rasters["water_body"]
wetland = rasters["wetland"]
FDD = rasters["FDD"]
TDD = rasters["TDD"]
Ground_Ice = rasters["Ground_Ice"]
ALT = rasters["ALT"]
Time = rasters["Time"]

n_samples = DEM.shape[0] * DEM.shape[1]

# Convert raster data into a feature matrix
X201920 = np.column_stack([
    DEM.flatten(), slope.flatten(), flat.flatten(), East.flatten(), Northeast.flatten(),
    Southeast.flatten(), North.flatten(), West.flatten(), Northwest.flatten(), Southwest.flatten(), South.flatten(),
    Profile_Curvature.flatten(),
    TWI.flatten(), Distance_Lake.flatten(), NDVI.flatten(),
    precipitation_sum.flatten(), Max_Summer_Precipitation.flatten(), Max_Summer_Temperature.flatten(),
    Distance_Faults.flatten(),
    sand0_5.flatten(),
    sand5_15.flatten(),
    sand15_30.flatten(),
    sand30_60.flatten(),
    sand60_100.flatten(),
    sand100_200.flatten(),
    clay0_5.flatten(),
    clay5_15.flatten(),
    clay15_30.flatten(),
    clay30_60.flatten(),
    clay60_100.flatten(),
    clay100_200.flatten(),
    silt0_5.flatten(),
    silt5_15.flatten(),
    silt15_30.flatten(),
    silt30_60.flatten(),
    silt60_100.flatten(),
    silt100_200.flatten(),
    hard_rock.flatten(),
    weak_rock.flatten(), semi_hard_rock.flatten(), loose_rock.flatten(),
    Distance_QTR.flatten(), Bareland.flatten(), grassland.flatten(), meadow.flatten(), water_body.flatten(),
    wetland.flatten(),
    FDD.flatten(), TDD.flatten(), Ground_Ice.flatten(), ALT.flatten(), Time.flatten()
])

# Convert the dependent variable raster data to the target vector
y201920 = binary_variable.flatten()

# Check if -9999 exists in y
y_mask201920 = (y201920 == -9999) | (y201920 == -1)

# Use Boolean indexing to remove rows containing -9999 in y
X201920 = X201920[~y_mask201920]
y201920 = y201920[~y_mask201920]

# Check if -9999 exists in X，If a row contains -9999, the mask for that row is True
mask = (X201920 == -9999).any(axis=1)

# Use Boolean indexing to remove rows containing -9999 in X
X_clean201920 = X201920[~mask]
y_clean201920 = y201920[~mask]

# print("“After removing rows containing -9999:”")
# print("X201920 shape:", X_clean201920.shape)
# print("y201920 shape:", y_clean201920.shape)

#Combined Feature Matrix
X2016_2020 = np.vstack([
    X_clean201617,
    X_clean201718,
    X_clean201819,
    X_clean201920
])

# Combined label vector
y2016_2020 = np.concatenate([
    y_clean201617,
    y_clean201718,
    y_clean201819,
    y_clean201920
])
# Count the number of 1 values in y_clean201617
num_ones = np.sum(y_clean201617 == 1)

#Separate samples with 0 and 1 values
X201617_0 = X_clean201617[y_clean201617 == 0]
y201617_0 = y_clean201617[y_clean201617 == 0]
X201617_1 = X_clean201617[y_clean201617 == 1]
y201617_1 = y_clean201617[y_clean201617 == 1]

# Ensure the number of samples with value 1 is less than or equal to the number of samples with value 0
assert num_ones <= len(X201617_0)

# Downsample the samples with value 0 to equal the number of samples with value 1
X201617_0_downsampled, y201617_0_downsampled = resample(X201617_0, y201617_0, n_samples=num_ones, random_state=42)

# Recombine the downsampled data.
X201617_balanced = np.vstack([X201617_0_downsampled, X201617_1])
y201617_balanced = np.concatenate([y201617_0_downsampled, y201617_1])

print("X201617_balanced shape:", X201617_balanced.shape)
print("y201617_balanced shape:", y201617_balanced.shape)

"""
Apply the same procedure to obtain 
    X201718_balanced,
    X201819_balanced,
    X201920_balanced,y201718_balanced,
    y201819_balanced,
    y201920_balanced
"""
num_ones = np.sum(y_clean201718 == 1)
X201718_0 = X_clean201718[y_clean201718 == 0]
y201718_0 = y_clean201718[y_clean201718 == 0]
X201718_1 = X_clean201718[y_clean201718 == 1]
y201718_1 = y_clean201718[y_clean201718 == 1]
assert num_ones <= len(X201718_0)
X201718_0_downsampled, y201718_0_downsampled = resample(X201718_0, y201718_0, n_samples=num_ones, random_state=42)
X201718_balanced = np.vstack([X201718_0_downsampled, X201718_1])
y201718_balanced = np.concatenate([y201718_0_downsampled, y201718_1])
print("X201718_balanced shape:", X201718_balanced.shape)
print("y201718_balanced shape:", y201718_balanced.shape)

num_ones = np.sum(y_clean201819 == 1)
X201819_0 = X_clean201819[y_clean201819 == 0]
y201819_0 = y_clean201819[y_clean201819 == 0]
X201819_1 = X_clean201819[y_clean201819 == 1]
y201819_1 = y_clean201819[y_clean201819 == 1]
assert num_ones <= len(X201819_0)
X201819_0_downsampled, y201819_0_downsampled = resample(X201819_0, y201819_0, n_samples=num_ones, random_state=42)
X201819_balanced = np.vstack([X201819_0_downsampled, X201819_1])
y201819_balanced = np.concatenate([y201819_0_downsampled, y201819_1])
print("X201819_balanced shape:", X201819_balanced.shape)
print("y201819_balanced shape:", y201819_balanced.shape)

num_ones = np.sum(y_clean201920 == 1)
X201920_0 = X_clean201920[y_clean201920 == 0]
y201920_0 = y_clean201920[y_clean201920 == 0]
X201920_1 = X_clean201920[y_clean201920 == 1]
y201920_1 = y_clean201920[y_clean201920 == 1]
assert num_ones <= len(X201920_0)
X201920_0_downsampled, y201920_0_downsampled = resample(X201920_0, y201920_0, n_samples=num_ones, random_state=42)
X201920_balanced = np.vstack([X201920_0_downsampled, X201920_1])
y201920_balanced = np.concatenate([y201920_0_downsampled, y201920_1])
print("X201920_balanced shape:", X201920_balanced.shape)
print("y201920_balanced shape:", y201920_balanced.shape)

X_combined = np.vstack([
    X201617_balanced,
    X201718_balanced,
    X201819_balanced,
    X201920_balanced
])

y_combined = np.concatenate([
    y201617_balanced,
    y201718_balanced,
    y201819_balanced,
    y201920_balanced
])

print("Combined dataset")
print("X_combined shape:", X_combined.shape)
print("y_combined shape:", y_combined.shape)

# Check the number of each category:
unique, counts = np.unique(y_combined, return_counts=True)
print("Category distribution:", dict(zip(unique, counts)))

# List of feature names (ensure consistency with the column order of X_combined)
feature_names = [
    'DEM', 'Slope', 'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South', 'Profile_Curvature',
    'TWI', 'Distance_Lake', 'NDVI',
    'Precipitation_sum', 'Max_Summer_Precipitation', 'Max_Summer_Temperature',
    'Distance_Faults', 'sand0_5','sand5_15','sand15_30','sand30_60','sand60_100','sand100_200',
    'clay0_5','clay5_15','clay15_30','clay30_60','clay60_100','clay100_200',
    'silt0_5','silt5_15','silt15_30','silt30_60', 'silt60_100','silt100_200', 'hard_rock', 'weak_rock', 'semi_hard_rock', 'loose_rock',
    'Distance_QTR', 'Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland',
    'FDD', 'TDD', 'Ground_Ice', 'ALT',
    'Time'
]

# Determine the position of the Time feature in the feature matrix
time_index = feature_names.index('Time')

# Extract the Time feature from X_combined as the grouping criterion
groups = X_combined[:, time_index]
X_notime = np.delete(X_combined, time_index, axis=1)
y_notime = y_combined

X_train, X_test, y_train, y_test = train_test_split(X_notime, y_notime, test_size=0.3, random_state=42)
# 1.Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create RFECV object
#Using 10-fold cross-validation
min_features_to_select = 1
cv = StratifiedKFold(10)

rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=cv,
    scoring='accuracy',
    min_features_to_select=min_features_to_select,
    n_jobs=-1
)

# 3. Perform feature selection
rfecv.fit(X_train, y_train)

Features_names = [
    "DEM", "slope", "flat", "East", "Northeast", "Southeast", "North", "West", "Northwest", "Southwest","South",
    "Profile_Curvature", "TWI", "Distance_Lake", "NDVI", "precipitation_sum",
    "Max_Summer_Precipitation", "Max_Summer_Temperature", "Distance_Faults",
   'sand0_5','sand5_15','sand15_30','sand30_60','sand60_100','sand100_200',
    'clay0_5','clay5_15','clay15_30','clay30_60','clay60_100','clay100_200',
    'silt0_5','silt5_15','silt15_30','silt30_60', 'silt60_100','silt100_200', "hard_rock", "weak_rock", "semi_hard_rock", "loose_rock",
    "Distance_QTR", "Bareland", "grassland", "meadow", "water_body", "wetland",
    "FDD", "TDD", "Ground_Ice", "ALT"
]


#Obtain the selected feature mask
selected_features_mask = rfecv.support_
#Extract optimal feature names based on the mask
selected_features_names = [feature for feature, is_selected in zip(Features_names, selected_features_mask) if is_selected]

# RFECV outputs
# print('Optimal number of features:', rfecv.n_features_)
# print('Selected feature mask:', rfecv.support_)
# print('Ranking of each feature:', rfecv.ranking_)

# Output optimal feature names
# print("optimal feature names：", selected_features_names)

# Get the selected feature indices
selected_features_idx = np.where(rfecv.support_)[0]
# print("Selected feature indices:", selected_features_idx)

# 5. Transform the dataset
X_train_selected2 = rfecv.transform(X_train)
X_test_selected2 = rfecv.transform(X_test)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Visualizing the relationship between the number of features and cross-validation scores
# plt.figure()
# plt.xlabel("the number of features")
# plt.ylabel("cross-validation scores")
# mean_scores = rfecv.cv_results_['mean_test_score']
# std_scores = rfecv.cv_results_['std_test_score']
#
# # Plot the average score
# plt.plot(range(1, len(mean_scores) + 1), mean_scores, label='average score', color='darkorange')
#
# # Plot the standard deviation range
# plt.fill_between(range(1, len(mean_scores) + 1), mean_scores - std_scores, mean_scores + std_scores, alpha=0.3, color='darkorange')
#
# plt.legend(loc="lower right")
# plt.show()

# Retrain the model using the selected feature subset
selected_features =  selected_features_names
support = rfecv.support_
time_index = feature_names.index('Time')

# Extract the Time feature from X_combined as the grouping criterion
groups = X_combined[:, time_index]
X_notime = np.delete(X_combined, time_index, axis=1)  
y_notime = y_combined  
X_selected = X_notime[:, support]

# Split the training set and test set
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_selected, y_notime, test_size=0.3, random_state=42)

# Define the RF model
rf_model2 = RandomForestClassifier(random_state=42)

# Define the hyperparameter range
param_ranges = {
    'n_estimators': [100, 1000],  # Range for the number of trees
    'max_depth': [2, 20],  # Range for the maximum depth of each tree
    'min_samples_split': [2, 10],  # Range for the minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 20],  # Range for the minimum number of samples required to be in a leaf node
    'max_features': [1, 10]  # Range for the maximum number of features
}

n_samples = 100
n_params = len(param_ranges)
lhs_samples = lhs(n_params, samples=n_samples, criterion='maximin')

param_names = list(param_ranges.keys())
param_values = np.array([param_ranges[name] for name in param_names]).T
param_samples = param_values[0] + lhs_samples * (param_values[1] - param_values[0])


cv_strategy = StratifiedKFold(n_splits=10)


def evaluate_params(params):
    rf_model2.set_params(**params)
    scores = cross_val_score(rf_model2, X_train2, y_train2, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

results = Parallel(n_jobs=-1)(
    delayed(evaluate_params)(
        {param_names[i]: int(param_samples[j, i]) if param_names[i] in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'] else param_samples[j, i] for i in range(n_params)}
    ) for j in range(n_samples)
)


best_score = max(results)
best_index = results.index(best_score)
best_params = {param_names[i]: int(param_samples[best_index, i]) if param_names[i] in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'] else param_samples[best_index, i] for i in range(n_params)}
print("best_params：", best_params)
print("best_score：", best_score)

#Plot development probability map
feature_names = [
    'DEM', 'Slope', 'flat', 'East', 'Northeast', 'Southeast', 'North', 'West', 'Northwest', 'Southwest', 'South', 'Profile_Curvature',
    'TWI', 'Distance_Lake', 'NDVI',
    'Precipitation_sum', 'Max_Summer_Precipitation', 'Max_Summer_Temperature',
    'Distance_Faults', 'sand0_5', 'sand5_15', 'sand15_30', 'sand30_60', 'sand60_100', 'sand100_200',
    'clay0_5', 'clay5_15', 'clay15_30', 'clay30_60', 'clay60_100', 'clay100_200',
    'silt0_5', 'silt5_15', 'silt15_30', 'silt30_60', 'silt60_100', 'silt100_200', 'hard_rock', 'weak_rock', 'semi_hard_rock',
    'loose_rock',
    'Distance_QTR', 'Bareland', 'Grassland', 'Meadow', 'Water_body', 'Wetland',
    'FDD', 'TDD', 'Ground_Ice', 'ALT',
    'Time'
]

time_index = feature_names.index('Time')

X_notime = np.delete(X_combined, time_index, axis=1)
y_notime = y_combined

selected_features = selected_features_names

X_selected = X_notime[:, rfecv.support_]


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_selected, y_notime, test_size=0.3, random_state=42)

feature_indices = [Features_names.index(f) for f in selected_features]

X_selected = X_notime[:, feature_indices]

rf_model = RandomForestClassifier(**best_params, random_state=42)
rf_model.fit(X_selected, y_notime)
"""
read driven data 2020-2021 2021-2022
"""
## Reading topographic factors: DEM, slope, aspect (flat, north, northeast, east, northwest, west, southeast, southwest, south), and profile curvature.
DEM = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\DEM.tif')
slope = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\slope.tif')
#aspect
flat = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\flat slope.tif')
East = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\East slope.tif')
Northeast = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Northeast slope.tif')
Southeast = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Southeast slope.tif')
North = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\North slope.tif')
West = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\west slope.tif')
Northwest = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Northwest slope.tif')
Southwest = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Southwest slope.tif')
South = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\South slope.tif')

Profile_Curvature = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Profile Curvature.tif')

# Hydrological Vegetation：TWI, Distance to Rivers and Lakes, NDVI
TWI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\TWI.tif')
Distance_Lake = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Lake.tif')
NDVI = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\NDVI202009_202108mean.tif')

# Climate factors: Cumulative precipitation , Maximum precipitation, Highest temperature in summer
precipitation_sum = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\202009_202108Cumulative Precipitation.tif')
Max_Summer_Precipitation = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\202009_202108Maximum Summer Precipitation.tif')
Max_Summer_Temperature = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\202009_202108Maximum Summer Temperature.tif')

# Topography
Distance_Faults = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Fault.tif')
# soil_texture
sand0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand0_5.tif')
sand5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand5_15.tif')
sand15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand15_30.tif')
sand30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand30_60.tif')
sand60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand60_100.tif')
sand100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\sand100_200.tif')
clay0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay0_5.tif')
clay5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay5_15.tif')
clay15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay15_30.tif')
clay30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay30_60.tif')
clay60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay60_100.tif')
clay100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\clay100_200.tif')
silt0_5 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt0_5.tif')
silt5_15 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt5_15.tif')
silt15_30 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt15_30.tif')
silt30_60 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt30_60.tif')
silt60_100 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt60_100.tif')
silt100_200 = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\silt100_200.tif')
# Lithology: Hard rock、Weak rock 、Semi-Hard rock、 Loose rock
hard_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\hard rock.tif')
weak_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\weak rock.tif')
semi_hard_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\semi-hard rock.tif')
loose_rock = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\loose rock.tif')

# anthropogenic factors: Distance to QTR  LULC(grassland、meadow、water body、wetland、bare land)
Distance_QTR = read_raster(
    r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Distance from Qinghai-Tibet Railway.tif')
# LULC
Bareland = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\bareland.tif')
grassland = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\grassland.tif')
meadow = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\meadow.tif')
water_body = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Water_body.tif')
wetland = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Categorical variables\Wetland.tif')

# permafrost characteristics（FDD、TDD、Ground Ice Content、Active layer thickness）
FDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\202009_202108FDD.tif')
TDD = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\202009_202108TDD.tif')
Ground_Ice = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Ground ice content.tif')
ALT = read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\Active layer thickness.tif')


#The binary_variable is the RTS expansion raster data for two consecutive years
binary_variable = read_raster(
    r'RTSEvo model driving data and results/Experiment 2/inputs/2016-2022 RTS Expansion Raster\2020-2021expansion.tif')

rasters = {
    "DEM": DEM, "slope": slope, "flat": flat, "East": East, "Northeast": Northeast,
    "Southeast": Southeast, "North": North, "West": West, "Northwest": Northwest, "Southwest": Southwest, "South": South,
    "Profile_Curvature": Profile_Curvature, "TWI": TWI, "Distance_Lake": Distance_Lake,
    "NDVI": NDVI, "precipitation_sum": precipitation_sum, "Max_Summer_Precipitation": Max_Summer_Precipitation,
    "Max_Summer_Temperature": Max_Summer_Temperature, "Distance_Faults": Distance_Faults,
    "sand0_5": sand0_5, "sand5_15": sand5_15, "sand15_30": sand15_30, "sand30_60": sand30_60, "sand60_100": sand60_100,
    "sand100_200": sand100_200,
    "clay0_5": clay0_5, "clay5_15": clay5_15, "clay15_30": clay15_30, "clay30_60": clay30_60, "clay60_100": clay60_100,
    "clay100_200": clay100_200,
    "silt0_5": silt0_5, "silt5_15": silt5_15, "silt15_30": silt15_30, "silt30_60": silt30_60, "silt60_100": silt60_100,
    "silt100_200": silt100_200, "hard_rock": hard_rock, "weak_rock": weak_rock, "semi_hard_rock": semi_hard_rock,
    "loose_rock": loose_rock, "Distance_QTR": Distance_QTR, "Bareland": Bareland, "grassland": grassland, "meadow": meadow,
    "water_body": water_body, "wetland": wetland,
    "FDD": FDD, "TDD": TDD, "Ground_Ice": Ground_Ice, "ALT": ALT,
    "binary_variable": binary_variable
}

reference_raster = rasters["DEM"]

global_mask = np.zeros_like(reference_raster, dtype=bool)

for key, raster in rasters.items():
    mask = (raster == -9999)
    global_mask |= mask

for key, raster in rasters.items():

    raster = raster.astype(np.float32)
    raster[global_mask] = -9999
    rasters[key] = raster

DEM = rasters["DEM"]
slope = rasters["slope"]
flat = rasters["flat"]
East = rasters["East"]
Northeast = rasters["Northeast"]
Southeast = rasters["Southeast"]
North = rasters["North"]
West = rasters["West"]
Northwest = rasters["Northwest"]
Southwest = rasters["Southwest"]
South = rasters["South"]

Profile_Curvature = rasters["Profile_Curvature"]
TWI = rasters["TWI"]
Distance_Lake = rasters["Distance_Lake"]
NDVI = rasters["NDVI"]
precipitation_sum = rasters["precipitation_sum"]
Max_Summer_Precipitation = rasters["Max_Summer_Precipitation"]
Max_Summer_Temperature = rasters["Max_Summer_Temperature"]
Distance_Faults = rasters["Distance_Faults"]
sand0_5 = rasters["sand0_5"]
sand5_15 = rasters["sand5_15"]
sand15_30 = rasters["sand15_30"]
sand30_60 = rasters["sand30_60"]
sand60_100 = rasters["sand60_100"]
sand100_200 = rasters["sand100_200"]

clay0_5 = rasters["clay0_5"]
clay5_15 = rasters["clay5_15"]
clay15_30 = rasters["clay15_30"]
clay30_60 = rasters["clay30_60"]
clay60_100 = rasters["clay60_100"]
clay100_200 = rasters["clay100_200"]

silt0_5 = rasters["silt0_5"]
silt5_15 = rasters["silt5_15"]
silt15_30 = rasters["silt15_30"]
silt30_60 = rasters["silt30_60"]
silt60_100 = rasters["silt60_100"]
silt100_200 = rasters["silt100_200"]
hard_rock = rasters["hard_rock"]
weak_rock = rasters["weak_rock"]
semi_hard_rock = rasters["semi_hard_rock"]
loose_rock = rasters["loose_rock"]
Distance_QTR = rasters["Distance_QTR"]
Bareland = rasters["Bareland"]
grassland = rasters["grassland"]
meadow = rasters["meadow"]
water_body = rasters["water_body"]
wetland = rasters["wetland"]
FDD = rasters["FDD"]
TDD = rasters["TDD"]
Ground_Ice = rasters["Ground_Ice"]
ALT = rasters["ALT"]

binary_variable = rasters["binary_variable"]
n_samples = DEM.shape[0] * DEM.shape[1]

X202021 = np.column_stack([

    DEM.flatten(), slope.flatten(), flat.flatten(), East.flatten(), Northeast.flatten(),
    Southeast.flatten(), North.flatten(), West.flatten(), Northwest.flatten(), Southwest.flatten(), South.flatten(),
    Profile_Curvature.flatten(),

    TWI.flatten(), Distance_Lake.flatten(), NDVI.flatten(),

    precipitation_sum.flatten(), Max_Summer_Precipitation.flatten(), Max_Summer_Temperature.flatten(),

    Distance_Faults.flatten(), sand0_5.flatten(),
    sand5_15.flatten(),
    sand15_30.flatten(),
    sand30_60.flatten(),
    sand60_100.flatten(),
    sand100_200.flatten(),
    clay0_5.flatten(),
    clay5_15.flatten(),
    clay15_30.flatten(),
    clay30_60.flatten(),
    clay60_100.flatten(),
    clay100_200.flatten(),
    silt0_5.flatten(),
    silt5_15.flatten(),
    silt15_30.flatten(),
    silt30_60.flatten(),
    silt60_100.flatten(),
    silt100_200.flatten(), hard_rock.flatten(),
    weak_rock.flatten(), semi_hard_rock.flatten(), loose_rock.flatten(),

    Distance_QTR.flatten(), Bareland.flatten(), grassland.flatten(), meadow.flatten(), water_body.flatten(),
    wetland.flatten(),

    FDD.flatten(), TDD.flatten(), Ground_Ice.flatten(), ALT.flatten()
])


y202021 = binary_variable.flatten()

y_mask202021 = (y202021 == -9999)

X202021 = X202021[~y_mask202021]
y202021= y202021[~y_mask202021]

mask = (X202021 == -9999).any(axis=1)

X_clean202021 = X202021[~mask]
y_clean202021 = y202021[~mask]

print("After deleting rows that contain -9999：")
print("X202122 shape:", X_clean202021.shape)
print("y202122 shape:", y_clean202021.shape)


def read_raster(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open file: {file_path}")
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()

def save_raster(data, filename, transform, crs):
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(data, 1)
time_index = feature_names.index('Time')
#X_notime202021 = np.delete(X_clean202021, time_index, axis=1)
X_notime202021 = X_clean202021
y_notime202021 = y_clean202021
probabilities = rf_model.predict_proba(X_notime202021[:, support])

# Reading the preprocessed DEM raster data through "Probability Grid Map Processing.py"
DEM2 = read_raster(
     r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\DEM2.tif')
mask2 = (DEM2 != -9999)

# Create an array with the same shape as DEM2, with an initial value of -9999.
prob_y1 = np.full(DEM2.shape, -9999, dtype=float)
prob_y2 = np.full(DEM2.shape, -9999, dtype=float)

# Fill the predicted probabilities into the non-9999 regions.
prob_y1[mask2] = probabilities[:, 0]
prob_y2[mask2] = probabilities[:, 1]

# Read a reference raster file to obtain the transformation and coordinate reference system.
with rasterio.open(
        r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\DEM2.tif') as src:
    transform = src.transform
    crs = src.crs

#Save the probability raster files for non-RTS (y value of 1) and RTS (y value of 2) occurrences.
save_raster(prob_y2, r'RTSEvo model driving data and results\Experiment 2\outputs\RF occur 2020to2021.tif',
            transform, crs)

#Taking the 2021 RTS simulation of LR-CA as an example
aspect= read_raster(r'RTSEvo model driving data and results\Experiment 2\inputs\Numerical variables\aspect.tif')
rasters = {
    "aspect":aspect
}
# Select the DEM as the reference raster to determine the size of the global mask;
reference_raster = rasters["aspect"]

global_mask = np.zeros_like(reference_raster, dtype=bool)

for key, raster in rasters.items():
    mask = (raster == -9999)
    global_mask |= mask

for key, raster in rasters.items():
    raster = raster.astype(np.float32)
    raster[global_mask] = 0
    rasters[key] = raster
aspect = rasters["aspect"]

seed = 43
N = 3
neighborhood_weight = 0.759
alpha = 0.351
beta = 0.498
NON_RTS = 1
RTS = 2

def read_raster(path):
    with rasterio.open(path) as src:
        return np.array(src.read(1)), src.profile

def write_raster(array, profile, path):
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)

@cuda.jit
def calculate_aspect_effect_gpu(aspect, padded_landuse, aspect_effect, rows, cols):
    i, j = cuda.grid(2)
    if i < rows and j < cols:
        center_aspect = aspect[i, j]
        if center_aspect < 0:
            aspect_effect[i, j] = 1.0
            return

        total_weight = 0.0
        weighted_count = 0.0

        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0: continue
                row, col = i + di, j + dj
                if 0 <= row < rows and 0 <= col < cols:
                    dy = i - row
                    dx = col - j
                    direction = math.degrees(math.atan2(dy, dx)) % 360
                    angle_diff = min(abs((center_aspect + 180) % 360 - direction),
                                         360 - abs((center_aspect + 180) % 360 - direction))
                    weight = math.cos(math.radians(angle_diff/2))
                    weighted_count += (padded_landuse[row+1, col+1] == RTS) * weight
                    total_weight += weight

        aspect_effect[i, j] = weighted_count / (total_weight + 1e-10)

def calculate_neighborhood_effect_vectorized(padded_landuse, landuse, neighborhood_weight, N=3):
    neighborhoods = np.lib.stride_tricks.sliding_window_view(padded_landuse, (N, N))
    count_2 = np.sum(neighborhoods == RTS, axis=(2, 3)) - (landuse == RTS).astype(int)
    valid_pixels = (N*N-1)
    return (count_2 * neighborhood_weight) / valid_pixels

def calculate_aspect_effect_vectorized_cpu(landuse, aspect, rows, cols):
    aspect_effect = np.zeros((rows, cols))
    padded_landuse = np.pad(landuse, pad_width=1, mode='constant', constant_values=0)

    y, x = np.mgrid[0:rows, 0:cols]

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0: continue
            ny = y + di
            nx = x + dj
            valid = (ny >= 0) & (ny < rows) & (nx >= 0) & (nx < cols)

            dy = y[valid] - ny[valid]
            dx = nx[valid] - x[valid]
            direction = np.degrees(np.arctan2(dy, dx)) % 360

            center_aspect = aspect[y[valid], x[valid]]
            angle_diff = np.minimum(
                np.abs((center_aspect + 180) % 360 - direction),
                360 - np.abs((center_aspect + 180) % 360 - direction)
            )

            weight = (np.cos(np.radians(angle_diff/2)) + 1) / 2

            aspect_effect[y[valid], x[valid]] += (
                (padded_landuse[ny[valid]+1, nx[valid]+1] == RTS) * weight
            )

            aspect_effect[y[valid], x[valid]] += weight

    aspect_effect = np.where(aspect > 0,
                           aspect_effect / (aspect_effect + 1e-10),
                           1.0)
    aspect_effect[aspect < 0] = 1.0

    return aspect_effect

def ca_simulation_optimized(landuse, prob_2, aspect, target_areas, max_iterations=100,
                          seed=None, neighborhood_weight=0.8, use_gpu=True):
    if seed is not None:
        np.random.seed(seed)

    rows, cols = landuse.shape

    initial_rts = np.sum(landuse == RTS)
    print(f"initial RTS area: {initial_rts} (target area: {target_areas})")
    if initial_rts >= target_areas:
        raise ValueError("The initial RTS area has exceeded the target area！")

    Dk_2 = 1.0
    current_area_2 = initial_rts
    Gk_history = []
    Gk_current = target_areas - current_area_2
    Gk_history.append(Gk_current)

    aspect_effect = np.zeros((rows, cols))
    padded_landuse = np.pad(landuse, pad_width=1, mode='constant', constant_values=0)

    if use_gpu:
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(rows / threadsperblock[0])
        blockspergrid_y = math.ceil(cols / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

    for iteration in range(max_iterations):
        new_landuse = landuse.copy()

        if use_gpu:
            # GPU
            aspect_effect_gpu = cuda.to_device(np.zeros((rows, cols)))
            padded_landuse_gpu = cuda.to_device(padded_landuse)
            aspect_gpu = cuda.to_device(aspect)

            calculate_aspect_effect_gpu[blockspergrid, threadsperblock](
                aspect_gpu, padded_landuse_gpu, aspect_effect_gpu, rows, cols)
            aspect_effect = aspect_effect_gpu.copy_to_host()
        else:
            # CPU
            aspect_effect = calculate_aspect_effect_vectorized_cpu(landuse, aspect, rows, cols)

        neighborhood_effect = calculate_neighborhood_effect_vectorized(
            padded_landuse, landuse, neighborhood_weight)

        R_i = np.random.rand(rows, cols)
        RA_i = 1 + beta * ((-np.log(R_i))**alpha - 0.5)

        combined_prob = prob_2 * Dk_2 * neighborhood_effect * aspect_effect * RA_i
        combined_prob = np.clip(combined_prob, 0, 1)

        rand_matrix = np.random.rand(rows, cols)
        mask_1_to_2 = (landuse == NON_RTS) & (rand_matrix < combined_prob)
        new_landuse = np.where(mask_1_to_2, RTS, new_landuse)

        current_area_2 = np.sum(new_landuse == RTS)
        Gk_current = target_areas - current_area_2
        Gk_history.append(Gk_current)

        adjustment_type = "remain(initial iteration)"

        if len(Gk_history) >= 3:
            Gk_t_minus_2 = Gk_history[-3]
            Gk_t_minus_1 = Gk_history[-2]

            delta_prev = abs(Gk_t_minus_1) - abs(Gk_t_minus_2)
            delta_current = abs(Gk_current) - abs(Gk_t_minus_1)

            if delta_current <= delta_prev:
                Dk_2 = Dk_2
                adjustment_type = "Maintain (convergence)"
            elif delta_current > 0 and delta_prev > 0:
                adjustment_factor = abs(Gk_current) / (abs(Gk_t_minus_1) + 1e-10)
                Dk_2 = Dk_2 * adjustment_factor
                adjustment_type = f"Increase (development deficit exacerbated, coefficient{adjustment_factor:.3f}）"
            elif delta_current < 0 and delta_prev < 0:
                adjustment_factor = abs(Gk_t_minus_1) / (abs(Gk_current) + 1e-10)
                Dk_2 = Dk_2 * adjustment_factor
                adjustment_type = f"Reduce (overdevelopment exacerbates, coefficient{adjustment_factor:.3f}）"
            else:
                Dk_2 = Dk_2
                adjustment_type = "Keep (other cases)"

        Dk_2 = np.clip(Dk_2, 0.1, 10.0)

        if len(Gk_history) > 10:
            Gk_history.pop(0)

        # Output iteration information
        print(f"\niteration {iteration}:")
        print(f"Number of converted pixels: {np.sum(mask_1_to_2)}")
        print(f"Current RTS area: {current_area_2} (target: {target_areas})")
        print(f"Remaining targets:: {Gk_current}")
        print(f"Adaptive adjustment: {adjustment_type}")
        print(f"Adaptive coefficient Dk_2: {Dk_2:.4f}")
        print("-"*50)

        if current_area_2 >= target_areas:
            print(f"Reach target area (iteration {iteration})")
            break

        landuse = new_landuse
        padded_landuse = np.pad(landuse, pad_width=1, mode='constant', constant_values=0)

    return landuse


def calculate_metrics(landuse_initial, landuse_simulated, landuse_actual):
    metrics = {}

    mask = (landuse_initial == NON_RTS)
    actual_change = ((landuse_initial == NON_RTS) & (landuse_actual == RTS)).astype(int)
    simulated_change = ((landuse_initial == NON_RTS) & (landuse_simulated == RTS)).astype(int)

    TP = np.sum((simulated_change == 1) & (actual_change == 1))
    FP = np.sum((simulated_change == 1) & (actual_change == 0))
    FN = np.sum((simulated_change == 0) & (actual_change == 1))
    TN = np.sum((simulated_change == 0) & (actual_change == 0))
    total_pixels = TP + FP + FN + TN

    if total_pixels > 0:
        hit_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
        overall_accuracy = (TP + TN) / total_pixels
        p0 = overall_accuracy
        pe = (float(TP + FN) * float(TP + FP) + float(FP + TN) * float(FN + TN)) / (float(total_pixels) ** 2)
        kappa = (p0 - pe) / (1 - pe) if pe < 1 else 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = hit_rate
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fom = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        denominator = np.sqrt((TP + FP) * (TP + FN))
        lps = TP / denominator if denominator > 0 else 0

        metrics.update({
            'change_TP': TP, 'change_FP': FP, 'change_FN': FN, 'change_TN': TN,
            'hit_rate': hit_rate,
            'change_overall_accuracy': overall_accuracy,
            'change_kappa': kappa,
            'change_precision': precision,
            'change_recall': recall,
            'change_f1': f1,
            'change_fom': fom,
            'change_lps': lps
        })

    TP_all = np.sum((landuse_simulated == RTS) & (landuse_actual == RTS))
    FP_all = np.sum((landuse_simulated == RTS) & (landuse_actual == NON_RTS))
    FN_all = np.sum((landuse_simulated == NON_RTS) & (landuse_actual == RTS))
    TN_all = np.sum((landuse_simulated == NON_RTS) & (landuse_actual == NON_RTS))
    total_pixels_all = TP_all + FP_all + FN_all + TN_all

    if total_pixels_all > 0:
        overall_accuracy_all = (TP_all + TN_all) / total_pixels_all
        p0_all = overall_accuracy_all
        pe_all = (float(TP_all + FN_all) * float(TP_all + FP_all) + float(FP_all + TN_all) * float(FN_all + TN_all)) / (float(total_pixels_all) ** 2)
        kappa_all = (p0_all - pe_all) / (1 - pe_all) if pe_all < 1 else 0

        precision_all = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0
        recall_all = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0
        f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all) if (precision_all + recall_all) > 0 else 0

        metrics.update({
            'full_TP': TP_all, 'full_FP': FP_all, 'full_FN': FN_all, 'full_TN': TN_all,
            'full_overall_accuracy': overall_accuracy_all,
            'full_kappa': kappa_all,
            'full_precision': precision_all,
            'full_recall': recall_all,
            'full_f1': f1_all
        })

    return metrics


def main():
    try:
        cuda.detect()
        use_gpu = True
        print("CUDA GPU detected, GPU acceleration will be used")
    except:
        use_gpu = False
        print("No CUDA GPU detected, using CPU vectorized computations")

    landuse_2020, profile = read_raster(
        r'RTSEvo model driving data and results\Experiment 3\Prediction for 2021 & 2022\inputs\RTS2020.tif')
    prob_2, _ = read_raster(
        r'RTSEvo model driving data and results\Experiment 3\Prediction for 2021 & 2022\inputs\RF occur prob for 2021.tif')
    prob_2 = np.nan_to_num(prob_2, nan=0)
    prob_2 = np.clip(prob_2, 0, 1)

    landuse_2021_actual, _ = read_raster(
        r"RTSEvo model driving data and results\Experiment 3\Prediction for 2021 & 2022\validation data\RTS2021.tif")

    output_dir = r"RTSEvo model driving data and results\Experiment 3\Prediction for 2021 & 2022\outputs"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===== Running simulation, seed={seed}, neighborhood weight={neighborhood_weight} =====")

    try:
        landuse_2021_predict = ca_simulation_optimized(
            landuse_2020.copy(), prob_2, aspect,
            target_areas=103587.75, max_iterations=100,
            seed=seed, neighborhood_weight=neighborhood_weight, use_gpu=use_gpu
        )

        metrics = calculate_metrics(landuse_2020, landuse_2021_predict, landuse_2021_actual)

        print(f"- Kappa: {metrics['full_kappa']:.4f}")
        print(f"- FoM: {metrics['change_fom']:.4f}")


        output_path = os.path.join(output_dir, "2021RF-CA_Bestseed43.tif")
        write_raster(landuse_2021_predict, profile, output_path)
        print(f"result save: {output_path}")

    except ValueError as e:
        print(f"Simulation error: {str(e)}")


if __name__ == "__main__":
    main()
