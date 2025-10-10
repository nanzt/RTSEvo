"""
@Description：
This study presents a novel dynamic evolution model for RTS that integrates machine learning with cellular automata.
The principal contribution of this study lies in breaking through the constraints of traditional static susceptibility
assessments. For the first time, it achieves regional-scale dynamic simulation and short-term forecasting of RTS.
The proposed modular framework is readily transferable to the modelling and prediction of other thermokarst hazards,
thereby providing a new tool for elucidating the cascading mechanisms of permafrost-degradation disasters.

@License：
Copyright (c) 2025 Xu Jiwei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from osgeo import gdal
import rasterio
import numpy as np
import rasterio
import numpy as np
from math import cos, sin, radians
import matplotlib.pyplot as plt
import os
import numba
from numba import cuda
import math
from tqdm import tqdm
import itertools
import time
from pyDOE import lhs

# Function to read raster data
def read_raster(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open file：{file_path}")
    band = dataset.GetRasterBand(1)
    return band.ReadAsArray()

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

aspect_min = np.min(aspect)
aspect_max = np.max(aspect)
print(f"aspect min: {aspect_min}")
print(f"aspect max: {aspect_max}")


# Set parameter range and sampling quantity
N_SAMPLES = 500
N_MIN, N_MAX = 3, 20
NW_MIN, NW_MAX = 0.01, 1.0
ALPHA_MIN, ALPHA_MAX = 0.01, 0.5
BETA_MIN, BETA_MAX = 0.01, 1.0
SEEDS = [43]

NON_RTS = 1
RTS = 2


def read_raster(path):
    with rasterio.open(path) as src:
        return np.array(src.read(1)), src.profile


def write_raster(array, profile, path):
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)


@cuda.jit
def calculate_Erosion_effect_gpu(aspect, padded_landuse, Erosion_effect, rows, cols, N):
    i, j = cuda.grid(2)
    if i < rows and j < cols:
        center_aspect = aspect[i, j]
        if center_aspect < 0:
            Erosion_effect[i, j] = 1.0
            return

        total_weight = 0.0
        weighted_count = 0.0
        pad_width = N // 2

        for di in range(-pad_width, pad_width + 1):
            for dj in range(-pad_width, pad_width + 1):
                if di == 0 and dj == 0: continue
                row, col = i + di, j + dj
                if 0 <= row < rows and 0 <= col < cols:
                    dy = i - row
                    dx = col - j
                    direction = math.degrees(math.atan2(dy, dx)) % 360
                    angle_diff = min(abs((center_aspect + 180) % 360 - direction),
                                     360 - abs((center_aspect + 180) % 360 - direction))
                    weight = math.cos(math.radians(angle_diff / 2))
                    weighted_count += (padded_landuse[row + pad_width, col + pad_width] == RTS) * weight
                    total_weight += weight

        Erosion_effect[i, j] = weighted_count / (total_weight + 1e-10)


def calculate_neighborhood_effect_vectorized(padded_landuse, landuse, neighborhood_weight, N=3):
    neighborhoods = np.lib.stride_tricks.sliding_window_view(padded_landuse, (N, N))
    count_2 = np.sum(neighborhoods == RTS, axis=(2, 3)) - (landuse == RTS).astype(int)
    valid_pixels = (N * N - 1)
    return (count_2 * neighborhood_weight) / valid_pixels


def calculate_Erosion_effect_vectorized_cpu(landuse, aspect, rows, cols, N=3):
    Erosion_effect = np.zeros((rows, cols))
    pad_width = N // 2
    padded_landuse = np.pad(landuse, pad_width=pad_width, mode='constant', constant_values=0)

    for di in range(-pad_width, pad_width + 1):
        for dj in range(-pad_width, pad_width + 1):
            if di == 0 and dj == 0: continue

            y_coords = np.arange(rows) + di
            x_coords = np.arange(cols) + dj
            valid_y = (y_coords >= 0) & (y_coords < rows)
            valid_x = (x_coords >= 0) & (x_coords < cols)

            for i in range(rows):
                if not valid_y[i]: continue
                for j in range(cols):
                    if not valid_x[j]: continue

                    dy = i - (i + di)
                    dx = (j + dj) - j
                    direction = np.degrees(np.arctan2(dy, dx)) % 360

                    center_aspect = aspect[i, j]
                    angle_diff = min(
                        abs((center_aspect + 180) % 360 - direction),
                        360 - abs((center_aspect + 180) % 360 - direction)
                    )

                    weight = (np.cos(np.radians(angle_diff / 2)) + 1) / 2

                    if padded_landuse[i + di + pad_width, j + dj + pad_width] == RTS:
                        Erosion_effect[i, j] += weight

    Erosion_effect = np.where(aspect > 0, Erosion_effect, 1.0)
    Erosion_effect[aspect < 0] = 1.0

    return Erosion_effect

def ca_simulation_optimized(landuse, prob_2, aspect, target_areas, max_iterations=100,
                            seed=None, neighborhood_weight=0.8, alpha=0.05, beta=0.3,
                            N=3, use_gpu=True):
    if seed is not None:
        np.random.seed(seed)

    rows, cols = landuse.shape
    pad_width = N // 2

    # 初始检查
    initial_rts = np.sum(landuse == RTS)
    if initial_rts >= target_areas:
        raise ValueError("The initial RTS area has exceeded the target area!！")

    Dk_2 = 1.0
    current_area_2 = initial_rts
    Gk_history = [target_areas - current_area_2]

    Erosion_effect = np.zeros((rows, cols))
    padded_landuse = np.pad(landuse, pad_width=pad_width, mode='constant', constant_values=0)

    if use_gpu:
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(rows / threadsperblock[0])
        blockspergrid_y = math.ceil(cols / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

    for iteration in range(max_iterations):
        new_landuse = landuse.copy()

        if use_gpu:
            Erosion_effect_gpu = cuda.to_device(np.zeros((rows, cols)))
            padded_landuse_gpu = cuda.to_device(padded_landuse)
            aspect_gpu = cuda.to_device(aspect)

            calculate_Erosion_effect_gpu[blockspergrid, threadsperblock](
                aspect_gpu, padded_landuse_gpu, Erosion_effect_gpu, rows, cols, N)
            Erosion_effect = Erosion_effect_gpu.copy_to_host()
        else:
            Erosion_effect = calculate_Erosion_effect_vectorized_cpu(landuse, aspect, rows, cols, N)

        neighborhood_effect = calculate_neighborhood_effect_vectorized(
            padded_landuse, landuse, neighborhood_weight, N)

        R_i = np.random.rand(rows, cols)
        RA_i = 1 + beta * ((-np.log(R_i)) ** alpha - 0.5)

        combined_prob = prob_2 * Dk_2 * neighborhood_effect * Erosion_effect * RA_i
        combined_prob = np.clip(combined_prob, 0, 1)

        rand_matrix = np.random.rand(rows, cols)
        mask_1_to_2 = (landuse == NON_RTS) & (rand_matrix < combined_prob)
        new_landuse = np.where(mask_1_to_2, RTS, new_landuse)

        current_area_2 = np.sum(new_landuse == RTS)
        Gk_current = target_areas - current_area_2
        Gk_history.append(Gk_current)

        if len(Gk_history) >= 3:
            Gk_t_minus_2 = Gk_history[-3]
            Gk_t_minus_1 = Gk_history[-2]

            delta_prev = abs(Gk_t_minus_1) - abs(Gk_t_minus_2)
            delta_current = abs(Gk_current) - abs(Gk_t_minus_1)

            if delta_current <= delta_prev:
                Dk_2 = Dk_2
            elif delta_current > 0 and delta_prev > 0:
                adjustment_factor = abs(Gk_current) / (abs(Gk_t_minus_1) + 1e-10)
                Dk_2 = Dk_2 * adjustment_factor
            elif delta_current < 0 and delta_prev < 0:
                adjustment_factor = abs(Gk_t_minus_1) / (abs(Gk_current) + 1e-10)
                Dk_2 = Dk_2 * adjustment_factor

        Dk_2 = np.clip(Dk_2, 0.1, 10.0)

        if len(Gk_history) > 10:
            Gk_history.pop(0)

        if current_area_2 >= target_areas:
            break

        landuse = new_landuse
        padded_landuse = np.pad(landuse, pad_width=pad_width, mode='constant', constant_values=0)

    return landuse


def calculate_fom(landuse_initial, landuse_simulated, landuse_actual):
    mask = (landuse_initial == NON_RTS)
    actual_change = ((landuse_initial == NON_RTS) & (landuse_actual == RTS)).astype(int)
    simulated_change = ((landuse_initial == NON_RTS) & (landuse_simulated == RTS)).astype(int)

    TP = np.sum((simulated_change == 1) & (actual_change == 1))
    FP = np.sum((simulated_change == 1) & (actual_change == 0))
    FN = np.sum((simulated_change == 0) & (actual_change == 1))

    denominator = TP + FP + FN
    if denominator > 0:
        return TP / denominator
    else:
        return 0.0


def generate_latin_hypercube_samples(n_samples):
    # "Generate a 4-dimensional Latin hypercube sampling"（N, neighborhood_weight, alpha, beta）
    lhd = lhs(4, samples=n_samples, criterion='maximin')
    samples = []
    for sample in lhd:
        N_val = int(np.round(N_MIN + sample[0] * (N_MAX - N_MIN)))
        if N_val % 2 == 0:
            N_val = max(N_MIN, min(N_val - 1, N_MAX))

        # neighborhood_weight: 0.01-1.0
        nw_val = NW_MIN + sample[1] * (NW_MAX - NW_MIN)

        # alpha: 0.01-0.5
        alpha_val = ALPHA_MIN + sample[2] * (ALPHA_MAX - ALPHA_MIN)

        # beta: 0.01-1.0
        beta_val = BETA_MIN + sample[3] * (BETA_MAX - BETA_MIN)

        samples.append((N_val, nw_val, alpha_val, beta_val))

    return samples


def parameter_tuning():
    """Parameter Calibration Main Function"""
    landuse_2019, profile = read_raster(
        r'RTSEvo model driving data and results\Experiment 3\Calibration using 2020\inputs\RTS2019.tif')
    prob_2, _ = read_raster(
        r'RTSEvo model driving data and results\Experiment 3\Calibration using 2020\inputs\LR occur prob for 2020.tif')
    prob_2 = np.nan_to_num(prob_2, nan=0)
    prob_2 = np.clip(prob_2, 0, 1)

    landuse_2020_actual, _ = read_raster(
        r"RTSEvo model driving data and results\Experiment 3\Calibration using 2020\validation data\RTS2020.tif")

    try:
        cuda.detect()
        use_gpu = True
        print("CUDA GPU detected, will use GPU acceleration.")
    except:
        use_gpu = False
        print("No CUDA GPU detected, will use CPU for computation.")

    param_samples = generate_latin_hypercube_samples(N_SAMPLES)

    param_combinations = []
    for params in param_samples:
        for seed in SEEDS:
            param_combinations.append((*params, seed))

    best_fom = 0
    best_params = {}
    best_landuse = None
    results = []

    print(f"Starting parameter calibration, there are a total of {len(param_combinations)} parameter combinations...")

    for i, (N, nw, alpha, beta, seed) in enumerate(tqdm(param_combinations, desc="Parameter Calibration Progress")):
        try:
            start_time = time.time()

            landuse_2020_predict = ca_simulation_optimized(
                landuse_2019.copy(), prob_2, aspect,
                target_areas=94213, max_iterations=100,
                seed=seed, neighborhood_weight=nw,
                alpha=alpha, beta=beta, N=N, use_gpu=use_gpu
            )

            fom = calculate_fom(landuse_2019, landuse_2020_predict, landuse_2020_actual)
            execution_time = time.time() - start_time

            results.append({
                'N': N,
                'neighborhood_weight': nw,
                'alpha': alpha,
                'beta': beta,
                'seed': seed,
                'fom': fom,
                'time': execution_time
            })

            if fom > best_fom:
                best_fom = fom
                best_params = {
                    'N': N,
                    'neighborhood_weight': nw,
                    'alpha': alpha,
                    'beta': beta,
                    'seed': seed,
                    'fom': fom
                }
                best_landuse = landuse_2020_predict.copy()
                print(f"\n*** The new best FOM: {fom:.4f} ***")
                print(f"Parameter: N={N}, nw={nw:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, seed={seed}")

        except Exception as e:
            print(f"\nParameter combination error: {str(e)}")
            continue

    # Save the best results
    if best_landuse is not None:
        output_dir = r"RTSEvo model driving data and results\Experiment 3\Calibration using 2020\outputs"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "LR2020_optimized_best_result.tif")
        write_raster(best_landuse, profile, output_path)

        results_path = os.path.join(output_dir, "LR2020_parameter_tuning_results_lhs.csv")
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("Best parameter combination::")
    print("=" * 60)
    for key, value in best_params.items():
        print(f"{key}: {value}")

    # Display the top 10 results sorted by FOM
    print(f"\nthe top 10 best parameter combinations:")
    sorted_results = sorted(results, key=lambda x: x['fom'], reverse=True)[:10]
    for i, result in enumerate(sorted_results):
        print(f"{i + 1}. FOM={result['fom']:.4f}, N={result['N']}, nw={result['neighborhood_weight']:.3f}, "
              f"alpha={result['alpha']:.3f}, beta={result['beta']:.3f}, seed={result['seed']}")

    return best_params, best_landuse, results


if __name__ == "__main__":
    best_params, best_landuse, all_results = parameter_tuning()
