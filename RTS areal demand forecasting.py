"""
@Description: RTSEvo v1.0 - Module 1: RTS Areal Demand Forecasting

    This module implements the first core component of the RTSEvo framework: forecasting the
    total regional RTS area for target years. It establishes a top-down, macro-scale constraint
    that governs the overall magnitude of RTS expansion in the spatial allocation module.

@Date: Feb 5，2026

@Website: https://permalab.science
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import rasterio
import glob
import os


def calculate_area_from_tif(tif_file, target_value=2):
    try:
        with rasterio.open(tif_file) as src:
            data = src.read(1)

            transform = src.transform
            pixel_area = abs(transform[0] * transform[4])
            target_pixels = np.sum(data == target_value)
            total_area = target_pixels * pixel_area
            return total_area

    except Exception as e:
        print(f"Error processing file {tif_file} : {e}")
        return 0


def process_tif_files_from_multiple_directories(train_directories, validation_directories, file_pattern="RTS*.tif"):
    all_files = []
    directories_info = []

    for directory in train_directories:
        search_path = os.path.join(directory, file_pattern)
        tif_files = glob.glob(search_path)
        if not tif_files:
            tif_files = glob.glob(os.path.join(directory, "**", file_pattern), recursive=True)

        print(f"Found {len(tif_files)} training files in {directory}:")
        for file in tif_files:
            print(f"  - {file}")
            all_files.append(file)
            directories_info.append(('train', file))

    for directory in validation_directories:
        search_path = os.path.join(directory, file_pattern)
        tif_files = glob.glob(search_path)
        if not tif_files:
            tif_files = glob.glob(os.path.join(directory, "**", file_pattern), recursive=True)

        print(f"Found {len(tif_files)} validation files in {directory}:")
        for file in tif_files:
            print(f"  - {file}")
            all_files.append(file)
            directories_info.append(('validation', file))

    if not all_files:
        raise FileNotFoundError(
            f"No files matching {file_pattern} were found. Please check the file paths and pattern.")

    all_files.sort()

    years = []
    areas = []
    file_names = []
    data_types = []

    for data_type, tif_file in directories_info:
        filename = os.path.basename(tif_file)
        file_names.append(filename)

        year_str = None
        possible_patterns = [
            filename.replace('RTS', '').replace('.tif', ''),
            filename.replace('RTS', '').replace('.TIF', ''),
            filename.replace('rts', '').replace('.tif', ''),
        ]

        for pattern in possible_patterns:
            try:
                year = int(pattern)
                year_str = pattern
                break
            except ValueError:
                continue

        if year_str is None:
            print(f"Warning: Unable to extract year from filename {filename}, skipping this file")
            continue

        try:
            year = int(year_str)
            area = calculate_area_from_tif(tif_file, target_value=2)

            years.append(year)
            areas.append(area)
            data_types.append(data_type)

            print(f"Processing file: {filename}, Year: {year}, Type: {data_type}, Area: {area:,.0f} square meters")

        except Exception as e:
            print(f"Error processing file {filename} : {e}")
            continue

    if len(years) == 0:
        raise ValueError("No TIFF files were successfully processed. Please check the file format and content.")

    df = pd.DataFrame({
        'year': years,
        'area': areas,
        'filename': file_names,
        'data_type': data_types
    })

    df = df.sort_values('year').reset_index(drop=True)

    return df


try:
    train_directories = [
        r"RTSEvo model driving data and results\Experiment 1\inputs"
    ]

    validation_directories = [
        r"RTSEvo model driving data and results\Experiment 1\validation data"
    ]

    df = process_tif_files_from_multiple_directories(
        train_directories=train_directories,
        validation_directories=validation_directories,
        file_pattern="RTS*.tif"
    )

    print(f"\nSuccessfully processed {len(df)} files:")
    print(df[['year', 'area', 'filename', 'data_type']])


    train_data = df[df['data_type'] == 'train'].copy()
    test_data = df[df['data_type'] == 'validation'].copy()


    expected_train_years = [2016, 2017, 2018, 2019, 2020]
    expected_test_years = [2021, 2022]

    missing_train_years = [year for year in expected_train_years if year not in train_data['year'].values]
    missing_test_years = [year for year in expected_test_years if year not in test_data['year'].values]

    if missing_train_years:
        print(f"\nWarning: Training data for the following years is missing: {missing_train_years}")
    if missing_test_years:
        print(f"Warning: Validation data for the following years is missing: {missing_test_years}")

    train_data['time_index'] = np.arange(len(train_data))
    if len(test_data) > 0:
        test_data['time_index'] = np.arange(len(train_data), len(train_data) + len(test_data))

    if len(train_data) == 0:
        raise ValueError("Training data is empty, please check the data paths.")

    print(f"\nTraining data years: {list(train_data['year'])}")
    if len(test_data) > 0:
        print(f"Validation data years: {list(test_data['year'])}")

    ts_train = pd.Series(train_data['area'].values, index=train_data['time_index'])

    model = ExponentialSmoothing(
        ts_train,
        trend='add',
        initialization_method='estimated',
        damped_trend=True
    ).fit()

    fitted_values = model.fittedvalues


    forecast_years = len(test_data)
    forecast = model.forecast(steps=forecast_years)

    mae_train = mean_absolute_error(ts_train, fitted_values)
    rmse_train = np.sqrt(mean_squared_error(ts_train, fitted_values))
    mape_train = np.mean(np.abs((ts_train - fitted_values) / ts_train)) * 100

    if len(test_data) > 0:
        actual_test = test_data['area'].values
        mae_test = mean_absolute_error(actual_test, forecast)
        rmse_test = np.sqrt(mean_squared_error(actual_test, forecast))
        mape_test = np.mean(np.abs((actual_test - forecast) / actual_test)) * 100
    else:
        mae_test = rmse_test = mape_test = np.nan

    ss_res_train = np.sum((ts_train - fitted_values) ** 2)
    ss_tot_train = np.sum((ts_train - np.mean(ts_train)) ** 2)
    r_squared_train = 1 - (ss_res_train / ss_tot_train)

    print("\n" + "=" * 60)
    print("MODEL SUMMARY WITH VALIDATION")
    print("=" * 60)

    print("\nModel Parameters:")
    print("-" * 40)
    print(f"Smoothing Level (α): {model.params['smoothing_level']:.4f}")
    print(f"Smoothing Trend (β): {model.params['smoothing_trend']:.4f}")
    print(f"Initial Level: {model.params['initial_level']:,.0f}")
    print(f"Initial Trend: {model.params['initial_trend']:,.0f}")

    print("\nTraining Period Predictions:")
    print("-" * 50)
    for year, actual, fitted in zip(train_data['year'], ts_train, fitted_values):
        error = actual - fitted
        error_pct = (error / actual) * 100
        print(f"{year}: Actual = {actual:>8,.0f}, Predicted = {fitted:>8,.0f}, "
              f"Error = {error:>7,.0f} ({error_pct:+.1f}%)")

    if len(test_data) > 0:
        print(f"\nValidation Period:")
        print("-" * 50)
        for year, actual, predicted in zip(test_data['year'], actual_test, forecast):
            error = actual - predicted
            error_pct = (error / actual) * 100
            print(f"{year}: Actual = {actual:>8,.0f}, Predicted = {predicted:>8,.0f}, "
                  f"Error = {error:>7,.0f} ({error_pct:+.1f}%)")

    print(f"\nModel Evaluation Metrics:")
    print("-" * 40)
    print(f"  R²:   {r_squared_train:.4f}")

    if len(test_data) > 0:
        print(f"  MAPE: {mape_test:.2f}%")

    print(f"\nModel Performance Assessment:")
    print("-" * 40)


    def assess_performance(r_squared):
        if r_squared > 0.9:
            return "Excellent"
        elif r_squared > 0.7:
            return "Good"
        elif r_squared > 0.5:
            return "Moderate"
        else:
            return "Poor"


    print(f"Training R²: {r_squared_train:.4f} ({assess_performance(r_squared_train)})")


    print("\n" + "=" * 60)
    print("SAVING RESULTS TO EXCEL FILE")
    print("=" * 60)


    results_df = pd.DataFrame()


    train_results = pd.DataFrame({
        'Year': train_data['year'],
        'Actual_Area': train_data['area'],
        'Fitted_Area': fitted_values,
        'Error': train_data['area'] - fitted_values,
        'Error_Percentage': ((train_data['area'] - fitted_values) / train_data['area']) * 100,
        'Data_Type': 'Training'
    })


    if len(test_data) > 0:
        test_results = pd.DataFrame({
            'Year': test_data['year'],
            'Actual_Area': test_data['area'],
            'Forecast_Area': forecast,
            'Error': test_data['area'] - forecast,
            'Error_Percentage': ((test_data['area'] - forecast) / test_data['area']) * 100,
            'Data_Type': 'Validation'
        })
    else:
        test_results = pd.DataFrame()

    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'RMSE', 'MAPE'],
        'Training_Value': [r_squared_train, mae_train, rmse_train, mape_train],
        'Validation_Value': [None, mae_test if len(test_data) > 0 else None,
                             rmse_test if len(test_data) > 0 else None,
                             mape_test if len(test_data) > 0 else None]
    })

    params_df = pd.DataFrame({
        'Parameter': ['Smoothing Level (α)', 'Smoothing Trend (β)', 'Initial Level', 'Initial Trend'],
        'Value': [model.params['smoothing_level'], model.params['smoothing_trend'],
                  model.params['initial_level'], model.params['initial_trend']]
    })

    output_filename = r'RTSEvo model driving data and results\Experiment 1\outputs\RTS_Area_Analysis_Results.xlsx'

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:

        df[['year', 'area', 'filename', 'data_type']].to_excel(writer, sheet_name='Original_Data', index=False)

        train_results.to_excel(writer, sheet_name='Training_Results', index=False)

        if len(test_results) > 0:
            test_results.to_excel(writer, sheet_name='Validation_Results', index=False)

        metrics_df.to_excel(writer, sheet_name='Evaluation_Metrics', index=False)

        params_df.to_excel(writer, sheet_name='Model_Parameters', index=False)

    print(f"\nResults successfully saved to: {output_filename}")
    print("File contains the following sheets:")
    print("  - Original_Data: Original area data from TIFF files")
    print("  - Training_Results: Training period actual vs fitted values")
    print("  - Validation_Results: Validation period actual vs forecast values")
    print("  - Evaluation_Metrics: Model performance metrics")
    print("  - Model_Parameters: Model parameters")

except Exception as e:
    print(f"The program encountered an error during execution: {e}")
