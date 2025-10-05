import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib as mpl

data = {
    "year": [2017, 2018, 2019, 2020, 2021, 2022],
    "area": [5782200, 6769800, 8134000, 9421300, 9951500, 10424700]
}
df = pd.DataFrame(data)
df['time_index'] = np.arange(len(df))

train_data = df[df['year'] <= 2020]
test_data = df[df['year'] >= 2021]

ts_train = pd.Series(train_data['area'].values, index=train_data['time_index'])

model = ExponentialSmoothing(
    ts_train,
    trend='add',
    initialization_method='estimated',
    damped_trend=True
).fit()

fitted_values = model.fittedvalues

forecast_years = 2
forecast = model.forecast(steps=forecast_years)
forecast_years_list = [2021, 2022]

mae_train = mean_absolute_error(ts_train, fitted_values)
rmse_train = np.sqrt(mean_squared_error(ts_train, fitted_values))
mape_train = np.mean(np.abs((ts_train - fitted_values) / ts_train)) * 100

actual_test = test_data['area'].values
mae_test = mean_absolute_error(actual_test, forecast)
rmse_test = np.sqrt(mean_squared_error(actual_test, forecast))
mape_test = np.mean(np.abs((actual_test - forecast) / actual_test)) * 100

ss_res_train = np.sum((ts_train - fitted_values)**2)
ss_tot_train = np.sum((ts_train - np.mean(ts_train))**2)
r_squared_train = 1 - (ss_res_train / ss_tot_train)

# 优化布局
plt.tight_layout()


plt.show()

# 输出详细结果
print("\n" + "="*60)
print("MODEL SUMMARY WITH VALIDATION")
print("="*60)

print("\nModel Parameters:")
print("-" * 40)
print(f"Smoothing Level (α): {model.params['smoothing_level']:.4f}")
print(f"Smoothing Trend (β): {model.params['smoothing_trend']:.4f}")
print(f"Initial Level: {model.params['initial_level']:,.0f}")
print(f"Initial Trend: {model.params['initial_trend']:,.0f}")

print("\nTraining Period Predictions (2017-2020):")
print("-" * 50)
for year, actual, fitted in zip(train_data['year'], ts_train, fitted_values):
    error = actual - fitted
    error_pct = (error / actual) * 100
    print(f"{year}: Actual = {actual:>8,.0f}, Predicted = {fitted:>8,.0f}, "
          f"Error = {error:>7,.0f} ({error_pct:+.1f}%)")

print(f"\nValidation Period (2021-2022):")
print("-" * 50)
for year, actual, predicted in zip(test_data['year'], actual_test, forecast):
    error = actual - predicted
    error_pct = (error / actual) * 100
    print(f"{year}: Actual = {actual:>8,.0f}, Predicted = {predicted:>8,.0f}, "
          f"Error = {error:>7,.0f} ({error_pct:+.1f}%)")

print(f"\nModel Evaluation Metrics:")
print("-" * 40)
print("Training Period (2017-2020):")
print(f"  MAE:  {mae_train:,.0f}")
print(f"  RMSE: {rmse_train:,.0f}")
print(f"  MAPE: {mape_train:.2f}%")
print(f"  R²:   {r_squared_train:.4f}")

print("\nValidation Period (2021-2022):")
print(f"  MAE:  {mae_test:,.0f}")
print(f"  RMSE: {rmse_test:,.0f}")
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
