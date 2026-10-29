import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")
trend = np.arange(60) * 2
seasonality = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = np.random.normal(0, 5, 60)
sales = 100 + trend + seasonality + noise

df = pd.DataFrame({"Date": dates, "Sales": sales})
df.set_index("Date", inplace=True)
print("TIME SERIES DATA:\n", df.head())

decomposition = seasonal_decompose(df["Sales"], model="additive", period=12)
result = adfuller(df["Sales"])
print("\nADF Statistic:", result[0], "| p-value:", result[1])

train_size = int(len(df) * 0.8)
train = df["Sales"][:train_size]
test = df["Sales"][train_size:]

model = ARIMA(train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print("\nForecast MAE:", round(mae, 2), "| RMSE:", round(rmse, 2))
