import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import statsmodels.api as sm

#df = pd.read_excel(r"C:\Users\pravi\Documents\nikhil_project\Book1.xlsx")
#df

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Custom transformer for data processing
class WeeklyMeanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert 'Date' column to datetime index
       # X['Date'] = pd.to_datetime(X['Date'])
        X.set_index('Date', inplace=True)

        # Keep only 'Date' and 'Closing' columns
        X = X[['Closing Volume']]

        # Resample to weekly frequency and take mean
        X_weekly = X.resample('W').mean()

        return X_weekly

# Define the pipeline
pipeline0 = Pipeline([
    ('data_processing', WeeklyMeanTransformer())
])



import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Custom transformer for STL decomposition and forecasting
class STLForecaster(BaseEstimator, TransformerMixin):
    def __init__(self, period=52, robust=True, seasonal=11):
        self.period = period
        self.robust = robust
        self.seasonal = seasonal

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fit the STL-W model to the training data
        stl = STL(X, period=self.period, robust=self.robust, seasonal=self.seasonal)
        res = stl.fit()

        # Get the seasonal and trend components
        seasonal = res.seasonal.values
        trend = res.trend.values

        result = trend + seasonal

        # Generate the forecasts for the next 52 weeks
        forecast_index = pd.date_range(start=X.index[-1], periods=52, freq='W')
        forecast_data = seasonal[:52] + trend[:52]

        # Create a new DataFrame for the forecasts
        forecast = pd.DataFrame({'Closing Volume': forecast_data}, index=forecast_index)

        return forecast

# Example usage:
# Assuming 'dff' is the original DataFrame containing the 'Closing Volume' column

# Define the pipeline
pipeline1 = Pipeline([
    ('stl_forecaster', STLForecaster(period=52, robust=True, seasonal=11))
])



import joblib
from sklearn.pipeline import Pipeline

# Assume `preprocessing_pipeline` is your preprocessing pipeline object and `model` is your trained ML model object.

pipeline_stockprediction = Pipeline(steps = [
    ('pipeline0', pipeline0),
    ('pipeline1', pipeline1),
])


