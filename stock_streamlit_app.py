import streamlit as st
import pandas as pd
import joblib

from stock_pipelines import WeeklyMeanTransformer
from stock_pipelines import STLForecaster


# Set up the layout of the app.
st.set_page_config(
    page_title='Stocks Price Predictor App',
    page_icon=':chart_with_upwards_trend:',
    layout='wide'  # Display the app in wide mode
)

# Set app title and header
st.title('Stocks Price Predictor App')
st.write("-- By Nikhil")
st.header('Choose a file and click the "Upload" button to get started.')


# Create a file uploader and save the uploaded file to a pandas DataFrame.
uploaded_file = st.file_uploader('Upload a file', type=['csv', 'xlsx', 'txt'])
if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f'Error: {e}')
    else:
        st.success('File uploaded successfully!')

        # Show a preview of the uploaded data.
        st.subheader('Preview of uploaded data')
        st.dataframe(data.head())

        # Create a button to show the next year's forecast.
        if st.button('Show next year\'s forecast'):
            try:
                # Load the pipeline object from the file.
                model_pipeline = joblib.load('stockprediction_pipeline.joblib')

                # Preprocess the input data and make a prediction.
                forecast = model_pipeline.fit_transform(data)
            except Exception as e:
                st.error(f'Error: {e}')
            else:
                # Show the forecast.
                st.subheader('Forecast for next year in weekly basis')
                st.dataframe(forecast)


