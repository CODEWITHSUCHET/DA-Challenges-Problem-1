# Weather Forecasting Pipeline

## Project Overview
The **Weather Forecasting Pipeline** is an automated system designed to predict the temperature for the next hour based on weather data. The pipeline generates synthetic weather data, trains a machine learning model (Random Forest Regressor), and forecasts future temperatures. This project demonstrates how to collect, process, and predict time-series weather data, with potential applications in weather services, IoT systems, and predictive analytics.

### Why It Matters
Weather prediction plays a crucial role in multiple industries such as agriculture, logistics, and event planning. By automating temperature predictions, this system can be integrated into real-time applications, helping businesses and individuals plan better based on the forecast.

## Key Features & Technologies

- **Synthetic Data Generation**: The system can generate synthetic weather data for training the model.
- **Random Forest Regressor**: Uses the Random Forest algorithm to predict the next hour’s temperature based on multiple weather features.
- **Scheduled Forecasting**: The system runs every minute to update the dataset, train the model, and forecast the next temperature.
- **Model Persistence**: The trained model is saved and reused, reducing the need for retraining on every run.
- **Logging**: The application logs important events, such as data updates, model training status, and forecast results.
- **Data Storage**: Forecast results and weather data are stored in CSV files, ensuring persistence across runs.

### Key Technologies Used:
- **Python**: Main programming language used to implement the project.
- **pandas**: Data manipulation and handling.
- **NumPy**: Used for numerical operations, such as generating synthetic data.
- **Scikit-learn**: For implementing machine learning algorithms (Random Forest Regressor).
- **Joblib**: For saving and loading the trained machine learning model.
- **Schedule**: For running the forecasting pipeline on a regular basis.
- **Logging**: To track application behavior and outcomes.
- **CSV Files**: For storing the weather data and forecast results.

## Setup Instructions

Follow these simple steps to run the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/weather-forecasting-pipeline.git
   cd weather-forecasting-pipeline
