import pandas as pd
import numpy as np
import os
import joblib
import schedule
import time
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Constants
DATA_FILE = "weather_data.csv"
MODEL_FILE = "weather_model.pkl"
FORECAST_FILE = "latest_forecast.csv"
FEATURES = [
    "temperature", "humidity", "wind_speed", "wind_direction",
    "pressure", "precipitation", "cloud_coverage"
]

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# -----------------------------
# Step 1: Generate / Fetch Data
# -----------------------------
def generate_synthetic_weather_data(hours=1):
    now = datetime.utcnow()
    timestamps = [now - timedelta(hours=i) for i in range(hours)]
    data = {
        "date_time": sorted(timestamps),
        "temperature": np.random.uniform(15, 35, hours),
        "humidity": np.random.uniform(30, 90, hours),
        "wind_speed": np.random.uniform(1, 10, hours),
        "wind_direction": np.random.uniform(0, 360, hours),
        "pressure": np.random.uniform(980, 1030, hours),
        "precipitation": np.random.uniform(0, 10, hours),
        "cloud_coverage": np.random.uniform(10, 100, hours),
        "weather_condition": np.random.choice(["clear", "cloudy", "rainy", "snowy"], hours)
    }
    return pd.DataFrame(data)

def update_dataset(new_data: pd.DataFrame):
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE, parse_dates=["date_time"])
        full = pd.concat([existing, new_data]).drop_duplicates(subset=["date_time"])
    else:
        full = new_data
    full.sort_values("date_time", inplace=True)
    full.to_csv(DATA_FILE, index=False)
    return full

# -----------------------------
# Step 2: Train or Load Model
# -----------------------------
def train_model(df: pd.DataFrame):
    df = df.dropna()
    df["forecast_temperature"] = df["temperature"].shift(-1)
    df = df.dropna()

    if len(df) < 10:
        logging.warning("⚠️ Not enough data to train the model. Minimum 10 samples required.")
        return None

    X = df[FEATURES]
    y = df["forecast_temperature"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"✅ Model trained with MSE: {mse:.2f}")

    joblib.dump(model, MODEL_FILE)
    return model

def load_or_train_model(df: pd.DataFrame):
    if os.path.exists(MODEL_FILE):
        logging.info("📦 Loading existing model...")
        return joblib.load(MODEL_FILE)
    else:
        logging.info("🧠 Training new model...")
        return train_model(df)

# -----------------------------
# Step 3: Make Forecast
# -----------------------------
def forecast_next_hour(model, latest_data: pd.DataFrame):
    if model is None:
        logging.warning("❌ Forecast skipped: No model available.")
        return None

    # Ensure input is a DataFrame with correct feature names
    input_row = latest_data[FEATURES].iloc[-1].values.reshape(1, -1)
    input_df = pd.DataFrame(input_row, columns=FEATURES)  # Convert to DataFrame with feature names

    predicted_temp = model.predict(input_df)[0]

    forecast_time = latest_data["date_time"].iloc[-1] + timedelta(hours=1)
    forecast = pd.DataFrame([{
        "date_time": forecast_time,
        "predicted_temperature": predicted_temp
    }])

    forecast.to_csv(FORECAST_FILE, index=False)
    logging.info(f"🌤️ Forecast saved for {forecast_time.strftime('%Y-%m-%d %H:%M:%S')} - Temp: {predicted_temp:.2f}°C")
    return forecast

# -----------------------------
# Step 4: Full Pipeline Runner
# -----------------------------
def run_forecast_pipeline():
    logging.info("🔁 Running weather forecast pipeline...")

    # Check if we need to bootstrap with more data
    if not os.path.exists(DATA_FILE):
        logging.info("🆕 No existing data found. Bootstrapping with 24 hours of synthetic data...")
        new_data = generate_synthetic_weather_data(hours=24)
    else:
        new_data = generate_synthetic_weather_data(hours=1)

    # Update dataset
    full_data = update_dataset(new_data)

    # Train or load model
    model = load_or_train_model(full_data)
    if model is None:
        logging.warning("❌ Model training skipped due to insufficient data.")
        return

    # Forecast
    forecast_next_hour(model, full_data)
    logging.info("✅ Pipeline complete.\n")

# -----------------------------
# Step 5: Schedule Regular Runs
# -----------------------------
schedule.every(1).minutes.do(run_forecast_pipeline)

if __name__ == "__main__":
    logging.info("🚀 Weather Forecast System Started...")
    run_forecast_pipeline()  # Run once at start
    while True:
        schedule.run_pending()
        time.sleep(1)
