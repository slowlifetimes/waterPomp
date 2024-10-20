from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# Model Machine Learning Sederhana
model = LinearRegression()

# Data pelatihan awal (dummy data)
# [waterFlowRate, temp, humidity, numResidents, specialEvent]
training_data = np.array([
    [15, 28, 65, 4, 0],
    [10, 25, 70, 3, 1],
    [12, 30, 60, 5, 0]
])

# Target (konsumsi air total dalam liter)
target_data = np.array([200, 180, 220])

# Melatih model
model.fit(training_data, target_data)

# Struktur data yang diharapkan dari ESP32
class WaterData(BaseModel):
    waterFlowRate: float
    temp: float
    humidity: float
    numResidents: int
    specialEvent: int

@app.post("/")
def predict_water_consumption(data: WaterData):
    # Siapkan data untuk prediksi
    input_data = np.array([[data.waterFlowRate, data.temp, data.humidity, data.numResidents, data.specialEvent]])

    # Lakukan prediksi
    predicted_consumption = model.predict(input_data)

    # Format respon dalam JSON
    response = {
        'predicted_consumption': predicted_consumption[0],
        'percent_change_from_previous_day': 5.5  # Misalnya perubahan konsumsi dari hari sebelumnya
    }

    return response
