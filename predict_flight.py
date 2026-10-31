
import joblib
import pandas as pd

# Load our trained model.
saved_data = joblib.load("flight_phase_model.joblib")
model = saved_data["model"]
features = saved_data["features"]
# Enter one example of sensor readings.
reading = {
    "altitude_m": 500.0,
    "vertical_speed_mps": 8.0,
    "accel_x": 0.02,
    "accel_y": -0.01,
    "accel_z": 9.81,
    "pitch_deg": 8.0,
    "roll_deg": 0.5,
}

# Keep the feature order used during training.
features = [
    "altitude_m",
    "vertical_speed_mps",
    "accel_x",
    "accel_y",
    "accel_z",
    "pitch_deg",
    "roll_deg",
]

input_data = pd.DataFrame([reading], columns=features)

prediction = model.predict(input_data)[0]

print("Sensor readings:", reading)
print("Predicted flight phase:", prediction)
