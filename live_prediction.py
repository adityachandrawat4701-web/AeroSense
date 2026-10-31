
import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from physics_simulator import simulate_flight

# Load the trained model and its expected feature names
saved_data = joblib.load("flight_phase_model.joblib")
model = saved_data["model"]
features = saved_data["features"]

# Generate a new virtual flight
flight = simulate_flight(seed=123)

# Run for 60 simulated seconds, showing one reading per second
sample_interval = 10  # Simulator runs at 0.1-second intervals
duration_seconds = 60

results = []

print("========== AEROSENSE LIVE PREDICTION ==========")
print("Running a 60-second virtual flight...\n")

try:
    for i in range(0, len(flight), sample_interval):
        row = flight.iloc[i]

        sensor_reading = pd.DataFrame(
            [[row[feature] for feature in features]],
            columns=features,
        )

        prediction = model.predict(sensor_reading)[0]
        actual_phase = row["flight_phase"]

        results.append({
            "predicted": prediction,
            "actual": actual_phase,
        })

        print(
            f"Time: {row['timestamp_s']:6.1f} s | "
            f"Altitude: {row['altitude_m']:7.2f} m | "
            f"Vertical speed: {row['vertical_speed_mps']:6.2f} m/s | "
            f"Predicted: {prediction:8s} | "
            f"Actual: {actual_phase}"
        )

        time.sleep(1)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")

if results:
    results_df = pd.DataFrame(results)
    accuracy = accuracy_score(
        results_df["actual"],
        results_df["predicted"],
    )

    print("\n========== RUN SUMMARY ==========")
    print("Readings processed:", len(results_df))
    print(f"Prediction agreement: {accuracy:.2%}")
    print("\nPredicted phase counts:")
    print(results_df["predicted"].value_counts().to_string())

print("\nAeroSense run finished.")
