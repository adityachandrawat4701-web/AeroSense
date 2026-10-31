
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

all_flights = []
dt = 1.0

for flight_id in range(100):
    # Every flight gets slightly different characteristics
    climb_speed = rng.uniform(5, 10)
    cruise_time = rng.integers(40, 100)
    descent_speed = rng.uniform(3, 7)
    ground_time = rng.integers(10, 25)

    climb_time = rng.integers(40, 80)
    landing_time = rng.integers(10, 20)

    phases = (
        ["ground"] * ground_time
        + ["climb"] * climb_time
        + ["cruise"] * cruise_time
        + ["descent"] * int(climb_time * climb_speed / descent_speed)
        + ["landing"] * landing_time
        + ["ground"] * ground_time
    )

    altitude = 0.0
    previous_altitude = 0.0
    rows = []

    for t, phase in enumerate(phases):
        # A simplified vertical-speed profile for each phase
        if phase == "ground":
            vertical_speed = 0.0
            pitch = rng.normal(0, 0.3)

        elif phase == "climb":
            vertical_speed = climb_speed
            pitch = rng.normal(8, 1.5)

        elif phase == "cruise":
            vertical_speed = rng.normal(0, 0.15)
            pitch = rng.normal(1, 0.8)

        elif phase == "descent":
            vertical_speed = -descent_speed
            pitch = rng.normal(-3, 1.2)

        else:  # landing
            vertical_speed = -rng.uniform(0.2, 1.0)
            pitch = rng.normal(2, 1.0)

        # Integrate altitude and prevent below-ground values
        altitude = max(0.0, altitude + vertical_speed * dt)

        # Sensor-like measurements with simulated noise
        measured_altitude = max(0.0, altitude + rng.normal(0, 1.5))
        measured_vertical_speed = (
            (altitude - previous_altitude) / dt + rng.normal(0, 0.2)
        )

        accel_x = rng.normal(0, 0.15)
        accel_y = rng.normal(0, 0.15)
        accel_z = rng.normal(9.81, 0.2)
        roll = rng.normal(0, 2.0)

        # Store one sample per simulated second
        rows.append({
            "flight_id": flight_id,
            "time_s": t,
            "altitude_m": measured_altitude,
            "vertical_speed_mps": measured_vertical_speed,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "flight_phase": phase,
        })

        previous_altitude = altitude

    all_flights.extend(rows)

df = pd.DataFrame(all_flights)
df.to_csv("multi_flight_data.csv", index=False)

print("Dataset generated successfully!")
print(f"Flights: {df['flight_id'].nunique()}")
print(f"Total samples: {len(df)}")
print("\nSamples per phase:")
print(df["flight_phase"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("\nSaved to multi_flight_data.csv")
