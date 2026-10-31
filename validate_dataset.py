
import pandas as pd

# Load our newly generated dataset
df = pd.read_csv("multi_flight_data.csv")

print("========== DATASET VALIDATION ==========")

# 1. Dataset size
print("\n1. Dataset size")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Unique flights:", df["flight_id"].nunique())

# 2. Missing values
print("\n2. Missing values per column")
print(df.isnull().sum())

# 3. Samples belonging to each phase
print("\n3. Samples per flight phase")
print(df["flight_phase"].value_counts())

# 4. Number of phases in each flight
print("\n4. Phases present in each flight")
print(df.groupby("flight_id")["flight_phase"].nunique().describe())

# 5. Altitude and vertical-speed ranges
print("\n5. Sensor ranges")
print(df[["altitude_m", "vertical_speed_mps"]].describe())

# 6. Check whether each flight contains all expected phases
expected_phases = {
    "ground", "climb", "cruise", "descent", "landing"
}

phase_sets = df.groupby("flight_id")["flight_phase"].apply(set)

bad_flights = [
    flight_id
    for flight_id, phases in phase_sets.items()
    if phases != expected_phases
]

print("\n6. Flights missing or containing unexpected phases")
print("Count:", len(bad_flights))
print("Flight IDs:", bad_flights[:10])

# 7. Check impossible negative altitude readings
negative_altitudes = (df["altitude_m"] < 0).sum()
print("\n7. Negative altitude readings:", negative_altitudes)

print("\n========== VALIDATION COMPLETE ==========")
