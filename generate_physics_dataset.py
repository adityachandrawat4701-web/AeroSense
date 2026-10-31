
import pandas as pd
from physics_simulator import simulate_flight

NUM_FLIGHTS = 100
all_flights = []

for flight_id in range(NUM_FLIGHTS):
    flight = simulate_flight(seed=flight_id)

    flight["flight_id"] = flight_id
    all_flights.append(flight)

dataset = pd.concat(all_flights, ignore_index=True)
dataset.to_csv("physics_flight_dataset.csv", index=False)

print("========== PHYSICS DATASET ==========")
print("Flights:", dataset["flight_id"].nunique())
print("Total samples:", len(dataset))
print("\nGround-truth phase counts:")
print(dataset["flight_phase"].value_counts().to_string())
print("\nMissing values:", dataset.isnull().sum().sum())

print("\nPhases present in each flight:")
print(dataset.groupby("flight_id")["flight_phase"].nunique().value_counts().sort_index())
print("\nSaved physics_flight_dataset.csv")
