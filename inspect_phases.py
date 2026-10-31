
import pandas as pd

df = pd.read_csv("physics_flight_dataset.csv")

print("========== PHASE INSPECTION ==========")

# Count samples in each phase for every flight
phase_counts = pd.crosstab(
    df["flight_id"],
    df["flight_phase"]
)

print("\nSamples per phase across all flights:")
print(df["flight_phase"].value_counts())

print("\nFlights containing each phase:")
for phase in ["ground", "climb", "cruise", "descent", "landing"]:
    if phase in phase_counts.columns:
        count = (phase_counts[phase] > 0).sum()
    else:
        count = 0

    print(f"{phase}: {count} flights")

print("\nFlights with no landing samples:")
if "landing" in phase_counts.columns:
    missing_landing = phase_counts.index[
        phase_counts["landing"] == 0
    ].tolist()
else:
    missing_landing = phase_counts.index.tolist()

print(missing_landing)

print("\nExample flight phase counts:")
print(phase_counts.head(10).to_string())
