
import pandas as pd
from physics_simulator import simulate_flight

results = []

for flight_id in range(5):
    data = simulate_flight(seed=flight_id)

    results.append({
        "flight_id": flight_id,
        "duration_s": round(data["timestamp_s"].iloc[-1], 1),
        "max_altitude_m": round(data["true_altitude_m"].max(), 1),
        "max_climb_speed_mps": round(
            data["true_vertical_speed_mps"].max(), 2
        ),
        "phase_count": data["flight_phase"].nunique(),
        "samples": len(data),
    })

summary = pd.DataFrame(results)

print("========== FLIGHT VARIATION CHECK ==========")
print(summary.to_string(index=False))

print("\nDo durations differ?",
      summary["duration_s"].nunique() > 1)
print("Do maximum altitudes differ?",
      summary["max_altitude_m"].nunique() > 1)
print("Do sample counts differ?",
      summary["samples"].nunique() > 1)
