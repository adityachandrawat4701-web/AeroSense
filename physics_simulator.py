
import numpy as np
import pandas as pd


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def simulate_flight(seed=42, max_altitude_m=120.0):
    rng = np.random.default_rng(seed)

    dt = 0.1
    ground_before = rng.uniform(5, 10)
    cruise_duration = rng.uniform(35, 60)
    climb_speed = rng.uniform(4, 6)
    descent_speed = rng.uniform(2, 3)
    max_accel = rng.uniform(0.5, 0.8)
    braking_accel = rng.uniform(0.4, 0.7)

    # Approximate climb time, followed by a short transition.
    climb_duration = max_altitude_m / climb_speed + 8.0
    cruise_start = ground_before + climb_duration
    descent_start = cruise_start + cruise_duration
    max_steps = int((descent_start + max_altitude_m / descent_speed + 60) / dt)

    time_s = np.arange(max_steps + 1) * dt
    altitude = np.zeros(max_steps + 1)
    vertical_speed = np.zeros(max_steps + 1)
    vertical_accel = np.zeros(max_steps + 1)
    phases = np.full(max_steps + 1, "ground", dtype=object)

    landed = False
    touchdown_index = None

    for i in range(1, len(time_s)):
        t = time_s[i]
        h = altitude[i - 1]
        v = vertical_speed[i - 1]

        if landed:
            altitude[i] = 0.0
            vertical_speed[i] = 0.0
            vertical_accel[i] = 0.0
            phases[i] = "ground"

            if i >= touchdown_index + int(5 / dt):
                time_s = time_s[:i + 1]
                altitude = altitude[:i + 1]
                vertical_speed = vertical_speed[:i + 1]
                vertical_accel = vertical_accel[:i + 1]
                phases = phases[:i + 1]
                break
            continue

        if t < ground_before:
            phase = "ground"
            target_speed = 0.0

        elif t < cruise_start:
            phase = "climb"

            # Begin slowing before the altitude ceiling.
            remaining_height = max(0.0, max_altitude_m - h)
            braking_speed = np.sqrt(
                2.0 * braking_accel * remaining_height
            )
            target_speed = min(climb_speed, braking_speed)

            # Smooth the initial takeoff.
            ramp_duration = 3.0
            if t < ground_before + ramp_duration:
                target_speed *= smoothstep(
                    (t - ground_before) / ramp_duration
                )

        elif t < descent_start:
            phase = "cruise"
            target_speed = 0.0

        else:
            phase = "descent"
            safe_speed = np.sqrt(
                2.0 * braking_accel * max(h, 0.0)
            )
            target_speed = -min(descent_speed, safe_speed)

            # Smoothly start the descent.
            if t < descent_start + 5.0:
                target_speed *= smoothstep(
                    (t - descent_start) / 5.0
                )

        dv = np.clip(
            target_speed - v,
            -max_accel * dt,
            max_accel * dt,
        )
        new_speed = v + dv
        new_altitude = h + 0.5 * (v + new_speed) * dt

        # Prevent the simulated aircraft from exceeding its ceiling.
        if new_altitude >= max_altitude_m:
            new_altitude = max_altitude_m
            new_speed = 0.0

        if new_altitude <= 0.0 and t >= descent_start:
            altitude[i] = 0.0
            vertical_speed[i] = 0.0
            vertical_accel[i] = 0.0
            phases[i] = "landing"
            landed = True
            touchdown_index = i
        else:
            altitude[i] = max(0.0, new_altitude)
            vertical_speed[i] = new_speed
            vertical_accel[i] = dv / dt

            if phase == "descent" and new_altitude < 15.0:
                phases[i] = "landing"
            else:
                phases[i] = phase

    n = len(time_s)

    measured_altitude = np.maximum(
        0.0, altitude + rng.normal(0.0, 0.5, n)
    )
    measured_speed = (
        vertical_speed + rng.normal(0.0, 0.15, n)
    )
    measured_accel = (
        vertical_accel + rng.normal(0.0, 0.08, n)
    )

    return pd.DataFrame({
        "timestamp_s": time_s,
        "true_altitude_m": altitude,
        "true_vertical_speed_mps": vertical_speed,
        "true_vertical_accel_mps2": vertical_accel,
        "altitude_m": measured_altitude,
        "vertical_speed_mps": measured_speed,
        "vertical_accel_mps2": measured_accel,
        "flight_phase": phases,
    })


if __name__ == "__main__":
    data = simulate_flight()

    data.to_csv("physics_test.csv", index=False)

    print("Samples:", len(data))
    print("Maximum true altitude:",
          round(data["true_altitude_m"].max(), 2), "m")
    print("Touchdown reached:",
          bool((data["flight_phase"] == "landing").any()))
    print("Final altitude:",
          round(data["true_altitude_m"].iloc[-1], 2), "m")
    print("\nGround-truth phase counts:")
    print(data["flight_phase"].value_counts().to_string())
    print("\nSaved physics_test.csv")
