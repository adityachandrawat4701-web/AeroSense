
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_flight():
    rng = np.random.default_rng(seed=42)

    # One simulated measurement per second.
    time_s = np.arange(321, dtype=float)

    # Key points in our simplified flight profile.
    key_times = [0, 20, 100, 220, 285, 320]
    key_altitudes = [0, 0, 1500, 1500, 0, 0]

    # Ideal altitude and vertical speed.
    ideal_altitude = np.interp(
        time_s, key_times, key_altitudes
    )
    vertical_speed = np.gradient(ideal_altitude, 1.0)

    # Simulated sensor measurements.
    altitude_m = np.maximum(
        0, ideal_altitude + rng.normal(0, 2, len(time_s))
    )

    # Small sensor noise, without differentiating noisy altitude.
    vertical_speed_mps = (
        vertical_speed + rng.normal(0, 0.15, len(time_s))
    )

    accel_x = rng.normal(0, 0.08, len(time_s))
    accel_y = rng.normal(0, 0.08, len(time_s))
    accel_z = 9.81 + rng.normal(0, 0.12, len(time_s))

    pitch_deg = np.zeros(len(time_s))
    pitch_deg[(time_s >= 20) & (time_s < 100)] = 8
    pitch_deg[(time_s >= 220) & (time_s < 285)] = -5
    pitch_deg += rng.normal(0, 0.5, len(time_s))

    roll_deg = rng.normal(0, 1.0, len(time_s))

    # Ground-truth labels for our simulated flight.
    flight_phase = np.select(
        [
            time_s < 20,
            (time_s >= 20) & (time_s < 100),
            (time_s >= 100) & (time_s < 220),
            (time_s >= 220) & (time_s < 285),
            time_s >= 285,
        ],
        ["ground", "climb", "cruise", "descent", "landing"],
        default="unknown",
    )

    return pd.DataFrame({
        "timestamp_s": time_s.astype(int),
        "altitude_m": altitude_m,
        "vertical_speed_mps": vertical_speed_mps,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "flight_phase": flight_phase,
    })


def main():
    telemetry = simulate_flight()

    # Save all generated telemetry.
    telemetry.to_csv("flight_data.csv", index=False)

    # Set up live plots.
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    altitude_line, = axes[0].plot([], [], linewidth=2)
    speed_line, = axes[1].plot([], [], linewidth=2)

    axes[0].set_title("AeroSense - Live Virtual Flight")
    axes[0].set_ylabel("Altitude (m)")
    axes[1].set_ylabel("Vertical speed (m/s)")
    axes[1].set_xlabel("Simulated time (seconds)")

    axes[0].set_xlim(0, 320)
    axes[0].set_ylim(-50, 1600)
    axes[1].set_ylim(-30, 30)

    for ax in axes:
        ax.grid(True)

    phase_text = fig.text(
        0.5, 0.94, "Phase: Ground",
        ha="center", fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Reveal 20 new simulated seconds every real second.
    chunk_size = 20

    for end in range(chunk_size, len(telemetry) + chunk_size, chunk_size):
        if not plt.fignum_exists(fig.number):
            break

        end = min(end, len(telemetry))
        visible = telemetry.iloc[:end]

        altitude_line.set_data(
            visible["timestamp_s"], visible["altitude_m"]
        )
        speed_line.set_data(
            visible["timestamp_s"], visible["vertical_speed_mps"]
        )

        phase_text.set_text(
            f"Simulated time: {end - 1} s | "
            f"Phase: {visible['flight_phase'].iloc[-1].title()}"
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(1.0)

    plt.ioff()

    print("AeroSense v0.2 - Simulation finished")
    print(f"Total measurements: {len(telemetry)}")
    print("\nMeasurements per phase:")
    print(telemetry["flight_phase"].value_counts().to_string())
    print("\nSaved flight_data.csv")

    # Save the final graph.
    fig.savefig("flight_profile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
