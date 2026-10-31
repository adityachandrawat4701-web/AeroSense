
import time

import joblib
import pandas as pd
import streamlit as st

from physics_simulator import simulate_flight

st.set_page_config(
    page_title="AeroSense",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 AeroSense")
st.subheader("Virtual Flight Monitoring Dashboard")
st.caption("Simulated sensor data • Machine-learning phase classification")

saved_data = joblib.load("flight_phase_model.joblib")
model = saved_data["model"]
features = saved_data["features"]

# Initialize session state
if "flight" not in st.session_state:
    st.session_state.flight = None
    st.session_state.index = 0
    st.session_state.history = []
if "running_toggle" not in st.session_state:
    st.session_state.running_toggle = False


def stop_flight():
    st.session_state.flight = None
    st.session_state.index = 0
    st.session_state.history = []
    st.session_state.running_toggle = False


col1, col2, col3 = st.columns(3)

with col1:
    start_clicked = st.button("▶ Start new flight", use_container_width=True)

with col2:
    st.toggle("Run live simulation", key="running_toggle")

with col3:
    st.button(
        "⏹ Stop flight",
        use_container_width=True,
        on_click=stop_flight,
    )

if start_clicked:
    st.session_state.flight = simulate_flight(
        seed=int(time.time()) % 100000
    )
    st.session_state.index = 0
    st.session_state.history = []
    st.rerun()

if st.session_state.flight is None:
    st.info("Click Start new flight to begin.")
    st.stop()


@st.fragment(run_every=0.2 if st.session_state.running_toggle else None)
def flight_dashboard():
    flight = st.session_state.flight

    if st.session_state.running_toggle:
        start = st.session_state.index
        end = min(start + 2, len(flight))

        if start < end:
            batch = flight.iloc[start:end]

            sensor_data = batch[features]
            predictions = model.predict(sensor_data)

            for (_, row), prediction in zip(batch.iterrows(), predictions):
                st.session_state.history.append({
                    "time_s": float(row["timestamp_s"]),
                    "altitude_m": float(row["altitude_m"]),
                    "vertical_speed_mps": float(row["vertical_speed_mps"]),
                    "vertical_accel_mps2": float(row["vertical_accel_mps2"]),
                    "predicted_phase": prediction,
                    "true_phase": row["flight_phase"],
                })

            st.session_state.index = end

    history = pd.DataFrame(st.session_state.history)

    if history.empty:
        st.info("Turn on Run live simulation to stream readings.")
        return

    latest = history.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Altitude", f"{latest['altitude_m']:.1f} m")
    m2.metric("Vertical speed", f"{latest['vertical_speed_mps']:.2f} m/s")
    m3.metric("Predicted phase", str(latest["predicted_phase"]).upper())
    m4.metric("True phase", str(latest["true_phase"]).upper())

    st.markdown("### Flight telemetry")
    st.line_chart(
        history.set_index("time_s")[["altitude_m"]],
        x_label="Simulation time (s)",
        y_label="Altitude (m)",
    )

    st.markdown("### Prediction history")
    st.dataframe(
        history.tail(15).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    agreement = (
        history["predicted_phase"] == history["true_phase"]
    ).mean()
    st.metric("Prediction agreement so far", f"{agreement:.2%}")

    if st.session_state.index >= len(flight):
        st.success("Virtual flight completed.")
    elif st.session_state.running_toggle:
        st.caption("🟢 Simulation running")
    else:
        st.caption("⏸ Simulation paused")


flight_dashboard()
