
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configure the dashboard
st.set_page_config(
    page_title="Vehicle Trajectory Prediction",
    page_icon="🚗",
    layout="wide"
)

# Custom dashboard styling
st.markdown("""
<style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 750;
        letter-spacing: -1px;
    }

    h2, h3 {
        font-weight: 650;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 20px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Dashboard heading
st.title("🚗 Vehicle Trajectory Prediction Dashboard")
st.write(
    "Explore vehicle movements, compare actual and predicted trajectories, "
    "and evaluate our machine learning model."
)


# Project output folder
OUTPUT_DIR = Path("outputs")

# Load test dataset and model predictions
X_test = np.load(OUTPUT_DIR / "X_test.npy")
y_test = np.load(OUTPUT_DIR / "y_test.npy")
predictions = np.load(
    OUTPUT_DIR / "relative_linear_test_predictions.npy"
)

# Display dataset information
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Test Vehicles", len(X_test))
col2.metric("Observed Frames", X_test.shape[1])
col3.metric("Future Prediction Frames", y_test.shape[1])


# Vehicle selection
st.divider()
st.subheader("Explore Vehicle Trajectories")

selected_vehicle = st.selectbox(
    "Select a test vehicle:",
    options=range(len(X_test)),
    format_func=lambda x: f"Vehicle {x + 1}"
)

# Get trajectories for the selected vehicle
observed = X_test[selected_vehicle]
actual = y_test[selected_vehicle]
predicted = predictions[selected_vehicle]


# Calculate prediction errors
errors = np.linalg.norm(actual - predicted, axis=1)

ade = np.mean(errors)
fde = errors[-1]

# Display individual vehicle metrics
metric1, metric2 = st.columns(2)

metric1.metric("Average Displacement Error (ADE)", f"{ade:.2f} m")
metric2.metric("Final Displacement Error (FDE)", f"{fde:.2f} m")


# Create trajectory visualization

# Create styled trajectory visualization
fig, ax = plt.subplots(figsize=(11, 7))

# Dark chart background
fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#171B22")

# Observed trajectory
ax.plot(
    observed[:, 0], observed[:, 1],
    color="#3B82F6",
    marker="o",
    markersize=4,
    linewidth=2.5,
    label="Observed Trajectory"
)

# Actual future trajectory
ax.plot(
    actual[:, 0], actual[:, 1],
    color="#22C55E",
    marker="o",
    markersize=3,
    linewidth=2.5,
    label="Actual Future"
)

# Predicted future trajectory
ax.plot(
    predicted[:, 0], predicted[:, 1],
    color="#EF4444",
    marker="o",
    markersize=3,
    linestyle="--",
    linewidth=2.5,
    label="Predicted Future"
)

# Chart labels and styling
ax.set_title(
    f"Vehicle {selected_vehicle + 1} - Trajectory Comparison",
    color="white",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("X Position (meters)", color="white")
ax.set_ylabel("Y Position (meters)", color="white")

ax.tick_params(colors="white")
ax.grid(color="#64748B", alpha=0.25)
ax.legend(facecolor="#252B36", labelcolor="white", edgecolor="#475569")

for spine in ax.spines.values():
    spine.set_color("#64748B")

ax.axis("equal")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# Observed trajectory
ax.plot(
    observed[:, 0], observed[:, 1],
    "bo-", label="Observed Trajectory"
)

# Actual future trajectory
ax.plot(
    actual[:, 0], actual[:, 1],
    "go-", label="Actual Future Trajectory"
)

# Predicted future trajectory
ax.plot(
    predicted[:, 0], predicted[:, 1],
    "ro--", label="Predicted Future Trajectory"
)

ax.set_title(f"Vehicle {selected_vehicle + 1} Trajectory")
ax.set_xlabel("X Position (meters)")
ax.set_ylabel("Y Position (meters)")
ax.legend()
ax.grid(True)
ax.axis("equal")

st.pyplot(fig)
plt.close(fig)


# Overall test-set performance
st.divider()
st.subheader("Overall Model Performance")

col1, col2 = st.columns(2)

col1.metric("Test ADE", "7.03 meters")
col2.metric("Test FDE", "20.68 meters")

st.caption(
    "Performance measured across 170 test vehicles "
    "from 69 scenarios."
)

# Model methodology
st.divider()
st.header("Model Methodology")

st.markdown("""
Our vehicle trajectory prediction system uses **Relative Linear Regression**.

- **Input:** The vehicle's previous 11 observed positions.
- **Output:** Predicted positions for the next 80 frames.
- **Input features:** Relative X and Y coordinates.
- **Prediction approach:** The model learns the relationship between
  past relative positions and future relative positions.
- **Dataset:** A selected subset of the Waymo Open Motion Dataset.
""")

# Model limitations
st.header("Model Limitations")

st.markdown("""
The current model has some limitations:

- It uses historical vehicle positions as its input.
- It does not use road geometry or map information.
- It does not explicitly model interactions with nearby vehicles.
- It may struggle to predict sudden turns or changes in direction.

These limitations help explain why predicted trajectories can differ
from the actual future trajectories.
""")

# Project Summary
st.divider()
st.header("Project Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Scenarios Processed", "496")
col2.metric("Vehicle Trajectories", "1,167")
col3.metric("Test Vehicles", "170")

st.markdown("""
**Dataset split:**
- Training: 823 vehicles
- Validation: 174 vehicles
- Testing: 170 vehicles

**Final model:** Relative Linear Regression

**Final test performance:**
- ADE: 7.03 meters
- FDE: 20.68 meters
""")

# Prediction error over time
st.divider()
st.subheader("Prediction Error Over Time")

error_data = pd.read_csv(
    OUTPUT_DIR / "prediction_error_over_time.csv"
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    error_data["Future Time (seconds)"],
    error_data["Average Error (meters)"],
    marker="o",
    linewidth=2
)

ax.set_title("Average Prediction Error at Different Future Horizons")
ax.set_xlabel("Future Time (seconds)")
ax.set_ylabel("Average Error (meters)")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

st.dataframe(error_data, use_container_width=True)
