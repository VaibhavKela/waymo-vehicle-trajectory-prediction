import numpy as np
import pandas as pd

# Load the extracted dataset
df = pd.read_csv("outputs/multiple_scenarios_trajectories.csv")

# Keep only valid states
df = df[df["valid"] == True].copy()

# Keep only vehicles (object_type = 1)
df = df[df["object_type"] == 1].copy()

# Store input and target trajectories
X = []
y = []

# Group agents by scenario and track
for (scenario_id, track_index), agent in df.groupby(
    ["scenario_id", "track_index"]
):

    # Sort frames chronologically
    agent = agent.sort_values("frame_index")

    # Separate observation and future frames
    observed = agent[agent["period"] == "observation"]
    future = agent[agent["period"] == "future"]

    # Require exactly 11 observed and 80 future frames
    if len(observed) != 11 or len(future) != 80:
        continue

    # Extract observed X and Y positions
    observed_positions = observed[["x", "y"]].to_numpy()

    # Extract future X and Y positions
    future_positions = future[["x", "y"]].to_numpy()

    # Store trajectories
    X.append(observed_positions)
    y.append(future_positions)

# Convert lists into NumPy arrays
X = np.array(X)
y = np.array(y)

# Save prepared datasets
np.save("outputs/X_observed.npy", X)
np.save("outputs/y_future.npy", y)

# Display dataset information
print("\nTRAINING DATA PREPARATION")
print("=" * 40)

print("Input shape (X):", X.shape)
print("Target shape (y):", y.shape)
print("Number of training examples:", len(X))

print("\nDATASET SAVED")
print("outputs/X_observed.npy")
print("outputs/y_future.npy")