import pandas as pd
import matplotlib.pyplot as plt

# Load our extracted trajectory dataset
df = pd.read_csv("outputs/first_scenario_trajectories.csv")

# Remove invalid trajectory points
df = df[df["valid"] == True]

# Create a figure for our plot
plt.figure(figsize=(12, 8))

# Get all unique agent track IDs
track_ids = df["track_index"].unique()

# Plot each agent separately
for track_id in track_ids:

    # Select data belonging to this agent
    agent = df[df["track_index"] == track_id]

    # Separate observed and future trajectory points
    observed = agent[agent["period"] == "observation"]
    future = agent[agent["period"] == "future"]

    # Plot observed trajectory
    plt.plot(
        observed["x"],
        observed["y"],
        marker="o",
        label=f"Track {track_id} - Observed"
    )

    # Plot actual future trajectory
    plt.plot(
        future["x"],
        future["y"],
        marker=".",
        linestyle="--",
        label=f"Track {track_id} - Actual Future"
    )

    # Mark the final observed position
    if not observed.empty:
        plt.scatter(
            observed["x"].iloc[-1],
            observed["y"].iloc[-1],
            marker="X",
            s=100
        )

# Add graph labels and formatting
plt.title("Waymo Agent Trajectories: Observed vs Actual Future")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.legend()
plt.grid(True)
plt.axis("equal")

# Display the graph
plt.show()