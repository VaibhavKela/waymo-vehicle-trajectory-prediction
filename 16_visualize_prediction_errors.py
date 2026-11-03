import pandas as pd
import matplotlib.pyplot as plt

# Load prediction error results
df = pd.read_csv("outputs/prediction_error_over_time.csv")

# Create the plot
plt.figure(figsize=(9, 5))

plt.plot(
    df["Future Time (seconds)"],
    df["Average Error (meters)"],
    marker="o",
    linewidth=2
)

plt.title("Vehicle Trajectory Prediction Error Over Time")
plt.xlabel("Prediction Horizon (seconds)")
plt.ylabel("Average Prediction Error (meters)")

plt.xticks([1, 2, 4, 6, 8])
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the graph
plt.savefig("outputs/prediction_error_over_time.png", dpi=300)

plt.show()

print("Graph saved successfully.")