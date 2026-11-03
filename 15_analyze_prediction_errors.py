
import numpy as np
import pandas as pd

# 1. Load actual and predicted validation trajectories
y_val = np.load("outputs/y_val.npy")
predictions = np.load(
    "outputs/relative_linear_validation_predictions.npy"
)

# 2. Calculate Euclidean distance at every future frame
errors = np.linalg.norm(predictions - y_val, axis=2)

# 3. Select future time points
# Each frame represents 0.1 seconds
time_points = [10, 20, 40, 60, 80]

results = []

for frame in time_points:

    # Average error across all validation vehicles
    average_error = np.mean(errors[:, frame - 1])

    results.append({
        "Future Time (seconds)": frame / 10,
        "Average Error (meters)": average_error
    })

# 4. Display results
results_df = pd.DataFrame(results)

print("PREDICTION ERROR OVER TIME")
print("=" * 40)
print(results_df.to_string(index=False))

# 5. Save results
results_df.to_csv(
    "outputs/prediction_error_over_time.csv",
    index=False
)

print("\nResults saved to outputs/prediction_error_over_time.csv")
