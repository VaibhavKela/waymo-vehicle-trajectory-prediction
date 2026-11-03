
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Load validation data
X_val = np.load("outputs/X_val.npy")
y_val = np.load("outputs/y_val.npy")

# 2. Load the new Relative Linear Regression model
model = joblib.load(
    "outputs/relative_linear_regression_expanded.pkl"
)

# 3. Convert observed trajectories into relative coordinates
val_origin = X_val[:, -1:, :]
X_val_relative = X_val - val_origin

# 4. Flatten trajectories for the model
X_val_flat = X_val_relative.reshape(len(X_val), -1)

# 5. Generate future predictions
predictions_relative = model.predict(X_val_flat)

# 6. Convert predictions back into trajectory format
predictions_relative = predictions_relative.reshape(-1, 80, 2)

# 7. Convert relative predictions back to absolute coordinates
predictions = predictions_relative + val_origin

# 8. Select the first validation vehicle
vehicle_index = 0

observed = X_val[vehicle_index]
actual_future = y_val[vehicle_index]
predicted_future = predictions[vehicle_index]

# 9. Plot the trajectories
plt.figure(figsize=(10, 8))

plt.plot(
    observed[:, 0],
    observed[:, 1],
    "b-o",
    label="Observed trajectory"
)

plt.plot(
    actual_future[:, 0],
    actual_future[:, 1],
    "g--",
    marker=".",
    markersize=4,
    label="Actual future trajectory"
)

plt.plot(
    predicted_future[:, 0],
    predicted_future[:, 1],
    "r--",
    marker=".",
    markersize=4,
    label="Predicted future trajectory"
)

plt.scatter(
    observed[-1, 0],
    observed[-1, 1],
    color="black",
    label="Current position"
)

plt.title("Vehicle Trajectory Prediction - Relative Linear Regression")
plt.xlabel("X coordinate (meters)")
plt.ylabel("Y coordinate (meters)")
plt.legend()
plt.grid(True)
plt.axis("equal")

plt.savefig("outputs/relative_linear_validation_prediction.png", dpi=300)

plt.show()

print("Trajectory visualization saved successfully.")
