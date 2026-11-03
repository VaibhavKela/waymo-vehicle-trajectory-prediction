import numpy as np

# Load testing data
X_test = np.load("outputs/X_test.npy")
y_test = np.load("outputs/y_test.npy")

# Dataset settings
OBSERVATION_INTERVAL = 0.1  # seconds
FUTURE_INTERVAL = 0.1       # seconds

print("CONSTANT VELOCITY BASELINE")
print("==============================")
print("Test examples:", X_test.shape[0])
print("Observed frames:", X_test.shape[1])
print("Future frames:", y_test.shape[1])

# Get the first and last observed positions
first_position = X_test[:, 0, :]
last_position = X_test[:, -1, :]

# Calculate elapsed observation time
observation_time = (X_test.shape[1] - 1) * OBSERVATION_INTERVAL

# Estimate velocity in X and Y directions
velocity = (last_position - first_position) / observation_time

print("\nESTIMATED VELOCITY")
print("==============================")
print("First vehicle velocity:", velocity[0], "meters/second")

# Get the latest observed position of every vehicle
current_position = X_test[:, -1, :]

# Create future time steps: 0.1, 0.2, ..., 8.0 seconds
future_times = np.arange(1, 81) * FUTURE_INTERVAL

# Predict future positions using constant velocity
predicted_trajectories = (
    current_position[:, np.newaxis, :]
    + velocity[:, np.newaxis, :] * future_times[np.newaxis, :, np.newaxis]
)

print("\nPREDICTIONS GENERATED")
print("==============================")
print("Predicted trajectories shape:", predicted_trajectories.shape)

print("\nFIRST TEST VEHICLE")
print("==============================")
print("Current position:", current_position[0])
print("Predicted first future position:", predicted_trajectories[0, 0])
print("Predicted final future position:", predicted_trajectories[0, -1])

# Calculate distance between predicted and actual positions
distances = np.sqrt(
    np.sum((predicted_trajectories - y_test) ** 2, axis=2)
)

# Calculate ADE for each vehicle
ade_per_vehicle = np.mean(distances, axis=1)

# Calculate FDE for each vehicle
fde_per_vehicle = distances[:, -1]

# Calculate overall average metrics
mean_ade = np.mean(ade_per_vehicle)
mean_fde = np.mean(fde_per_vehicle)

print("\nCONSTANT VELOCITY MODEL EVALUATION")
print("========================================")
print(f"Average Displacement Error (ADE): {mean_ade:.4f} meters")
print(f"Final Displacement Error (FDE): {mean_fde:.4f} meters")