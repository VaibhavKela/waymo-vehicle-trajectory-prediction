import numpy as np
import tensorflow as tf

# Load validation data
X_val = np.load("outputs/X_val.npy")
y_val = np.load("outputs/y_val.npy")

# Convert observed positions to relative coordinates
last_position = X_val[:, -1:, :]
X_val_relative = X_val - last_position

# Load trained model
model = tf.keras.models.load_model(
    "outputs/lstm_trajectory_model.keras"
)

# Predict future relative positions
predictions_relative = model.predict(
    X_val_relative
)

# Convert predictions back to original coordinates
predictions = predictions_relative + last_position

# Calculate displacement errors
distances = np.linalg.norm(
    predictions - y_val,
    axis=2
)

# Average Displacement Error
ade = np.mean(distances)

# Final Displacement Error
fde = np.mean(distances[:, -1])

print("\nLSTM VALIDATION RESULTS")
print("=" * 40)
print(f"ADE: {ade:.4f} meters")
print(f"FDE: {fde:.4f} meters")