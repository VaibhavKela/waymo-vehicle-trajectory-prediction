import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load motion feature datasets
X_train = np.load("outputs/X_motion_train.npy")
X_test = np.load("outputs/X_motion_test.npy")

y_train = np.load("outputs/y_relative_train.npy")
y_test = np.load("outputs/y_relative_test.npy")

# 2. Flatten input features
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# 3. Flatten future positions
y_train_flat = y_train.reshape(len(y_train), -1)

print("TRAINING DATA")
print("=" * 40)
print("X_train:", X_train_flat.shape)
print("y_train:", y_train_flat.shape)
print("X_test:", X_test_flat.shape)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train_flat, y_train_flat)

# 5. Predict future positions
predictions_flat = model.predict(X_test_flat)

# 6. Reshape predictions into trajectories
predictions = predictions_flat.reshape(-1, 80, 2)

# 7. Calculate prediction errors
errors = np.linalg.norm(predictions - y_test, axis=2)

# 8. Calculate ADE and FDE
ade = np.mean(errors)
fde = np.mean(errors[:, -1])

print("\nMOTION FEATURE MODEL RESULTS")
print("=" * 40)
print(f"ADE: {ade:.4f} meters")
print(f"FDE: {fde:.4f} meters")

# 9. Save the model and predictions
joblib.dump(
    model,
    "outputs/motion_linear_regression.pkl"
)

np.save(
    "outputs/motion_linear_predictions.npy",
    predictions
)

print("\nModel and predictions saved successfully.")