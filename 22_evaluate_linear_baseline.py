
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load the new training and validation datasets
X_train = np.load("outputs/X_train.npy")
X_val = np.load("outputs/X_val.npy")
y_train = np.load("outputs/y_train.npy")
y_val = np.load("outputs/y_val.npy")

# 2. Get the last observed position of each vehicle
train_origin = X_train[:, -1:, :]
val_origin = X_val[:, -1:, :]

# 3. Convert observed positions into relative coordinates
X_train_relative = X_train - train_origin
X_val_relative = X_val - val_origin

# 4. Convert future target positions into relative coordinates
y_train_relative = y_train - train_origin

# 5. Flatten trajectories for Linear Regression
X_train_flat = X_train_relative.reshape(len(X_train), -1)
X_val_flat = X_val_relative.reshape(len(X_val), -1)
y_train_flat = y_train_relative.reshape(len(y_train), -1)

# 6. Train the model
model = LinearRegression()
model.fit(X_train_flat, y_train_flat)

# 7. Predict future relative positions
predictions_relative = model.predict(X_val_flat)

# 8. Reshape predictions back into trajectories
predictions_relative = predictions_relative.reshape(-1, 80, 2)

# 9. Convert relative predictions back to absolute coordinates
predictions = predictions_relative + val_origin

# 10. Evaluate the model
errors = np.linalg.norm(predictions - y_val, axis=2)

ade = np.mean(errors)
fde = np.mean(errors[:, -1])

print("RELATIVE LINEAR REGRESSION - VALIDATION")
print("=" * 45)
print(f"ADE: {ade:.4f} meters")
print(f"FDE: {fde:.4f} meters")

# 11. Save the model and validation predictions
joblib.dump(model, "outputs/relative_linear_regression_expanded.pkl")
np.save("outputs/relative_linear_validation_predictions.npy", predictions)

print("\nModel and validation predictions saved successfully.")
