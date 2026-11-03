import numpy as np
import joblib

# 1. Load the untouched test dataset
X_test = np.load("outputs/X_test.npy")
y_test = np.load("outputs/y_test.npy")

# 2. Load the trained Relative Linear Regression model
model = joblib.load(
    "outputs/relative_linear_regression_expanded.pkl"
)

# 3. Convert observed positions into relative coordinates
test_origin = X_test[:, -1:, :]
X_test_relative = X_test - test_origin

# 4. Flatten trajectories
X_test_flat = X_test_relative.reshape(len(X_test), -1)

# 5. Predict future relative positions
predictions_relative = model.predict(X_test_flat)

# 6. Reshape predictions into trajectories
predictions_relative = predictions_relative.reshape(-1, 80, 2)

# 7. Convert predictions back to absolute coordinates
predictions = predictions_relative + test_origin

# 8. Calculate displacement errors
errors = np.linalg.norm(predictions - y_test, axis=2)

# 9. Calculate final evaluation metrics
ade = np.mean(errors)
fde = np.mean(errors[:, -1])

print("\nFINAL TEST SET EVALUATION")
print("=" * 45)
print(f"Test vehicles: {len(X_test)}")
print(f"ADE: {ade:.4f} meters")
print(f"FDE: {fde:.4f} meters")

# 10. Save final test predictions
np.save(
    "outputs/relative_linear_test_predictions.npy",
    predictions
)

print("\nFinal test predictions saved successfully.")