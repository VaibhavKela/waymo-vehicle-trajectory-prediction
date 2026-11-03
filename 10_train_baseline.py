import joblib
import numpy as np

# Load training data
X_train = np.load("outputs/X_train.npy")
y_train = np.load("outputs/y_train.npy")

# Load testing data
X_test = np.load("outputs/X_test.npy")
y_test = np.load("outputs/y_test.npy")

print("DATASET SHAPES")
print("==============================")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Reshape input data for Scikit-learn
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Reshape future trajectories into 160 output values
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)

print("\nRESHAPED DATA")
print("==============================")
print("X_train_flat:", X_train_flat.shape)
print("X_test_flat:", X_test_flat.shape)
print("y_train_flat:", y_train_flat.shape)
print("y_test_flat:", y_test_flat.shape)

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model using our training data
model.fit(X_train_flat, y_train_flat)

# Save the trained model
joblib.dump(model, "outputs/linear_regression_baseline.pkl")

print("\nMODEL SAVED")
print("==============================")
print("Saved to: outputs/linear_regression_baseline.pkl")

print("\nMODEL TRAINING COMPLETED")
print("==============================")
print("Model:", model)
print("Training examples:", X_train_flat.shape[0])
print("Input features:", X_train_flat.shape[1])
print("Prediction outputs:", y_train_flat.shape[1])

# Generate predictions using the test dataset
y_pred_flat = model.predict(X_test_flat)

# Convert predictions back to trajectory format
y_pred = y_pred_flat.reshape(y_pred_flat.shape[0], 80, 2)

print("\nPREDICTIONS GENERATED")
print("==============================")
print("Predicted trajectory shape:", y_pred.shape)
print("Actual trajectory shape:", y_test.shape)

# Display one example
print("\nFIRST TEST EXAMPLE")
print("==============================")
print("Predicted first future position:", y_pred[0, 0])
print("Actual first future position:", y_test[0, 0])

print("\nPredicted final future position:", y_pred[0, -1])
print("Actual final future position:", y_test[0, -1])

# Calculate distance between predicted and actual positions
distances = np.sqrt(
    np.sum((y_pred - y_test) ** 2, axis=2)
)

# Average Displacement Error for each vehicle
ade_per_vehicle = np.mean(distances, axis=1)

# Final Displacement Error for each vehicle
fde_per_vehicle = distances[:, -1]

# Calculate overall metrics
mean_ade = np.mean(ade_per_vehicle)
mean_fde = np.mean(fde_per_vehicle)

print("\nMODEL EVALUATION")
print("==============================")
print(f"Average Displacement Error (ADE): {mean_ade:.4f} meters")
print(f"Final Displacement Error (FDE): {mean_fde:.4f} meters")