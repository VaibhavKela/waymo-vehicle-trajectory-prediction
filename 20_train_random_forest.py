import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# 1. Load motion feature datasets
X_train = np.load("outputs/X_motion_train.npy")
X_test = np.load("outputs/X_motion_test.npy")

y_train = np.load("outputs/y_relative_train.npy")
y_test = np.load("outputs/y_relative_test.npy")

# 2. Flatten the input and output trajectories
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

y_train_flat = y_train.reshape(len(y_train), -1)

print("TRAINING RANDOM FOREST")
print("=" * 40)
print("Training input:", X_train_flat.shape)
print("Training output:", y_train_flat.shape)

# 3. Create the Random Forest model
base_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 4. Extend Random Forest to predict multiple outputs
model = MultiOutputRegressor(base_model)

# 5. Train the model
model.fit(X_train_flat, y_train_flat)

# 6. Predict future positions
predictions_flat = model.predict(X_test_flat)

# 7. Reshape predictions into trajectories
predictions = predictions_flat.reshape(-1, 80, 2)

# 8. Calculate prediction errors
errors = np.linalg.norm(predictions - y_test, axis=2)

# 9. Calculate ADE and FDE
ade = np.mean(errors)
fde = np.mean(errors[:, -1])

print("\nRANDOM FOREST RESULTS")
print("=" * 40)
print(f"ADE: {ade:.4f} meters")
print(f"FDE: {fde:.4f} meters")

# 10. Save the model and predictions
joblib.dump(model, "outputs/random_forest_model.pkl")

np.save(
    "outputs/random_forest_predictions.npy",
    predictions
)

print("\nModel and predictions saved successfully.")