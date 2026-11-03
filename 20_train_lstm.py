import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load the datasets
X_train = np.load("outputs/X_train.npy")
y_train = np.load("outputs/y_train.npy")

X_val = np.load("outputs/X_val.npy")
y_val = np.load("outputs/y_val.npy")

# 2. Normalize coordinates relative to the last observed position
# This helps the model learn movement rather than absolute location.

X_train_last = X_train[:, -1:, :]
X_val_last = X_val[:, -1:, :]

X_train_relative = X_train - X_train_last
X_val_relative = X_val - X_val_last

y_train_relative = y_train - X_train_last
y_val_relative = y_val - X_val_last

# 3. Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(11, 2)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(160),
    Reshape((80, 2))
])

# 4. Configure model training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

# 5. Stop training when validation performance stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# 6. Train
history = model.fit(
    X_train_relative,
    y_train_relative,
    validation_data=(X_val_relative, y_val_relative),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 7. Save the model and training history
model.save("outputs/lstm_trajectory_model.keras")
np.save("outputs/lstm_training_history.npy", history.history["loss"])
np.save("outputs/lstm_validation_history.npy", history.history["val_loss"])

print("\nLSTM TRAINING COMPLETED")
print("Best validation loss:", min(history.history["val_loss"]))