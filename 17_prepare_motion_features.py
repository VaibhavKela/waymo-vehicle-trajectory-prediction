import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Load the original trajectory dataset
df = pd.read_csv("outputs/multiple_scenarios_trajectories.csv")

# 2. Keep valid vehicle states
df = df[
    (df["valid"] == True) &
    (df["object_type"] == 1)
].copy()

# 3. Prepare feature and target lists
X_motion = []
y_relative = []
agent_keys = []

# 4. Process each vehicle separately
for (scenario_id, track_index), agent in df.groupby(
    ["scenario_id", "track_index"]
):

    # Sort frames chronologically
    agent = agent.sort_values("frame_index")

    # Separate observed and future frames
    observed = agent[agent["period"] == "observation"]
    future = agent[agent["period"] == "future"]

    # Keep only complete trajectories
    if len(observed) != 11 or len(future) != 80:
        continue

    # 5. Extract the last observed position
    origin_x = observed.iloc[-1]["x"]
    origin_y = observed.iloc[-1]["y"]

    # 6. Calculate relative X and Y positions
    relative_x = observed["x"].to_numpy() - origin_x
    relative_y = observed["y"].to_numpy() - origin_y

    # 7. Extract velocity
    vx = observed["velocity_x"].to_numpy()
    vy = observed["velocity_y"].to_numpy()

    # 8. Calculate acceleration
    ax = np.diff(vx, prepend=vx[0]) / 0.1
    ay = np.diff(vy, prepend=vy[0]) / 0.1

    # 9. Convert heading into sine and cosine
    heading = observed["heading"].to_numpy()

    heading_sin = np.sin(heading)
    heading_cos = np.cos(heading)

    # 10. Combine all eight features
    features = np.column_stack([
        relative_x,
        relative_y,
        vx,
        vy,
        ax,
        ay,
        heading_sin,
        heading_cos
    ])

    # 11. Prepare future positions relative to current position
    future_x = future["x"].to_numpy() - origin_x
    future_y = future["y"].to_numpy() - origin_y

    future_positions = np.column_stack([
        future_x,
        future_y
    ])

    # 12. Store the results
    X_motion.append(features)
    y_relative.append(future_positions)

    agent_keys.append((scenario_id, track_index))


# 13. Convert lists into NumPy arrays
X_motion = np.array(X_motion)
y_relative = np.array(y_relative)

# 14. Recreate the same scenario-based train/test split
groups = np.array([scenario_id for scenario_id, _ in agent_keys])

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X_motion, y_relative, groups=groups)
)

# 15. Save the datasets
np.save("outputs/X_motion.npy", X_motion)
np.save("outputs/y_relative.npy", y_relative)

np.save("outputs/X_motion_train.npy", X_motion[train_indices])
np.save("outputs/X_motion_test.npy", X_motion[test_indices])

np.save("outputs/y_relative_train.npy", y_relative[train_indices])
np.save("outputs/y_relative_test.npy", y_relative[test_indices])

# 16. Display results
print("\nMOTION FEATURE DATASET")
print("=" * 40)

print("Total vehicle examples:", len(X_motion))
print("Training examples:", len(train_indices))
print("Testing examples:", len(test_indices))

print("\nARRAY SHAPES")
print("X_motion:", X_motion.shape)
print("y_relative:", y_relative.shape)

print("\nFEATURES PER FRAME:", X_motion.shape[2])

print("\nDATASET PREPARATION COMPLETED")