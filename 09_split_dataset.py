import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load the expanded dataset
X = np.load("outputs/X_observed.npy")
y = np.load("outputs/y_future.npy")

df = pd.read_csv("outputs/multiple_scenarios_trajectories.csv")

# Keep valid vehicle observations
df = df[
    (df["valid"] == True) &
    (df["object_type"] == 1)
]

# Identify agents with complete trajectories
valid_agents = []

for (scenario_id, track_index), agent in df.groupby(
    ["scenario_id", "track_index"]
):
    observed = agent[agent["period"] == "observation"]
    future = agent[agent["period"] == "future"]

    if len(observed) == 11 and len(future) == 80:
        valid_agents.append((scenario_id, track_index))

valid_agents = sorted(valid_agents)

# Each example belongs to a scenario
groups = np.array([
    scenario_id for scenario_id, _ in valid_agents
])

# Check that arrays and scenario IDs are aligned
assert len(X) == len(y) == len(groups)

# First split: 70% training, 30% temporary
first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    first_split.split(X, y, groups=groups)
)

# Second split: divide temporary data equally
# into validation and testing
second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_relative_idx, test_relative_idx = next(
    second_split.split(
        X[temp_idx],
        y[temp_idx],
        groups=groups[temp_idx]
    )
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]

# Create final datasets
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Save datasets
np.save("outputs/X_train.npy", X_train)
np.save("outputs/y_train.npy", y_train)

np.save("outputs/X_val.npy", X_val)
np.save("outputs/y_val.npy", y_val)

np.save("outputs/X_test.npy", X_test)
np.save("outputs/y_test.npy", y_test)

print("\nDATASET SPLIT COMPLETED")
print("=" * 40)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)

print("\nUnique scenarios:")
print("Training:", len(np.unique(groups[train_idx])))
print("Validation:", len(np.unique(groups[val_idx])))
print("Testing:", len(np.unique(groups[test_idx])))

# Verify that no scenarios overlap
train_scenarios = set(groups[train_idx])
val_scenarios = set(groups[val_idx])
test_scenarios = set(groups[test_idx])

assert train_scenarios.isdisjoint(val_scenarios)
assert train_scenarios.isdisjoint(test_scenarios)
assert val_scenarios.isdisjoint(test_scenarios)

print("\nScenario separation verified!")