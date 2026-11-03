
import pandas as pd

# 1. Load our extracted dataset
df = pd.read_csv(
    "outputs/first_scenario_trajectories.csv"
)

print("\nDATASET OVERVIEW")
print("=" * 40)

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

# 2. Display first five rows
print("\nFIRST FIVE ROWS")
print(df.head())

# 3. Check data types
print("\nDATA TYPES")
print(df.dtypes)

# 4. Check valid and invalid observations
print("\nVALIDITY CHECK")
print(df["valid"].value_counts())

# 5. Count valid observations for each vehicle
print("\nVALID OBSERVATIONS PER VEHICLE")

valid_counts = (
    df[df["valid"] == True]
    .groupby("track_index")
    .size()
)

print(valid_counts)

# 6. Calculate average speed for valid observations
df["speed"] = (
    df["velocity_x"]**2 +
    df["velocity_y"]**2
) ** 0.5

print("\nAVERAGE SPEED PER VEHICLE (m/s)")

average_speed = (
    df[df["valid"] == True]
    .groupby("track_index")["speed"]
    .mean()
)

print(average_speed)

# 7. Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())
