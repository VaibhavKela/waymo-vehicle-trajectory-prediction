import pandas as pd

# Load the extracted dataset
df = pd.read_csv("outputs/multiple_scenarios_trajectories.csv")

print("\nDATASET OVERVIEW")
print("=" * 40)
print("Total rows:", len(df))
print("Total scenarios:", df["scenario_id"].nunique())
print("Total target agents:", df[["scenario_id", "track_index"]].drop_duplicates().shape[0])

# 1. Check valid and invalid states
print("\nVALIDITY CHECK")
print(df["valid"].value_counts())

# 2. Count agent types
object_types = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Cyclist",
    4: "Other"
}

print("\nAGENT TYPE DISTRIBUTION")
print(
    df.groupby("object_type")["track_index"]
    .apply(lambda x: x.groupby(df.loc[x.index, "scenario_id"]).ngroups)
)

# Count unique agents by type across all scenarios
agents = df[["scenario_id", "track_index", "object_type"]].drop_duplicates()

for object_type, name in object_types.items():
    count = len(agents[agents["object_type"] == object_type])
    print(f"{name}: {count}")

# 3. Keep only valid trajectory states
valid_df = df[df["valid"] == True].copy()

# 4. Count valid observation and future frames per agent
agent_summary = (
    valid_df.groupby(["scenario_id", "track_index", "period"])["frame_index"]
    .nunique()
    .unstack(fill_value=0)
)

# Ensure both columns exist
for column in ["observation", "future"]:
    if column not in agent_summary.columns:
        agent_summary[column] = 0

print("\nVALID FRAMES PER AGENT")
print("Average observation frames:", agent_summary["observation"].mean())
print("Average future frames:", agent_summary["future"].mean())

# 5. Identify agents with complete trajectories
complete_agents = agent_summary[
    (agent_summary["observation"] == 11) &
    (agent_summary["future"] == 80)
]

print("\nCOMPLETE TRAJECTORY CHECK")
print("Total agents:", len(agent_summary))
print("Agents with 11 valid observation frames:", 
      (agent_summary["observation"] == 11).sum())
print("Agents with 80 valid future frames:", 
      (agent_summary["future"] == 80).sum())
print("Agents with complete trajectories:", len(complete_agents))

print("\nANALYSIS COMPLETED")