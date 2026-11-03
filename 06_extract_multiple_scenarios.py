import tensorflow as tf
import pandas as pd

from waymo_open_dataset.protos import scenario_pb2

# Path to our downloaded Waymo dataset
TFRECORD_PATH = (
    "data/uncompressed_scenario_training_training.tfrecord-00000-of-01000"
)

# Number of scenarios to process initially
MAX_SCENARIOS = 500

# Store extracted trajectory information
all_rows = []

# Read scenarios from the TFRecord file
dataset = tf.data.TFRecordDataset(TFRECORD_PATH)

scenario_count = 0

for raw_record in dataset.take(MAX_SCENARIOS):

    # Create a Scenario object
    scenario = scenario_pb2.Scenario()

    # Decode the raw TFRecord data
    scenario.ParseFromString(raw_record.numpy())

    scenario_count += 1

    # Get the current observation/future boundary
    current_time_index = scenario.current_time_index

    # Get only agents selected for prediction
    for target in scenario.tracks_to_predict:

        track_index = target.track_index
        track = scenario.tracks[track_index]

        # Extract every timestamp for this agent
        for frame_index, state in enumerate(track.states):

            # Determine whether this frame is observed or future
            if frame_index <= current_time_index:
                period = "observation"
            else:
                period = "future"

            all_rows.append({
                "scenario_id": scenario.scenario_id,
                "track_index": track_index,
                "object_id": track.id,
                "object_type": track.object_type,
                "frame_index": frame_index,
                "timestamp": scenario.timestamps_seconds[frame_index],
                "period": period,
                "x": state.center_x,
                "y": state.center_y,
                "heading": state.heading,
                "velocity_x": state.velocity_x,
                "velocity_y": state.velocity_y,
                "valid": state.valid
            })

    # Display progress every 10 scenarios
    if scenario_count % 10 == 0:
        print(f"Processed {scenario_count} scenarios")

# Convert extracted data into a Pandas DataFrame
df = pd.DataFrame(all_rows)

# Save the extracted trajectories
output_path = "outputs/multiple_scenarios_trajectories.csv"

df.to_csv(output_path, index=False)

print("\nEXTRACTION COMPLETED")
print("==============================")
print(f"Scenarios processed: {scenario_count}")
print(f"Total trajectory rows: {len(df)}")
print(f"Unique scenarios: {df['scenario_id'].nunique()}")
print(f"Unique target agents: {df[['scenario_id', 'track_index']].drop_duplicates().shape[0]}")
print(f"Output file: {output_path}")