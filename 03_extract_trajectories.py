
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from pathlib import Path
import csv

# 1. Locate the dataset
DATA_FILE = Path(
    "data/uncompressed_scenario_training_training.tfrecord-00000-of-01000"
)

# 2. Read the first scenario
dataset = tf.data.TFRecordDataset([str(DATA_FILE)])
record = next(iter(dataset))

scenario = scenario_pb2.Scenario()
scenario.ParseFromString(record.numpy())

# 3. Identify vehicles selected for prediction
target_track_indices = [
    item.track_index
    for item in scenario.tracks_to_predict
]

# 4. Prepare CSV output
output_file = Path("outputs/first_scenario_trajectories.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

rows = []

# 5. Extract each target vehicle's trajectory
for track_index in target_track_indices:

    track = scenario.tracks[track_index]

    for frame_index, state in enumerate(track.states):

        timestamp = scenario.timestamps_seconds[frame_index]

        # Identify observation or prediction period
        period = (
            "observation"
            if frame_index <= scenario.current_time_index
            else "future"
        )

        rows.append({
            "scenario_id": scenario.scenario_id,
            "track_index": track_index,
            "object_id": track.id,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "period": period,
            "x": state.center_x,
            "y": state.center_y,
            "heading": state.heading,
            "velocity_x": state.velocity_x,
            "velocity_y": state.velocity_y,
            "valid": state.valid
        })

# 6. Save extracted trajectories
with open(output_file, "w", newline="") as csvfile:

    fieldnames = rows[0].keys()

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

# 7. Display results
print("TRAJECTORY EXTRACTION COMPLETED")
print("=" * 40)

print("Target vehicles:", len(target_track_indices))
print("Total extracted rows:", len(rows))
print("Output file:", output_file)

print("\nRows per vehicle:")

for track_index in target_track_indices:
    count = sum(
        1 for row in rows
        if row["track_index"] == track_index
    )
    print(f"Track {track_index}: {count} frames")
    