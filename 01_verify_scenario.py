
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from pathlib import Path
from collections import Counter

# 1. Locate our dataset
DATA_FILE = Path(
    "data/uncompressed_scenario_training_training.tfrecord-00000-of-01000"
)

# 2. Read the first scenario
dataset = tf.data.TFRecordDataset([str(DATA_FILE)])

record = next(iter(dataset))

# 3. Parse the scenario
scenario = scenario_pb2.Scenario()
scenario.ParseFromString(record.numpy())

# 4. Display basic information
print("\nSCENARIO INFORMATION")
print("=" * 40)

print("Scenario ID:", scenario.scenario_id)
print("Number of tracks:", len(scenario.tracks))
print("Number of timestamps:", len(scenario.timestamps_seconds))
print("Number of map features:", len(scenario.map_features))

# 5. Inspect available scenario fields
print("\nSCENARIO FIELDS")
print("=" * 40)

for field in scenario.DESCRIPTOR.fields:
    print(field.name)

# 6. Count map feature types
print("\nMAP FEATURE TYPES")
print("=" * 40)

feature_counts = Counter()

for feature in scenario.map_features:
    populated_fields = [
        field.name
        for field, value in feature.ListFields()
        if field.name != "id"
    ]

    for field_name in populated_fields:
        feature_counts[field_name] += 1

for feature_type, count in feature_counts.items():
    print(f"{feature_type}: {count}")

print("\nInspection completed!")


# 7. Understand prediction-related information

print("\nPREDICTION STRUCTURE")
print("=" * 40)

print("Current time index:", scenario.current_time_index)

print("Self-driving car track index:",
      scenario.sdc_track_index)

print("Tracks selected for prediction:",
      len(scenario.tracks_to_predict))

print("\nPrediction track indices:")

for track_index in scenario.tracks_to_predict:
    print(track_index)

print("\nTimestamp information:")
print("First timestamp:", scenario.timestamps_seconds[0])
print("Last timestamp:", scenario.timestamps_seconds[-1])

print("\nTime interval between observations:",
      scenario.timestamps_seconds[1] -
      scenario.timestamps_seconds[0])

