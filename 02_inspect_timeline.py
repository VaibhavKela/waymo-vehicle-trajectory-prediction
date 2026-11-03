
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from pathlib import Path

# Locate dataset
DATA_FILE = Path(
    "data/uncompressed_scenario_training_training.tfrecord-00000-of-01000"
)

# Read first scenario
dataset = tf.data.TFRecordDataset([str(DATA_FILE)])
record = next(iter(dataset))

scenario = scenario_pb2.Scenario()
scenario.ParseFromString(record.numpy())

# Inspect timeline
print("SCENARIO TIMELINE")
print("=" * 40)

print("Total timestamps:", len(scenario.timestamps_seconds))
print("Current time index:", scenario.current_time_index)

print("First timestamp:", scenario.timestamps_seconds[0])
print("Current timestamp:", scenario.timestamps_seconds[scenario.current_time_index])
print("Last timestamp:", scenario.timestamps_seconds[-1])

print("\nTIME INTERVAL BETWEEN FRAMES:")
print(
    scenario.timestamps_seconds[1] -
    scenario.timestamps_seconds[0]
)

print("\nOBSERVATION FRAMES:", scenario.current_time_index + 1)
print(
    "FUTURE FRAMES:",
    len(scenario.timestamps_seconds) - scenario.current_time_index - 1
)