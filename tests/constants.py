from pathlib import Path

# Test directory paths
TEST_CONFIG_ROOT_DIR = Path(__file__).resolve().parent / "config"
TEST_DATA_ROOT_DIR = Path(__file__).resolve().parent / "data"
TEST_ROOT_DIR = Path(__file__).resolve().parent

# ROS bag specific constants (test-only, not in perception_dataset.constants)
INPUT_BAG_DIR_NAME = "input_bag"
DB3_FILE_PATTERN = "*.db3"
METADATA_YAML_FILENAME = "metadata.yaml"
JSON_EXTENSION = ".json"

# Token field names to exclude in T4 dataset comparisons (test-only)
TOKEN_FIELD_NAMES = [
    "token",
    "sample_data_token",
    "sensor_token",
    "category_token",
    "first_annotation_token",
    "last_annotation_token",
    "log_token",
    "log_tokens",
    "instance_token",
    "visibility_token",
    "first_sample_token",
    "last_sample_token",
    "scene_token",
    "calibrated_sensor_token",
    "ego_pose_token",
    "sample_token",
    "next",
    "prev",
    "attribute_tokens",
]

# Float tolerance for T4 dataset JSON comparisons (math.isclose); absorbs last-bit
# differences between platforms such as Python 3.10 (humble) and 3.12 (jazzy)
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-9
# Number of differing elements shown when a T4 dataset JSON comparison fails
MAX_REPORTED_DIFFERENCES = 5
