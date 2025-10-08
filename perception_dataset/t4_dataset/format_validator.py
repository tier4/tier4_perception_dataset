from pathlib import Path

from loguru import logger
from t4_devkit import Tier4

from perception_dataset.constants import EXTENSION_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.classes import schema_names


def validate_format(t4_dataset: Tier4, root_path: str):
    validate_scene(t4_dataset)
    validate_sample(t4_dataset)
    validate_sample_data(t4_dataset, Path(root_path))
    validate_ego_pose(t4_dataset)
    validate_calibrated_sensor(t4_dataset)
    validate_instance(t4_dataset)
    validate_sample_annotation(t4_dataset)


def find_in_table(t4_dataset: Tier4, table_name: str, token: str) -> bool:
    """This function is the modification of `get` of NuScenes not to raise an error.
    https://github.com/nutonomy/nuscenes-devkit/blob/28765b8477dbd3331bacd922fada867c2c4db1d7/python-sdk/nuscenes/nuscenes.py#L207-L225
    """
    assert table_name in t4_dataset._token2idx, f"{table_name} not found"

    token_ind = t4_dataset._token2idx[table_name].get(token)
    if token_ind is None:
        return False

    table = getattr(t4_dataset, table_name)
    if token_ind < 0 and token_ind >= len(table):
        return False

    return True


def _logger_wrapper(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        logger.info(f"`{func.__name__}` has passed!")
        return ret

    return wrapper


@_logger_wrapper
def validate_directory_structure(root_path: str):
    root_path = Path(root_path)
    anno_path = root_path / T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value
    data_path = root_path / T4_FORMAT_DIRECTORY_NAME.DATA.value

    assert anno_path.is_dir(), f"{anno_path} isn't found"
    assert data_path.is_dir(), f"{data_path} isn't found"

    for schema_name in schema_names:
        json_path = anno_path / (schema_name + EXTENSION_ENUM.JSON.value)
        assert json_path.is_file(), f"{schema_name} isn't found"


@_logger_wrapper
def validate_scene(t4_dataset: Tier4):
    assert len(t4_dataset.scene) == 1, "T4Dataset must have 1 scene."

    scene = t4_dataset.scene[0]
    assert find_in_table(t4_dataset, "log", scene.log_token), "scene.log_token isn't found in log"
    assert find_in_table(
        t4_dataset, "sample", scene.first_sample_token
    ), "scene.first_sample_token isn't found in log"
    assert find_in_table(
        t4_dataset, "sample", scene.last_sample_token
    ), "scene.last_sample_token isn't found in log"


@_logger_wrapper
def validate_sample(t4_dataset: Tier4):
    assert len(t4_dataset.sample) > 0, "There are no sample."

    no_next_token_count: int = 0
    no_prev_token_count: int = 0
    for sample in t4_dataset.sample:
        assert find_in_table(
            t4_dataset, "scene", sample.scene_token
        ), "sample.scene_token isn't found in scene."

        next_token = sample.next
        if next_token == "":
            no_next_token_count += 1
        else:
            assert find_in_table(t4_dataset, "sample", next_token), "sample.next isn't found in sample."

        prev_token = sample.prev
        if prev_token == "":
            no_prev_token_count += 1
        else:
            assert find_in_table(
                t4_dataset, "sample", prev_token
            ), "sample.prev_token isn't found in sample."

    assert no_next_token_count == len(
        t4_dataset.scene
    ), f"There are more than {len(t4_dataset.scene)} sample of empty scene.next."
    assert no_prev_token_count == len(
        t4_dataset.scene
    ), f"There are more than {len(t4_dataset.scene)} sample of empty scene.prev."


@_logger_wrapper
def validate_sample_data(t4_dataset: Tier4, root_path: Path):
    assert len(t4_dataset.sample_data), "There are no sample_data."

    no_next_token_count: int = 0
    no_prev_token_count: int = 0
    for sample_data in t4_dataset.sample_data:
        if not sample_data.is_valid:
            continue
        assert find_in_table(
            t4_dataset, "sample", sample_data.sample_token
        ), "sample_data.sample_token isn't found in sample."
        assert find_in_table(
            t4_dataset, "ego_pose", sample_data.ego_pose_token
        ), "sample_data.ego_pose_token isn't found in sample."
        assert find_in_table(
            t4_dataset, "calibrated_sensor", sample_data.calibrated_sensor_token
        ), "sample_data.calibrated_pose_token isn't found in sample."

        filename: str = sample_data.filename
        assert (root_path / filename).is_file(), f"{filename} isn't found."

        next_token = sample_data.next
        if next_token == "":
            no_next_token_count += 1
        else:
            assert find_in_table(
                t4_dataset, "sample_data", next_token
            ), "sample_data.next isn't found in sample."

        prev_token = sample_data.prev
        if prev_token == "":
            no_prev_token_count += 1
        else:
            assert find_in_table(
                t4_dataset, "sample_data", prev_token
            ), "sample_data.prev isn't found in sample_data."

    # NOTE(yukke42): There are len(t4_dataset.calibrated_sensor) sensors for each scene.
    expected_count = len(t4_dataset.calibrated_sensor) * len(t4_dataset.scene)
    assert (
        no_next_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.next."
    assert (
        no_prev_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.prev."


@_logger_wrapper
def validate_ego_pose(t4_dataset: Tier4):
    assert len(t4_dataset.ego_pose) > 0, "There are no ego_pose."


@_logger_wrapper
def validate_calibrated_sensor(t4_dataset: Tier4):
    assert len(t4_dataset.calibrated_sensor) > 0, "There are no calibrated_sensor."

    for calibrated_sensor in t4_dataset.calibrated_sensor:
        assert find_in_table(
            t4_dataset, "sensor", calibrated_sensor.sensor_token
        ), "calibrated_sensor.sensor_token isn't found in sensor."


@_logger_wrapper
def validate_instance(t4_dataset: Tier4):
    assert len(t4_dataset.instance) > 0, "There are no instance."

    for instance in t4_dataset.instance:
        if instance.nbr_annotations == 0:
            logger.warning(f"instance:{instance.token} has no 3D annotation")
            continue
        assert find_in_table(
            t4_dataset, "category", instance.category_token
        ), "instance.category_token isn't found in category."
        assert find_in_table(
            t4_dataset, "sample_annotation", instance.first_annotation_token
        ), "instance.first_annotation_token isn't found in sample_annotation."
        assert find_in_table(
            t4_dataset, "sample_annotation", instance.last_annotation_token
        ), "instance.last_annotation_token isn't found in sample_annotation."


@_logger_wrapper
def validate_sample_annotation(t4_dataset: Tier4):
    assert len(t4_dataset.sample_annotation) > 0, "There are no sample_annotation."

    no_next_token_count: int = 0
    no_prev_token_count: int = 0
    for sample_annotation in t4_dataset.sample_annotation:
        assert find_in_table(
            t4_dataset, "sample", sample_annotation.sample_token
        ), "sample_annotation.sample_token isn't found in sample."
        assert find_in_table(
            t4_dataset, "instance", sample_annotation.instance_token
        ), "sample_annotation.instance_token isn't found in instance."

        for i, attribute_token in enumerate(sample_annotation.attribute_tokens):
            assert find_in_table(
                t4_dataset, "attribute", attribute_token
            ), f"sample_annotation.attribute_tokens[{i}] isn't found in attribute."
        assert find_in_table(
            t4_dataset, "visibility", sample_annotation.visibility_token
        ), "sample_annotation.visibility_token isn't found in visibility."

        next_token = sample_annotation.next
        if next_token == "":
            no_next_token_count += 1
        else:
            assert find_in_table(
                t4_dataset, "sample_annotation", next_token
            ), "sample_annotation.next isn't found in sample."

        prev_token = sample_annotation.prev
        if prev_token == "":
            no_prev_token_count += 1
        else:
            assert find_in_table(
                t4_dataset, "sample_annotation", prev_token
            ), "sample_annotation.prev_token isn't found in sample."

    # NOTE(yukke42): There are len(t4_dataset.instance) instances for a scene.
    instance_3d = [instance for instance in t4_dataset.instance if instance.nbr_annotations != 0]
    expected_count = len(instance_3d) * len(t4_dataset.scene)
    assert (
        no_next_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.next."
    assert (
        no_prev_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.prev."
