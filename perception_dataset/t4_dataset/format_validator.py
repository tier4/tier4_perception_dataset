from pathlib import Path

from loguru import logger
from nuscenes.nuscenes import NuScenes

from perception_dataset.constants import EXTENSION_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.classes import schema_names


def validate_format(nusc: NuScenes, root_path: str):
    validate_scene(nusc)
    validate_sample(nusc)
    validate_sample_data(nusc, Path(root_path))
    validate_ego_pose(nusc)
    validate_calibrated_sensor(nusc)
    validate_instance(nusc)
    validate_sample_annotation(nusc)


def find_in_table(nusc: NuScenes, table_name: str, token: str) -> bool:
    """This function is the modification of `get` of NuScenes not to raise an error.
    https://github.com/nutonomy/nuscenes-devkit/blob/28765b8477dbd3331bacd922fada867c2c4db1d7/python-sdk/nuscenes/nuscenes.py#L207-L225
    """
    assert table_name in nusc._token2ind, f"{table_name} not found"

    token_ind = nusc._token2ind[table_name].get(token)
    if token_ind is None:
        return False

    table = getattr(nusc, table_name)
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
def validate_scene(nusc: NuScenes):
    assert len(nusc.scene) == 1, "T4Dataset must have 1 scene."

    scene = nusc.scene[0]
    assert find_in_table(nusc, "log", scene["log_token"]), "scene.log_token isn't found in log"
    assert find_in_table(
        nusc, "sample", scene["first_sample_token"]
    ), "scene.first_sample_token isn't found in log"
    assert find_in_table(
        nusc, "sample", scene["last_sample_token"]
    ), "scene.last_sample_token isn't found in log"


@_logger_wrapper
def validate_sample(nusc: NuScenes):
    assert len(nusc.sample) > 0, "There are no sample."

    no_next_token_count: int = 0
    no_prev_token_count: int = 0
    for sample in nusc.sample:
        assert find_in_table(
            nusc, "scene", sample["scene_token"]
        ), "sample.scene_token isn't found in scene."

        next_token = sample["next"]
        if next_token == "":
            no_next_token_count += 1
        else:
            assert find_in_table(nusc, "sample", next_token), "sample.next isn't found in sample."

        prev_token = sample["prev"]
        if prev_token == "":
            no_prev_token_count += 1
        else:
            assert find_in_table(
                nusc, "sample", prev_token
            ), "sample.prev_token isn't found in sample."

    assert no_next_token_count == len(
        nusc.scene
    ), f"There are more than {len(nusc.scene)} sample of empty scene.next."
    assert no_prev_token_count == len(
        nusc.scene
    ), f"There are more than {len(nusc.scene)} sample of empty scene.prev."


@_logger_wrapper
def validate_sample_data(nusc: NuScenes, root_path: Path):
    assert len(nusc.sample_data), "There are no sample_data."

    no_next_token_count: int = 0
    no_prev_token_count: int = 0
    for sample_data in nusc.sample_data:
        if not sample_data["is_valid"]:
            continue
        assert find_in_table(
            nusc, "sample", sample_data["sample_token"]
        ), "sample_data.sample_token isn't found in sample."
        assert find_in_table(
            nusc, "ego_pose", sample_data["ego_pose_token"]
        ), "sample_data.ego_pose_token isn't found in sample."
        assert find_in_table(
            nusc, "calibrated_sensor", sample_data["calibrated_sensor_token"]
        ), "sample_data.calibrated_pose_token isn't found in sample."

        filename: str = sample_data["filename"]
        assert (root_path / filename).is_file(), f"{filename} isn't found."

        next_token = sample_data["next"]
        if next_token == "":
            no_next_token_count += 1
        else:
            assert find_in_table(
                nusc, "sample_data", next_token
            ), "sample_data.next isn't found in sample."

        prev_token = sample_data["prev"]
        if prev_token == "":
            no_prev_token_count += 1
        else:
            assert find_in_table(
                nusc, "sample_data", prev_token
            ), "sample_data.prev_token isn't found in sample."

    # NOTE(yukke42): There are len(nusc.calibrated_sensor) sensors for each scene.
    expected_count = len(nusc.calibrated_sensor) * len(nusc.scene)
    assert (
        no_next_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.next."
    assert (
        no_prev_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.prev."


@_logger_wrapper
def validate_ego_pose(nusc: NuScenes):
    assert len(nusc.ego_pose) > 0, "There are no ego_pose."


@_logger_wrapper
def validate_calibrated_sensor(nusc: NuScenes):
    assert len(nusc.calibrated_sensor) > 0, "There are no calibrated_sensor."

    for calibrated_sensor in nusc.calibrated_sensor:
        assert find_in_table(
            nusc, "sensor", calibrated_sensor["sensor_token"]
        ), "calibrated_sensor.sensor_token isn't found in sensor."


@_logger_wrapper
def validate_instance(nusc: NuScenes):
    assert len(nusc.instance) > 0, "There are no instance."

    for instance in nusc.instance:
        if instance["nbr_annotations"] == 0:
            logger.warning(f"instance:{instance['token']} has no 3D annotation")
            continue
        assert find_in_table(
            nusc, "category", instance["category_token"]
        ), "instance.category_token isn't found in category."
        assert find_in_table(
            nusc, "sample_annotation", instance["first_annotation_token"]
        ), "instance.first_annotation_token isn't found in sample_annotation."
        assert find_in_table(
            nusc, "sample_annotation", instance["last_annotation_token"]
        ), "instance.last_annotation_token isn't found in sample_annotation."


@_logger_wrapper
def validate_sample_annotation(nusc: NuScenes):
    assert len(nusc.sample_annotation) > 0, "There are no sample_annotation."

    no_next_token_count: int = 0
    no_prev_token_count: int = 0
    for sample_annotation in nusc.sample_annotation:
        assert find_in_table(
            nusc, "sample", sample_annotation["sample_token"]
        ), "sample_annotation.sample_token isn't found in sample."
        assert find_in_table(
            nusc, "instance", sample_annotation["instance_token"]
        ), "sample_annotation.instance_token isn't found in instance."

        for i, attribute_token in enumerate(sample_annotation["attribute_tokens"]):
            assert find_in_table(
                nusc, "attribute", attribute_token
            ), f"sample_annotation.attribute_tokens[{i}] isn't found in attribute."
        assert find_in_table(
            nusc, "visibility", sample_annotation["visibility_token"]
        ), "sample_annotation.visibility_token isn't found in visibility."

        next_token = sample_annotation["next"]
        if next_token == "":
            no_next_token_count += 1
        else:
            assert find_in_table(
                nusc, "sample_annotation", next_token
            ), "sample_annotation.next isn't found in sample."

        prev_token = sample_annotation["prev"]
        if prev_token == "":
            no_prev_token_count += 1
        else:
            assert find_in_table(
                nusc, "sample_annotation", prev_token
            ), "sample_annotation.prev_token isn't found in sample."

    # NOTE(yukke42): There are len(nusc.instance) instances for a scene.
    instance_3d = [instance for instance in nusc.instance if instance["nbr_annotations"] != 0]
    expected_count = len(instance_3d) * len(nusc.scene)
    assert (
        no_next_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.next."
    assert (
        no_prev_token_count == expected_count
    ), f"There are more than {expected_count} sample of empty scene.prev."
