from pathlib import Path
import uuid

import kognic.io.model as KognicModel
from kognic.io.model.scene.metadata.metadata import MetaData
import pytest

from perception_dataset.kognic.sequence_artifact import PENDING_CALIBRATION_ID
from perception_dataset.kognic.upload_dataset import KognicDatasetUploader
from perception_dataset.kognic.utils.upload_config import KognicUploadConfig

STAGING_DIR_NAME = "output_0"
EXTERNAL_ID = "my_annotation_dataset/6f947a66-6066-437a-8038-4f2af1431972"
CALIBRATION_ID = "calibration-uuid"
SCENE_UUID = "f3dc9d53-5b0e-4f7a-9d34-1e2b3c4d5e6f"


def build_artifact_scene(
    external_id: str = STAGING_DIR_NAME,
) -> KognicModel.LidarsAndCamerasSequence:
    """A scene as the converter writes it: identified by the staging directory."""
    return KognicModel.LidarsAndCamerasSequence(
        external_id=external_id,
        frames=[],
        calibration_id=PENDING_CALIBRATION_ID,
        metadata=MetaData(
            source_filename=external_id,
            dataset_id=external_id,
            inner_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, external_id)),
        ),
    )


@pytest.fixture
def created_scenes(mocker):
    """Capture the scenes handed to Kognic by ``upload_scene``."""
    uploader = KognicDatasetUploader(config=KognicUploadConfig(input_base=Path("unused")))

    mocker.patch(
        "perception_dataset.kognic.upload_dataset.load_sequence_artifact",
        side_effect=lambda _: build_artifact_scene(),
    )
    mocker.patch.object(
        KognicDatasetUploader, "_get_or_upload_calibration", return_value=CALIBRATION_ID
    )
    mocker.patch.object(KognicDatasetUploader, "_wait_for_scene_created", return_value=None)
    client = mocker.patch.object(
        KognicDatasetUploader, "kognic_io_client", new_callable=mocker.PropertyMock
    ).return_value
    client.lidars_and_cameras_sequence.create.return_value = mocker.Mock(scene_uuid=SCENE_UUID)

    def upload(external_id: str) -> KognicModel.LidarsAndCamerasSequence:
        assert uploader.upload_scene(Path("unused"), external_id) == SCENE_UUID
        return client.lidars_and_cameras_sequence.create.call_args.args[0]

    return upload


def test_upload_scene_sends_the_caller_external_id(created_scenes):
    scene = created_scenes(EXTERNAL_ID)

    assert scene.external_id == EXTERNAL_ID
    assert scene.calibration_id == CALIBRATION_ID


def test_upload_scene_sends_metadata_identity_matching_the_external_id(created_scenes):
    scene = created_scenes(EXTERNAL_ID)

    dumped = scene.metadata.model_dump(exclude_none=True)
    assert dumped["dataset_id"] == EXTERNAL_ID
    assert dumped["inner_uuid"] == str(uuid.uuid5(uuid.NAMESPACE_URL, EXTERNAL_ID))
    # The staging directory really is where the scene was read from, so it stays.
    assert dumped["source_filename"] == STAGING_DIR_NAME


def test_upload_scene_distinguishes_scenes_built_from_the_same_staging_directory(
    created_scenes,
):
    first = created_scenes("dataset_a/process_a")
    second = created_scenes("dataset_b/process_b")

    assert first.external_id != second.external_id
    assert first.metadata.inner_uuid != second.metadata.inner_uuid
