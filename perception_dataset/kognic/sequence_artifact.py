"""Serialization helpers for converted Kognic scene artifacts."""

import json
from pathlib import Path
from typing import Generator

import kognic.io.model as KognicModel

SEQUENCE_ARTIFACT_FILENAME = "lidars_and_cameras_sequence.json"
PENDING_CALIBRATION_ID = "pending-calibration-upload"


def _iter_resources(payload: dict) -> Generator[dict, None, None]:
    """Yield file-backed resources from a serialized sequence.

    Args:
        payload (dict): The serialized sequence payload.

    Yields:
        Generator[dict, None, None]: Each file-backed resource dictionary.
    """
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for field_name in ("point_clouds", "images"):
            resources = frame.get(field_name, [])
            if isinstance(resources, list):
                yield from resources


def save_sequence_artifact(
    sequence_path: Path,
    sequence: KognicModel.LidarsAndCamerasSequence,
) -> Path:
    """Persist a validated sequence with relocatable resource paths.

    Resource filenames, client filenames, and resource IDs are stored relative
    to the scene directory so the complete staging directory can be moved
    without invalidating the artifact.

    Args:
        sequence_path (Path): Staging directory that owns the sequence artifact
            and all referenced sensor files.
        sequence (KognicModel.LidarsAndCamerasSequence): Validated sequence
            model to serialize.

    Returns:
        Path: Path to the written ``lidars_and_cameras_sequence.json`` artifact.

    Raises:
        ValueError: If a referenced sensor resource is outside
            ``sequence_path`` and therefore cannot be stored as a relocatable
            path.
    """
    payload = sequence.model_dump(mode="json")
    sequence_root = sequence_path.resolve()
    for resource in _iter_resources(payload):
        resource_path = Path(resource["filename"]).resolve()
        try:
            relative_path = resource_path.relative_to(sequence_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Staged resource {resource_path} is outside {sequence_path}"
            ) from exc
        resource["filename"] = relative_path
        resource["client_filename"] = relative_path
        resource["resource_id"] = relative_path

    artifact_path = sequence_path / SEQUENCE_ARTIFACT_FILENAME
    artifact_path.write_text(json.dumps(payload, indent=2))
    return artifact_path


def load_sequence_artifact(sequence_path: Path) -> KognicModel.LidarsAndCamerasSequence:
    """Resolve resource paths and Pydantically reload a converted sequence.

    Relative resource filenames are resolved against the staging directory
    before the payload is passed to the Kognic Pydantic model. This lets the
    model validate both the JSON structure and the referenced local files.

    Args:
        sequence_path (Path): Staging directory containing
            ``lidars_and_cameras_sequence.json`` and its sensor resources.

    Returns:
        KognicModel.LidarsAndCamerasSequence: Fully validated sequence ready
            for calibration binding and upload.

    Raises:
        FileNotFoundError: If the sequence artifact or a referenced resource
            does not exist.
        json.JSONDecodeError: If the artifact is not valid JSON.
        pydantic.ValidationError: If the artifact does not satisfy the Kognic
            sequence schema.
    """
    payload = json.loads((sequence_path / SEQUENCE_ARTIFACT_FILENAME).read_text())
    for resource in _iter_resources(payload):
        resource_path = Path(resource["filename"])
        if not resource_path.is_absolute():
            resource["filename"] = str((sequence_path / resource_path).resolve())
    return KognicModel.LidarsAndCamerasSequence.model_validate(payload)
