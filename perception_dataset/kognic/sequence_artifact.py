"""Serialization helpers for converted Kognic scene artifacts."""

import json
from pathlib import Path
from typing import Generator

import kognic.io.model as KognicModel

SEQUENCE_ARTIFACT_FILENAME = "lidars_and_cameras_sequence.json"
PENDING_CALIBRATION_ID = "pending-calibration-upload"


def _iter_resources(payload: dict) -> Generator[dict, None, None]:
    """Yield file-backed resources from a serialized sequence."""
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
    """Persist a validated sequence with relocatable resource paths."""
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
    """Resolve resource paths and Pydantically reload a converted sequence."""
    payload = json.loads((sequence_path / SEQUENCE_ARTIFACT_FILENAME).read_text())
    for resource in _iter_resources(payload):
        resource_path = Path(resource["filename"])
        if not resource_path.is_absolute():
            resource["filename"] = str((sequence_path / resource_path).resolve())
    return KognicModel.LidarsAndCamerasSequence.model_validate(payload)
