import argparse
from dataclasses import dataclass
from os import PathLike
import os.path as osp

from t4_devkit import Tier4
from t4_devkit.common import save_json, serialize_dataclasses
from t4_devkit.sanity import print_sanity_result, sanity_check
from t4_devkit.schema import SchemaName


@dataclass
class SyncContext:
    sample_map: dict[str, str]
    sample_data_map: dict[str, str]
    ego_pose_map: dict[str, str]

    @classmethod
    def from_datasets(cls, src_t4: Tier4, tgt_t4: Tier4):
        sample_map = {}  # {src_sample_token: tgt_sample_token}

        for src_sample, tgt_sample in zip(src_t4.sample, tgt_t4.sample, strict=True):
            assert (
                src_sample.timestamp == tgt_sample.timestamp
            ), "Timestamps of sample records do not match."
            sample_map[src_sample.token] = tgt_sample.token

        sample_data_map = {}  # {src_sample_data_token: tgt_sample_data_token}
        for src_sample_data, tgt_sample_data in zip(
            src_t4.sample_data, tgt_t4.sample_data, strict=True
        ):
            assert (
                src_sample_data.timestamp == tgt_sample_data.timestamp
            ), "Timestamps of sample_data records do not match."
            sample_data_map[src_sample_data.token] = tgt_sample_data.token

        ego_pose_map = {}  # {src_ego_pose_token: tgt_ego_pose_token}
        for src_ego_pose, tgt_ego_pose in zip(src_t4.ego_pose, tgt_t4.ego_pose, strict=True):
            assert (
                src_ego_pose.timestamp == tgt_ego_pose.timestamp
            ), "Timestamps of ego_pose records do not match."
            ego_pose_map[src_ego_pose.token] = tgt_ego_pose.token

        return cls(sample_map, sample_data_map, ego_pose_map)


def sync_annotations(source_path: PathLike, target_path: PathLike):
    src_t4 = Tier4(source_path, verbose=False)
    tgt_t4 = Tier4(target_path, verbose=False)
    assert len(src_t4.sample) == len(
        tgt_t4.sample
    ), "Source and target datasets must have the same number of samples."

    # 1. Sync sample annotations/object annotations/surface annotations
    sync_context = SyncContext.from_datasets(src_t4, tgt_t4)
    _sync_sample_annotations(src_t4, tgt_t4, sync_context)
    _sync_object_annotations(src_t4, tgt_t4, sync_context)
    _sync_surface_annotations(src_t4, tgt_t4, sync_context)

    # 2. Copy and save other annotation files related to instances and categories
    _copy_and_save_records(src_t4, tgt_t4, SchemaName.ATTRIBUTE)
    _copy_and_save_records(src_t4, tgt_t4, SchemaName.CATEGORY)
    _copy_and_save_records(src_t4, tgt_t4, SchemaName.INSTANCE)
    _copy_and_save_records(src_t4, tgt_t4, SchemaName.VISIBILITY)


def _sync_sample_annotations(src_t4: Tier4, tgt_t4: Tier4, context: SyncContext):
    records = []
    for record in src_t4.sample_annotation.copy():
        record.sample_token = context.sample_map[record.sample_token]
        records.append(record)

    serialized = serialize_dataclasses(records)
    save_path = osp.join(tgt_t4.annotation_dir, SchemaName.SAMPLE_ANNOTATION.filename)
    save_json(serialized, save_path)


def _sync_object_annotations(src_t4: Tier4, tgt_t4: Tier4, context: SyncContext):
    records = []
    for record in src_t4.object_ann.copy():
        record.sample_data_token = context.sample_data_map[record.sample_data_token]
        records.append(record)

    serialized = serialize_dataclasses(records)
    save_path = osp.join(tgt_t4.annotation_dir, SchemaName.OBJECT_ANN.filename)
    save_json(serialized, save_path)


def _sync_surface_annotations(src_t4: Tier4, tgt_t4: Tier4, context: SyncContext):
    records = []
    for record in src_t4.surface_ann.copy():
        record.sample_data_token = context.sample_data_map[record.sample_data_token]
        records.append(record)

    serialized = serialize_dataclasses(records)
    save_path = osp.join(tgt_t4.annotation_dir, SchemaName.SURFACE_ANN.filename)
    save_json(serialized, save_path)


def _copy_and_save_records(src_t4: Tier4, tgt_t4: Tier4, schema_name: SchemaName):
    records = src_t4.get_table(schema_name).copy()
    serialized = serialize_dataclasses(records)
    save_path = osp.join(tgt_t4.annotation_dir, schema_name.filename)
    save_json(serialized, save_path)


def main():
    parser = argparse.ArgumentParser(description="Sync annotation files between datasets.")
    parser.add_argument("source", type=str, help="Path to the source dataset.")
    parser.add_argument("targets", nargs="+", help="Path to the target datasets.")
    args = parser.parse_args()

    source_path = args.source
    target_paths = args.targets

    for target_path in target_paths:
        sync_annotations(source_path, target_path)

        # run sanity check
        result = sanity_check(target_path)
        if not result.is_passed():
            print_sanity_result(result)
            raise RuntimeError(f"Sanity check failed for dataset at {target_path}")


if __name__ == "__main__":
    main()
