import json

from autoware_vehicle_msgs.msg import VelocityReport
from builtin_interfaces.msg import Time
from t4_devkit.sanity import sanity_check
from t4_devkit.schema.v1 import SchemaName
from t4_devkit.schema.v1.migrate_v0 import V0Migrator

from perception_dataset.t4_v1.backfill import Backfill
from perception_dataset.t4_v1.vehicle_table import VehicleTable
from tests.constants import TEST_DATA_ROOT_DIR, TEST_ROOT_DIR
from tests.t4_v1.bags import write_bag

V0_DATASET = TEST_DATA_ROOT_DIR / "t4_sample_0" / "non_annotated_dataset_tlr" / "sample_bag"
SAMPLE_TABLE = TEST_ROOT_DIR.parent / "config" / "vehicle_info" / "vehicles_sample.yaml"


def test_backfill(tmp_path):
    first_sample = min(
        sample.timestamp
        for sample in V0Migrator(V0_DATASET, tmp_path / "unused").tables(SchemaName.SAMPLE)
    )
    sec, usec = divmod(first_sample, 1_000_000)
    velocity = VelocityReport(longitudinal_velocity=2.0)
    velocity.header.stamp = Time(sec=sec, nanosec=usec * 1000)
    write_bag(
        tmp_path / "bag",
        [
            (
                "/vehicle/status/velocity_status",
                "autoware_vehicle_msgs/msg/VelocityReport",
                velocity,
                sec,
            )
        ],
    )
    metadata = {
        "vehicle_name": "sample_vehicle_01",
        "project_id": "sample_project",
        "map": {"map_id": "1", "version_id": "1-0"},
    }
    output_root = tmp_path / "t4dataset" / "1"
    backfill = Backfill(
        str(V0_DATASET),
        str(output_root),
        metadata,
        VehicleTable.load(SAMPLE_TABLE),
        [str(tmp_path / "bag")],
    )
    backfill.run("t4dataset", 1)

    def read(table):
        return json.loads((output_root / "annotation" / f"{table}.json").read_text())

    (vehicle,) = read("vehicle")
    assert (vehicle["name"], vehicle["model"], vehicle["length"]) == (
        "sample_vehicle_01",
        "sample_vehicle",
        4.89,
    )
    # one sample is within 100 ms of the velocity report
    assert [state["longitudinal_velocity"] for state in read("vehicle_state")] == [2.0]
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["project_id"] == "sample_project"
    assert "route: none added, the dataset has no map" in manifest["migration_notes"]
    reports = sanity_check(output_root.parent.as_posix(), revision="1").reports
    assert not [report.id for report in reports if report.is_failed()]
