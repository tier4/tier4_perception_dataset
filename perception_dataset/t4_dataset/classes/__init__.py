from .attribute import AttributeRecord, AttributeTable  # noqa: F401, F403
from .calibrated_sensor import CalibratedSensorRecord, CalibratedSensorTable  # noqa: F401, F403
from .category import CategoryRecord, CategoryTable  # noqa: F401, F403
from .ego_pose import EgoPoseRecord, EgoPoseTable  # noqa: F401, F403
from .instance import InstanceRecord, InstanceTable  # noqa: F401, F403
from .log import LogRecord, LogTable  # noqa: F401, F403
from .map import MapRecord, MapTable  # noqa: F401, F403
from .object_ann import ObjectAnnRecord, ObjectAnnTable  # noqa: F401, F403
from .sample import SampleRecord, SampleTable  # noqa: F401, F403
from .sample_annotation import SampleAnnotationRecord, SampleAnnotationTable  # noqa: F401, F403
from .sample_data import SampleDataRecord, SampleDataTable  # noqa: F401, F403
from .scene import SceneRecord, SceneTable  # noqa: F401, F403
from .sensor import SensorRecord, SensorTable  # noqa: F401, F403
from .surface_ann import SurfaceAnnRecord, SurfaceAnnTable  # noqa: F401, F403
from .visibility import VisibilityRecord, VisibilityTable  # noqa: F401, F403

schema_names = [
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
]
