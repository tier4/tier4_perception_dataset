import os.path as osp
import time
from typing import Any

import numpy as np
from nuscenes.utils.data_classes import Box as Box3D
from pyquaternion import Quaternion
from t4_devkit.common import load_json
from t4_devkit.schema import SchemaName, SensorModality, VisibilityLevel
from t4_devkit.typing import NDArrayF64
from t4_devkit.utils import is_box_in_image

__all__ = ("Tier4",)

# TODO: add support of retrieving 2D boxes.


class Tier4:
    """Database class for T4 dataset to help query and retrieve information from the database."""

    def __init__(self, version: str, data_root: str, verbose: bool = True) -> None:
        """Load database and creates reverse indexes and shortcuts.

        Args:
        ----
            version (str): Directory name of database json files.
            data_root (str): Path to the root directory of dataset.
            verbose (bool, optional): Whether to display status during load. Defaults to True.

        Examples:
        --------
            >>> t4 = Tier4("annotation", "data/tier4")
            ======
            Loading T4 tables in `annotation`...
            Reverse indexing...
            Done reverse indexing in 0.010 seconds.
            ======
            21 category
            8 attribute
            4 visibility
            31 instance
            7 sensor
            7 calibrated_sensor
            2529 ego_pose
            1 log
            1 scene
            88 sample
            2529 sample_data
            1919 sample_annotation
            0 object_ann
            0 surface_ann
            0 keypoint
            1 map
            Done loading in 0.046 seconds.
            ======

        """
        self.version = version
        self.data_root = data_root
        self.verbose = verbose

        assert osp.exists(self.data_root), f"Database directory is not found: {self.data_root}"

        start_time = time.time()
        if verbose:
            print("======\nLoading T4 tables in `{}`...".format(self.version))

        # assign tables explicitly
        self.attribute = self.__load_table__(SchemaName.ATTRIBUTE)
        self.calibrated_sensor = self.__load_table__(SchemaName.CALIBRATED_SENSOR)
        self.category = self.__load_table__(SchemaName.CATEGORY)
        self.ego_pose = self.__load_table__(SchemaName.EGO_POSE)
        self.instance = self.__load_table__(SchemaName.INSTANCE)
        self.keypoint = self.__load_table__(SchemaName.KEYPOINT)
        self.log = self.__load_table__(SchemaName.LOG)
        self.map = self.__load_table__(SchemaName.MAP)
        self.object_ann = self.__load_table__(SchemaName.OBJECT_ANN)
        self.sample_annotation = self.__load_table__(SchemaName.SAMPLE_ANNOTATION)
        self.sample_data = self.__load_table__(SchemaName.SAMPLE_DATA)
        self.sample = self.__load_table__(SchemaName.SAMPLE)
        self.scene = self.__load_table__(SchemaName.SCENE)
        self.sensor = self.__load_table__(SchemaName.SENSOR)
        self.surface_ann = self.__load_table__(SchemaName.SURFACE_ANN)
        self.visibility = self.__load_table__(SchemaName.VISIBILITY)

        # make reverse indexes for common lookups
        self.__make_reverse_index__(verbose)

        if verbose:
            for schema in SchemaName:
                print(f"{len(getattr(self, schema.value))} {schema.value}")
            elapsed_time = time.time() - start_time
            print(f"Done loading in {elapsed_time:.3f} seconds.\n======")

    def __load_table__(self, schema: SchemaName) -> dict[str, Any]:
        """Load schema table from a json file. If the schema is optional and
        there is no corresponding folder is not contained in dataset, returns empty dict.

        TODO: Return defined dataclass instead of dict.

        Args:
        ----
            schema (SchemaName): An enum member of `SchemaName`.

        Returns:
        -------
            dict[str, Any]: Loaded table data saved in `.json`.
        """
        filepath = osp.join(self.data_root, self.version, f"{schema.value}.json")
        if not osp.exists(filepath) and schema.is_optional():
            return {}
        assert osp.exists(filepath), f"{schema.value} is mandatory."
        return load_json(filepath)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """De-normalize database to create reverse indices for common cases.

        Args:
        ----
            verbose (bool): Whether to display outputs.

        Raises:
        ------
            ValueError: Expecting `map` table has `log_tokens` key.
        """
        start_time = time.time()
        if verbose:
            print("Reverse indexing...")

        self._token2idx: dict[str, Any] = {
            schema.value: {
                member["token"]: idx for idx, member in enumerate(getattr(self, schema.value))
            }
            for schema in SchemaName
        }

        for record in self.sample_annotation:
            instance = self.get("instance", record["instance_token"])
            record["category_name"] = self.get("category", instance["category_token"])["name"]

        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        for record in self.sample:
            record["data"] = {}
            record["anns"] = []

        for record in self.sample_data:
            if record["is_key_frame"]:
                sample_record = self.get("sample", record["sample_token"])
                sample_record["data"][record["channel"]] = record["token"]

        for ann_record in self.sample_annotation:
            sample_record = self.get("sample", ann_record["sample_token"])
            sample_record["anns"].append(ann_record["token"])

        if "log_tokens" not in self.map[0].keys():
            raise ValueError(
                "Error: log_tokens not in map table. This code is not compatible with the teaser dataset."
            )
        log_to_map = {}
        for map_record in self.map:
            for log_token in map_record["log_tokens"]:
                log_to_map[log_token] = map_record["token"]
        for log_record in self.log:
            log_record["map_token"] = log_to_map[log_record["token"]]

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Done reverse indexing in {elapsed_time:.3f} seconds.\n======")

    def get(self, schema: str | SchemaName, token: str) -> dict[str, Any]:
        """Return a record identified by the associated token.

        Args:
        ----
            schema (str | SchemaName): Name of schema.
            token (str): Token to identify the specific record.

        Returns:
        -------
            dict[str, Any]: Table record.
        """
        if isinstance(schema, str):
            schema = SchemaName.from_str(schema)
        schema_name: str = schema.value

        assert schema in SchemaName, f"Schema {schema} not found."

        return getattr(self, schema_name)[self.get_idx(schema_name, token)]

    def get_idx(self, schema: str | SchemaName, token: str) -> int:
        """Return the index of the record in a table in constant runtime.

        Args:
        ----
            schema (str | SchemaName): Name of schema.
            token (str): Token of record.

        Returns:
        -------
            int: The index of the record in table.
        """
        if isinstance(schema, SchemaName):
            schema = schema.value
        return self._token2idx[schema][token]

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """Return the file path to a raw data recorded in `sample_data`.

        Args:
        ----
            sample_data_token (str): Token of `sample_data`.

        Returns:
        -------
            str: File path.
        """
        sd_record = self.get("sample_data", sample_data_token)
        return osp.join(self.data_root, sd_record["filename"])

    def get_sample_data(
        self,
        sample_data_token: str,
        selected_ann_tokens: list[str] | None = None,
        as_3d: bool = True,
        visibility: VisibilityLevel = VisibilityLevel.NONE,
    ) -> tuple[str, list[Box3D], NDArrayF64 | None]:
        """Return the data path as well as all annotations related to that `sample_data`.

        # TODO: add support of retrieving 2D boxes.

        Args:
        ----
            sample_data_token (str): Token of `sample_data`.
            selected_ann_tokens (list[str] | None, optional)
            as_3d (bool, optional): Whether to return 3D or 2D boxes. Defaults to True.
            visibility (VisibilityLevel, optional): If `sample_data` is an image,
                this sets required visibility for only 3D boxes.

        Returns:
        -------
            tuple[str, list[Box3D], NDArrayF64 | None]: Data path, a list of boxes and 3x3 camera intrinsic matrix.
        """
        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        cs_record = self.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = self.get("sensor", cs_record["sensor_token"])
        pose_record = self.get("ego_pose", sd_record["ego_pose_token"])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record["modality"] == SensorModality.CAMERA:
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            img_size = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            img_size = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        boxes: list[Box3D]
        if selected_ann_tokens is not None:
            boxes = list(map(self.get_box3d, selected_ann_tokens)) if as_3d else None
        else:
            boxes = self.get_boxes3d(sample_data_token) if as_3d else None

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if as_3d:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)

                if sensor_record["modality"] == SensorModality.CAMERA and not is_box_in_image(
                    box,
                    cam_intrinsic,
                    img_size,
                    visibility=visibility,
                ):
                    continue
            else:
                raise NotImplementedError("we've not supported 2D boxes yet.")

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box3d(self, sample_annotation_token: str) -> Box3D:
        """Return a Box3D class from a `sample_annotation` record.

        Args:
        ----
            sample_annotation_token (str): Token of `sample_annotation`.

        Returns:
        -------
            Box3D: Instantiated Box3D.
        """
        record = self.get("sample_annotation", sample_annotation_token)
        return Box3D(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=record["category_name"],
            token=record["token"],
        )

    def get_boxes3d(self, sample_data_token: str) -> list[Box3D]:
        """Rerun a list of Box3D classes for all annotations of a particular `sample_data` record.
        It the `sample_data` is a keyframe, this returns annotations for the corresponding `sample`.

        Args:
        ----
            sample_data_token (str): Token of `sample_data`.

        Returns:
        -------
            list[Box3D]: List of instantiated Box3D classes.
        """
        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sd_record["sample_token"])

        if curr_sample_record["prev"] == "" or sd_record["is_key_frame"]:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record["anns"]))

        else:
            prev_sample_record = self.get("sample", curr_sample_record["prev"])

            curr_ann_recs = [
                self.get("sample_annotation", token) for token in curr_sample_record["anns"]
            ]
            prev_ann_recs = [
                self.get("sample_annotation", token) for token in prev_sample_record["anns"]
            ]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry["instance_token"]: entry for entry in prev_ann_recs}

            t0 = prev_sample_record["timestamp"]
            t1 = curr_sample_record["timestamp"]
            t = sd_record["timestamp"]

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec["instance_token"] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec["instance_token"]]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(prev_ann_rec["translation"], curr_ann_rec["translation"])
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(
                        q0=Quaternion(prev_ann_rec["rotation"]),
                        q1=Quaternion(curr_ann_rec["rotation"]),
                        amount=(t - t0) / (t1 - t0),
                    )

                    box = Box3D(
                        center,
                        curr_ann_rec["size"],
                        rotation,
                        name=curr_ann_rec["category_name"],
                        token=curr_ann_rec["token"],
                    )
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec["token"])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> NDArrayF64:
        """Return the velocity of an annotation.
        If corresponding annotation has a true velocity, this returns it.
        Otherwise, this estimates the velocity by computing the difference
        between the previous and next frame.
        If it is failed to estimate the velocity, values are set to np.nan.

        Args:
        ----
            sample_annotation_token (str): Token of `sample_annotation`.
            max_time_diff (float, optional): Max allowed time difference
                between consecutive samples. Defaults to 1.5.

        Returns:
        -------
            NDArrayF64: Velocity in the order of (vx, vy, vz) in m/s.
        """
        current = self.get("sample_annotation", sample_annotation_token)

        # If the real velocity is annotated, returns it
        if current.get("velocity") is not None:
            return current["velocity"]

        has_prev = current["prev"] != ""
        has_next = current["next"] != ""

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get("sample_annotation", current["prev"])
        else:
            first = current

        if has_next:
            last = self.get("sample_annotation", current["next"])
        else:
            last = current

        pos_last = np.array(last["translation"])
        pos_first = np.array(first["translation"])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get("sample", last["sample_token"])["timestamp"]
        time_first = 1e-6 * self.get("sample", first["sample_token"])["timestamp"]
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff
