from __future__ import annotations

import os.path as osp
import time
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
from nuscenes.nuscenes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
import rerun as rr
import rerun.blueprint as rrb
from t4_devkit.common.box import Box2D, Box3D
from t4_devkit.common.geometry import is_box_in_image
from t4_devkit.common.timestamp import sec2us, us2sec
from t4_devkit.schema import SchemaName, SensorModality, VisibilityLevel, build_schema

if TYPE_CHECKING:
    from t4_devkit.typing import CamIntrinsicType, SizeType, TranslationType, VelocityType

    from .schema import (
        Attribute,
        CalibratedSensor,
        Category,
        EgoPose,
        Instance,
        Keypoint,
        Log,
        Map,
        ObjectAnn,
        Sample,
        SampleAnnotation,
        SampleData,
        Scene,
        SchemaTable,
        Sensor,
        SurfaceAnn,
        Visibility,
    )

__all__ = ("Tier4",)

# currently need to calculate the color manually
# see https://github.com/rerun-io/rerun/issues/4409
COLOR_MAP = matplotlib.colormaps["turbo_r"]
COLOR_NORM = matplotlib.colors.Normalize(
    vmin=3.0,
    vmax=75.0,
)


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

        if not osp.exists(self.data_root):
            raise FileNotFoundError(f"Database directory is not found: {self.data_root}")

        start_time = time.time()
        if verbose:
            print(f"======\nLoading T4 tables in `{self.version}`...")

        # assign tables explicitly
        self.attribute: list[Attribute] = self.__load_table__(SchemaName.ATTRIBUTE)
        self.calibrated_sensor: list[CalibratedSensor] = self.__load_table__(
            SchemaName.CALIBRATED_SENSOR
        )
        self.category: list[Category] = self.__load_table__(SchemaName.CATEGORY)
        self.ego_pose: list[EgoPose] = self.__load_table__(SchemaName.EGO_POSE)
        self.instance: list[Instance] = self.__load_table__(SchemaName.INSTANCE)
        self.keypoint: list[Keypoint] = self.__load_table__(SchemaName.KEYPOINT)
        self.log: list[Log] = self.__load_table__(SchemaName.LOG)
        self.map: list[Map] = self.__load_table__(SchemaName.MAP)
        self.object_ann: list[ObjectAnn] = self.__load_table__(SchemaName.OBJECT_ANN)
        self.sample_annotation: list[SampleAnnotation] = self.__load_table__(
            SchemaName.SAMPLE_ANNOTATION
        )
        self.sample_data: list[SampleData] = self.__load_table__(SchemaName.SAMPLE_DATA)
        self.sample: list[Sample] = self.__load_table__(SchemaName.SAMPLE)
        self.scene: list[Scene] = self.__load_table__(SchemaName.SCENE)
        self.sensor: list[Sensor] = self.__load_table__(SchemaName.SENSOR)
        self.surface_ann: list[SurfaceAnn] = self.__load_table__(SchemaName.SURFACE_ANN)
        self.visibility: list[Visibility] = self.__load_table__(SchemaName.VISIBILITY)

        # make reverse indexes for common lookups
        self.__make_reverse_index__(verbose)

        if verbose:
            for schema in SchemaName:
                print(f"{len(self.get_table(schema))} {schema.value}")
            elapsed_time = time.time() - start_time
            print(f"Done loading in {elapsed_time:.3f} seconds.\n======")

    def __load_table__(self, schema: SchemaName) -> list[SchemaTable]:
        """Load schema table from a json file.

        If the schema is optional and there is no corresponding json file in dataset,
        returns empty list.

        Args:
        ----
            schema (SchemaName): An enum member of `SchemaName`.

        Returns:
        -------
            list[SchemaTable]: Loaded table data saved in `.json`.
        """
        filepath = osp.join(self.data_root, self.version, schema.filename)
        if not osp.exists(filepath) and schema.is_optional():
            return []

        if not osp.exists(filepath):
            raise FileNotFoundError(f"{schema.value} is mandatory.")

        return build_schema(schema, filepath)

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

        token2idx: dict[str, dict[str, int]] = {}
        for schema in SchemaName:
            token2idx[schema.value] = {}
            for idx, table in enumerate(self.get_table(schema.value)):
                table: SchemaTable
                token2idx[schema.value][table.token] = idx
        self._token2idx = token2idx

        self._label2id: dict[str, int] = {
            category.name: idx for idx, category in enumerate(self.category)
        }

        # add shortcuts
        for record in self.sample_annotation:
            instance: Instance = self.get("instance", record.instance_token)
            category: Category = self.get("category", instance.category_token)
            record.category_name = category.name

        for record in self.object_ann:
            instance: Instance = self.get("instance", record.instance_token)
            category: Category = self.get("category", instance.category_token)
            record.category_name = category.name

        for record in self.sample_data:
            cs_record: CalibratedSensor = self.get(
                "calibrated_sensor", record.calibrated_sensor_token
            )
            sensor_record: Sensor = self.get("sensor", cs_record.sensor_token)
            record.modality = sensor_record.modality
            record.channel = sensor_record.channel

        for record in self.sample_data:
            if record.is_key_frame:
                sample_record: Sample = self.get("sample", record.sample_token)
                sample_record.data[record.channel] = record.token

        for ann_record in self.sample_annotation:
            sample_record: Sample = self.get("sample", ann_record.sample_token)
            sample_record.ann_3ds.append(ann_record.token)

        for ann_record in self.object_ann:
            sd_record: SampleData = self.get("sample_data", ann_record.sample_data_token)
            sample_record: Sample = self.get("sample", sd_record.sample_token)
            sample_record.ann_2ds.append(ann_record.token)

        log_to_map: dict[str, str] = {}
        for map_record in self.map:
            for log_token in map_record.log_tokens:
                log_to_map[log_token] = map_record.token
        for log_record in self.log:
            log_record.map_token = log_to_map[log_record.token]

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Done reverse indexing in {elapsed_time:.3f} seconds.\n======")

    def get_table(self, schema: str | SchemaName) -> list[SchemaTable]:
        """Return the list of dataclasses of the corresponding schema table.

        Args:
        -----
            schema (str | SchemaName): Name of schema table.

        Returns:
        -------
            list[SchemaTable]: List of dataclasses.
        """
        if isinstance(schema, SchemaName):
            schema = schema.value
        return getattr(self, schema)

    def get(self, schema: str | SchemaName, token: str) -> SchemaTable:
        """Return a record identified by the associated token.

        Args:
        ----
            schema (str | SchemaName): Name of schema.
            token (str): Token to identify the specific record.

        Returns:
        -------
            SchemaTable: Table record of the corresponding token.
        """
        return self.get_table(schema)[self.get_idx(schema, token)]

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
        if self._token2idx.get(schema) is None:
            raise KeyError(f"{schema} is not registered.")
        if self._token2idx[schema].get(token) is None:
            raise KeyError(f"{token} is not registered in {schema}.")
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
        sd_record: SampleData = self.get("sample_data", sample_data_token)
        return osp.join(self.data_root, sd_record.filename)

    def get_sample_data(
        self,
        sample_data_token: str,
        selected_ann_tokens: list[str] | None = None,
        *,
        as_3d: bool = True,
        visibility: VisibilityLevel = VisibilityLevel.NONE,
    ) -> tuple[str, list[Box3D | Box2D], CamIntrinsicType | None]:
        """Return the data path as well as all annotations related to that `sample_data`.

        Args:
        ----
            sample_data_token (str): Token of `sample_data`.
            selected_ann_tokens (list[str] | None, optional)
            as_3d (bool, optional): Whether to return 3D or 2D boxes. Defaults to True.
            visibility (VisibilityLevel, optional): If `sample_data` is an image,
                this sets required visibility for only 3D boxes.

        Returns:
        -------
            tuple[str, list[Box3D | Box2D], CamIntrinsicType | None]: Data path, a list of boxes and 3x3 camera intrinsic matrix.
        """
        # Retrieve sensor & pose records
        sd_record: SampleData = self.get("sample_data", sample_data_token)
        cs_record: CalibratedSensor = self.get(
            "calibrated_sensor", sd_record.calibrated_sensor_token
        )
        sensor_record: Sensor = self.get("sensor", cs_record.sensor_token)
        pose_record: EgoPose = self.get("ego_pose", sd_record.ego_pose_token)

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record.modality == SensorModality.CAMERA:
            cam_intrinsic = cs_record.camera_intrinsic
            img_size = (sd_record.width, sd_record.height)
        else:
            cam_intrinsic = None
            img_size = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        boxes: list[Box3D | Box2D]
        if selected_ann_tokens is not None:
            boxes = (
                list(map(self.get_box3d, selected_ann_tokens))
                if as_3d
                else list(map(self.get_box2d, selected_ann_tokens))
            )
        else:
            boxes = (
                self.get_boxes3d(sample_data_token)
                if as_3d
                else self.get_boxes2d(sample_data_token)
            )

        if not as_3d:
            return data_path, boxes, cam_intrinsic

        # Make list of Box objects including coord system transforms.
        box_list: list[Box3D] = []
        for box in boxes:
            # Move box to ego vehicle coord system.
            box.translate(-pose_record.translation)
            box.rotate(pose_record.rotation.inverse)

            #  Move box to sensor coord system.
            box.translate(-cs_record.translation)
            box.rotate(cs_record.rotation.inverse)

            if sensor_record.modality == SensorModality.CAMERA and not is_box_in_image(
                box,
                cam_intrinsic,
                img_size,
                visibility=visibility,
            ):
                continue
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
        record: SampleAnnotation = self.get("sample_annotation", sample_annotation_token)
        return Box3D(
            record.translation,
            record.size,
            record.rotation,
            name=record.category_name,
            token=record.token,
        )

    def get_box2d(self, object_ann_token: str) -> Box2D:
        """Return a Box2D class from a `object_ann` record.

        Args:
        ----
            object_ann_token (str): Token of `object_ann`.

        Returns:
        -------
            Box2D: Instantiated Box2D.
        """
        record: ObjectAnn = self.get("object_ann", object_ann_token)
        return Box2D(record.bbox, name=record.category_name, token=record.token)

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
        sd_record: SampleData = self.get("sample_data", sample_data_token)
        curr_sample_record: Sample = self.get("sample", sd_record.sample_token)

        if curr_sample_record.prev == "" or sd_record.is_key_frame:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box3d, curr_sample_record.ann_3ds))

        else:
            prev_sample_record: Sample = self.get("sample", curr_sample_record.prev)

            curr_ann_recs: list[SampleAnnotation] = [
                self.get("sample_annotation", token) for token in curr_sample_record.ann_3ds
            ]
            prev_ann_recs: list[SampleAnnotation] = [
                self.get("sample_annotation", token) for token in prev_sample_record.ann_3ds
            ]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry.instance_token: entry for entry in prev_ann_recs}

            t0 = prev_sample_record.timestamp
            t1 = curr_sample_record.timestamp
            t = sd_record.timestamp

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec.instance_token in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec.instance_token]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(
                            prev_ann_rec.translation,
                            curr_ann_rec.translation,
                            strict=True,
                        )
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(
                        q0=prev_ann_rec.rotation,
                        q1=curr_ann_rec.rotation,
                        amount=(t - t0) / (t1 - t0),
                    )

                    box = Box3D(
                        center,
                        curr_ann_rec.size,
                        rotation,
                        name=curr_ann_rec.category_name,
                        token=curr_ann_rec.token,
                    )
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box3d(curr_ann_rec.token)

                boxes.append(box)
        return boxes

    def get_boxes2d(self, sample_data_token: str) -> list[Box2D]:
        """Rerun a list of Box2D classes for all annotations of a particular `sample_data` record.
        It the `sample_data` is a keyframe, this returns annotations for the corresponding `sample`.

        Args:
        ----
            sample_data_token (str): Token of `sample_data`.

        Returns:
        -------
            list[Box2D]: List of instantiated Box2D classes.
        """
        sd_record: SampleData = self.get("sample_data", sample_data_token)
        sample_record: Sample = self.get("sample", sd_record.sample_token)
        return list(map(self.get_box2d, sample_record.ann_2ds))

    def box_velocity(
        self, sample_annotation_token: str, max_time_diff: float = 1.5
    ) -> VelocityType:
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
            VelocityType: Velocity in the order of (vx, vy, vz) in m/s.
        """
        current: SampleAnnotation = self.get("sample_annotation", sample_annotation_token)

        # If the real velocity is annotated, returns it
        if current.velocity is not None:
            return current.velocity

        has_prev = current.prev != ""
        has_next = current.next != ""

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        first: SampleAnnotation
        if has_prev:
            first = self.get("sample_annotation", current.prev)
        else:
            first = current

        last: SampleAnnotation
        if has_next:
            last = self.get("sample_annotation", current.next)
        else:
            last = current

        pos_last = last.translation
        pos_first = first.translation
        pos_diff = pos_last - pos_first

        last_sample: Sample = self.get("sample", last.sample_token)
        first_sample: Sample = self.get("sample", first.sample_token)
        time_last = 1e-6 * last_sample.timestamp
        time_first = 1e-6 * first_sample.timestamp
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def render_scene(
        self,
        scene_token: str,
        max_time_seconds: float = np.inf,
        *,
        render_velocity: bool = False,
    ) -> None:
        """Render specified scene.

        Args:
        ----
            scene_token (str): Unique identifier of scene.
            max_time_seconds (float, optional): Max time length to be rendered [s]. Defaults to np.inf.
            render_velocity (bool, optional): Whether to render box velocity. Defaults to False.
        """
        camera_names = [
            sensor.channel.value
            for sensor in self.sensor
            if sensor.modality == SensorModality.CAMERA
        ]

        sensor_space_views = [
            rrb.Spatial2DView(name=camera, origin=f"world/ego_vehicle/{camera}")
            for camera in camera_names
        ]
        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D", origin="world"),
                rrb.TextDocumentView(origin="description", name="Description"),
                column_shares=[3, 1],
            ),
            rrb.Grid(*sensor_space_views),
            row_shares=[4, 2],
        )
        rr.init(
            application_id=f"t4-devkit@{scene_token}",
            recording_id=None,
            spawn=True,
            default_enabled=True,
            strict=True,
            default_blueprint=blueprint,
        )

        # render scene
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        scene: Scene = self.get("scene", scene_token)
        first_sample: Sample = self.get("sample", scene.first_sample_token)

        first_lidar_token = ""
        first_radar_tokens: list[str] = []
        first_camera_tokens: list[str] = []

        # FIXME: if sample data is not associated with the first sample, all frames are not rendered.
        for channel, sd_token in first_sample.data.items():
            self._render_sensor_calibration(sd_token)
            if channel.modality == SensorModality.LIDAR:
                first_lidar_token = sd_token
            elif channel.modality == SensorModality.RADAR:
                first_radar_tokens.append(sd_token)
            elif channel.modality == SensorModality.CAMERA:
                first_camera_tokens.append(sd_token)

        first_lidar_sd_record: SampleData = self.get("sample_data", first_lidar_token)
        max_timestamp_us = first_lidar_sd_record.timestamp + sec2us(max_time_seconds)

        self._render_lidar_and_ego(first_lidar_token, max_timestamp_us)
        self._render_radars(first_radar_tokens, max_timestamp_us)
        self._render_cameras(first_camera_tokens, max_timestamp_us)
        self._render_annotation_3ds(
            scene.first_sample_token, max_timestamp_us, render_velocity=render_velocity
        )
        self._render_annotation_2ds(scene.first_sample_token, max_timestamp_us)

    def _render_lidar_and_ego(self, first_lidar_token: str, max_timestamp_us: float) -> None:
        """Render lidar pointcloud and ego transform.

        Args:
        ----
            first_lidar_token (str): First sample data token corresponding to the lidar.
            max_timestamp_us (float): Max time length in [us].
        """
        current_lidar_token = first_lidar_token

        while current_lidar_token != "":
            sample_data: SampleData = self.get("sample_data", current_lidar_token)

            if max_timestamp_us < sample_data.timestamp:
                break

            rr.set_time_seconds("timestamp", us2sec(sample_data.timestamp))

            ego_pose: EgoPose = self.get("ego_pose", sample_data.ego_pose_token)
            rotation_xyzw = np.roll(ego_pose.rotation.q, shift=-1)
            rr.log(
                "world/ego_vehicle",
                rr.Transform3D(
                    translation=ego_pose.translation,
                    rotation=rr.Quaternion(xyzw=rotation_xyzw),
                    from_parent=False,
                ),
            )

            sensor_name = sample_data.channel.value
            pointcloud = LidarPointCloud.from_file(osp.join(self.data_root, sample_data.filename))
            points = pointcloud.points[:3].T  # (N, 3)
            point_distances = np.linalg.norm(points, axis=1)
            point_colors = COLOR_MAP(COLOR_NORM(point_distances))
            rr.log(f"world/ego_vehicle/{sensor_name}", rr.Points3D(points, colors=point_colors))
            current_lidar_token = sample_data.next

    def _render_radars(self, first_radar_tokens: list[str], max_timestamp_us: float) -> None:
        """Render radar pointcloud.

        Args:
        ----
            first_radar_tokens (list[str]): List of first sample data tokens corresponding to radars.
            max_timestamp_us (float): Max time length in [us].
        """
        for first_radar_token in first_radar_tokens:
            current_radar_token = first_radar_token
            while current_radar_token != "":
                sample_data: SampleData = self.get("sample_data", current_radar_token)

                if max_timestamp_us < sample_data.timestamp:
                    break

                rr.set_time_seconds("timestamp", us2sec(sample_data.timestamp))

                sensor_name = sample_data.channel.value
                pointcloud = RadarPointCloud.from_file(
                    osp.join(self.data_root, sample_data.filename)
                )
                points = pointcloud.points[:3].T  # (N, 3)
                point_distances = np.linalg.norm(points, axis=1)
                point_colors = COLOR_MAP(COLOR_NORM(point_distances))
                rr.log(f"world/ego_pose/{sensor_name}", rr.Points3D(points, colors=point_colors))
                current_radar_token = sample_data.next

    def _render_cameras(self, first_camera_tokens: list[str], max_timestamp_us: float) -> None:
        """Render camera images.

        Args:
        ----
            first_camera_tokens (list[str]): List of first sample data tokens corresponding to cameras.
            max_timestamp_us (float): Max time length in [us].
        """
        for first_camera_token in first_camera_tokens:
            current_camera_token = first_camera_token
            while current_camera_token != "":
                sample_data: SampleData = self.get("sample_data", current_camera_token)

                if max_timestamp_us < sample_data.timestamp:
                    break

                rr.set_time_seconds("timestamp", us2sec(sample_data.timestamp))

                sensor_name = sample_data.channel.value
                rr.log(
                    f"world/ego_vehicle/{sensor_name}",
                    rr.ImageEncoded(path=osp.join(self.data_root, sample_data.filename)),
                )
                current_camera_token = sample_data.next

    def _render_annotation_3ds(
        self,
        first_sample_token: str,
        max_timestamp_us: float,
        *,
        render_velocity: bool = False,
    ) -> None:
        """Render annotated 3D boxes.

        Args:
        ----
            first_sample_token (str): First sample token.
            max_timestamp_us (float): Max time length in [us].
            render_velocity (bool): Whether to render velocity arrow. Defaults to False.
        """
        current_sample_token = first_sample_token
        while current_sample_token != "":
            sample: Sample = self.get("sample", current_sample_token)

            if max_timestamp_us < sample.timestamp:
                break

            rr.set_time_seconds("timestamp", us2sec(sample.timestamp))

            centers: list[TranslationType] = []
            rotations: list[rr.Quaternion] = []
            sizes: list[SizeType] = []
            uuids: list[str] = []
            class_ids: list[int] = []
            velocities: list[VelocityType] = []
            for ann_token in sample.ann_3ds:
                ann: SampleAnnotation = self.get("sample_annotation", ann_token)

                centers.append(ann.translation)

                rotation_xyzw = np.roll(ann.rotation.q, shift=-1)
                rotations.append(rr.Quaternion(xyzw=rotation_xyzw))

                width, length, height = ann.size
                sizes.append((length, width, height))

                uuids.append(ann.instance_token)
                class_ids.append(self._label2id[ann.category_name])

                velocities.append(self.box_velocity(ann_token))

            rr.log(
                "world/ann3d/box",
                rr.Boxes3D(
                    sizes=sizes,
                    centers=centers,
                    rotations=rotations,
                    labels=uuids,
                    class_ids=class_ids,
                ),
            )

            if render_velocity:
                rr.log(
                    "world/ann3d/velocity",
                    rr.Arrows3D(vectors=velocities, origins=centers, class_ids=class_ids),
                )

            current_sample_token = sample.next

    def _render_annotation_2ds(self, first_sample_token: str, max_timestamp_us: float) -> None:
        """Render annotated 2D boxes.

        Args:
        ----
            first_sample_token (str): First sample token.
            max_timestamp_us (float): Max time length in [us].
        """
        current_sample_token = first_sample_token
        while current_sample_token != "":
            sample: Sample = self.get("sample", current_sample_token)

            if max_timestamp_us < sample.timestamp:
                break

            # FIXME: rendered camera image uses timestamp of sample data, but annotation box uses timestamp of sample
            # Therefore, 2d box will be shifted.
            rr.set_time_seconds("timestamp", us2sec(sample.timestamp))

            camera_anns: dict[str, dict] = {
                sd_token: {"sensor_name": channel.value, "boxes": [], "uuids": [], "class_ids": []}
                for channel, sd_token in sample.data.items()
                if channel.modality == SensorModality.CAMERA
            }
            for ann_token in sample.ann_2ds:
                ann: ObjectAnn = self.get("object_ann", ann_token)
                camera_anns[ann.sample_data_token]["boxes"].append(ann.bbox)
                camera_anns[ann.sample_data_token]["uuids"].append(ann.instance_token)
                camera_anns[ann.sample_data_token]["class_ids"].append(
                    self._label2id[ann.category_name]
                )

            for _, camera_ann in camera_anns.items():
                sensor_name: str = camera_ann["sensor_name"]
                rr.log(
                    f"world/ego_vehicle/{sensor_name}/ann2d",
                    rr.Boxes2D(
                        array=camera_ann["boxes"],
                        array_format=rr.Box2DFormat.XYXY,
                        labels=camera_ann["uuids"],
                        class_ids=camera_ann["class_ids"],
                    ),
                )
            current_sample_token = sample.next

    def _render_sensor_calibration(self, sample_data_token: str) -> None:
        """Render a fixed calibrated sensor transform.

        Args:
        -----
            sample_data_token (str): First sample data token corresponding to the sensor.
        """
        sample_data: SampleData = self.get("sample_data", sample_data_token)
        sensor_name = sample_data.channel.value
        calibrated_sensor: CalibratedSensor = self.get(
            "calibrated_sensor", sample_data.calibrated_sensor_token
        )
        rotation_xyzw = np.roll(calibrated_sensor.rotation.q, shift=-1)
        rr.log(
            f"world/ego_vehicle/{sensor_name}",
            rr.Transform3D(
                translation=calibrated_sensor.translation,
                rotation=rr.Quaternion(
                    xyzw=rotation_xyzw,
                ),
            ),
            static=True,
        )
        if sample_data.modality == SensorModality.CAMERA:
            rr.log(
                f"world/ego_vehicle/{sensor_name}",
                rr.Pinhole(
                    image_from_camera=calibrated_sensor.camera_intrinsic,
                    width=sample_data.width,
                    height=sample_data.height,
                ),
                static=True,
            )