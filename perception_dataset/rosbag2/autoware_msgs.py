from typing import Any, Dict, List, Optional, Union
import uuid

from autoware_auto_perception_msgs.msg import (
    DetectedObject,
    DetectedObjects,
    ObjectClassification,
    TrackedObject,
    TrackedObjects,
    TrafficLight,
    TrafficLightRoi,
    TrafficLightRoiArray,
    TrafficSignal,
    TrafficSignalArray,
)


def semantic_type_to_class_name(semantic_type: int) -> str:
    """https://github.com/tier4/tier4_autoware_msgs/blob/tier4/universe/tier4_perception_msgs/msg/object_recognition/Semantic.msg"""
    semantic_to_category: Dict[int, str] = {
        0: "unknown",
        1: "car",
        2: "truck",
        3: "bus",
        4: "bicycle",
        5: "motorbike",
        6: "pedestrian",
        7: "animal",
        11: "bicycle_without_rider",
        12: "motorbike_without_rider",
        21: "street_asset",
    }

    return semantic_to_category.get(semantic_type, "unknown")


def parse_dynamic_object_array(msg) -> List[Dict[str, Any]]:
    """
    The object message is archived, but used for synthetic data.
    https://github.com/tier4/autoware_iv_msgs/blob/main/autoware_perception_msgs/msg/object_recognition/DynamicObjectArray.msg

    Args:
        msg (autoware_perception_msgs.msg.DynamicObjectArray): autoware perception msg (.iv)

    Returns:
        List[Dict[str, Any]]: dict format
    """
    scene_annotation_list: List[Dict[str, Any]] = []
    for obj in msg.objects:
        # obj: Dynamic Object

        obj_uuid = uuid.UUID(bytes=obj.id.uuid.tobytes())
        category_name = semantic_type_to_class_name(obj.semantic.type)
        position: Dict[str, Any] = {
            "x": obj.state.pose_covariance.pose.position.x,
            "y": obj.state.pose_covariance.pose.position.y,
            "z": obj.state.pose_covariance.pose.position.z,
        }
        orientation: Dict[str, Any] = {
            "x": obj.state.pose_covariance.pose.orientation.x,
            "y": obj.state.pose_covariance.pose.orientation.y,
            "z": obj.state.pose_covariance.pose.orientation.z,
            "w": obj.state.pose_covariance.pose.orientation.w,
        }
        velocity: Dict[str, Optional[float]] = {
            "x": None,
            "y": None,
            "z": None,
        }
        acceleration: Dict[str, Optional[float]] = {
            "x": None,
            "y": None,
            "z": None,
        }
        dimension: Dict[str, Any] = {
            "width": obj.shape.dimensions.y,
            "length": obj.shape.dimensions.x,
            "height": obj.shape.dimensions.z,
        }
        label_dict: Dict[str, Any] = {
            "category_name": category_name,
            "instance_id": str(obj_uuid),
            "attribute_names": [],  # not available
            "three_d_bbox": {
                "translation": position,
                "velocity": velocity,
                "acceleration": acceleration,
                "size": dimension,
                "rotation": orientation,
            },
            "num_lidar_pts": 1,  # placeholder, the value is caluculated in the Rosbag2ToT4Converter
            "num_radar_pts": 0,
        }
        scene_annotation_list.append(label_dict)

    return scene_annotation_list


def object_classification_to_category_name(object_classification) -> str:
    """https://github.com/tier4/autoware_auto_msgs/blob/tier4/main/autoware_auto_perception_msgs/msg/ObjectClassification.idl"""
    cls_to_cat: Dict[int, str] = {
        0: "unknown",
        1: "car",
        2: "truck",
        3: "bus",
        4: "trailer",
        5: "motorcycle",
        6: "bicycle",
        7: "pedestrian",
    }

    return cls_to_cat.get(object_classification, "unknown")


def parse_perception_objects(msg) -> List[Dict[str, Any]]:
    """https://github.com/tier4/autoware_auto_msgs/tree/tier4/main/autoware_auto_perception_msgs


    Args:
        msg (autoware_auto_perception_msgs.msg.DetectedObjects): autoware detection msg (.core/.universe)

    Returns:
        List[Dict[str, Any]]: dict format
    """
    assert isinstance(
        msg, (DetectedObjects, TrackedObjects)
    ), f"Invalid object message type: {type(msg)}"

    def get_category_name(classification: List[ObjectClassification]) -> str:
        max_score: float = -1.0
        out_class_name: str = "unknown"
        for obj_cls in classification:
            if obj_cls.probability > max_score:
                max_score = obj_cls.probability
                out_class_name = object_classification_to_category_name(obj_cls.label)
        return out_class_name

    scene_annotation_list: List[Dict[str, Any]] = []
    for obj in msg.objects:
        obj: Union[DetectedObject, TrackedObject]
        pose = obj.kinematics.pose_with_covariance.pose

        if isinstance(obj, DetectedObject):
            obj_uuid = uuid.uuid4()  # random uuid
            velocity: Dict[str, Optional[float]] = {
                "x": None,
                "y": None,
                "z": None,
            }
            acceleration: Dict[str, Optional[float]] = {
                "x": None,
                "y": None,
                "z": None,
            }
        elif isinstance(obj, TrackedObject):
            obj_uuid = uuid.UUID(bytes=obj.object_id.uuid.tobytes())
            velocity: Dict[str, Optional[float]] = {
                "x": obj.kinematics.twist_with_covariance.twist.linear.x,
                "y": obj.kinematics.twist_with_covariance.twist.linear.y,
                "z": obj.kinematics.twist_with_covariance.twist.linear.z,
            }
            acceleration: Dict[str, Optional[float]] = {
                "x": obj.kinematics.acceleration_with_covariance.accel.linear.x,
                "y": obj.kinematics.acceleration_with_covariance.accel.linear.y,
                "z": obj.kinematics.acceleration_with_covariance.accel.linear.z,
            }
        else:
            raise ValueError(
                f"Object message is neither DetectedObject nor TrackedObject: {type(obj)}"
            )

        category_name = get_category_name(obj.classification)
        position: Dict[str, Any] = {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        }
        orientation: Dict[str, Any] = {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        }
        dimension: Dict[str, Any] = {
            "width": obj.shape.dimensions.y,
            "length": obj.shape.dimensions.x,
            "height": obj.shape.dimensions.z,
        }
        label_dict: Dict[str, Any] = {
            "category_name": category_name,
            "instance_id": str(obj_uuid),
            "attribute_names": [],  # not available
            "three_d_bbox": {
                "translation": position,
                "velocity": velocity,
                "acceleration": acceleration,
                "size": dimension,
                "rotation": orientation,
            },
            "num_lidar_pts": 1,  # TODO(yukke42): impl
            "num_radar_pts": 0,
        }
        scene_annotation_list.append(label_dict)

    return scene_annotation_list


def parse_traffic_lights(
    roi_msg: TrafficLightRoiArray, signal_msg: TrafficSignalArray
) -> List[Dict[str, Any]]:
    """https://github.com/tier4/autoware_auto_msgs/tree/tier4/main/autoware_auto_perception_msgs


    Args:
        msg (autoware_auto_perception_msgs.msg.DetectedObjects): autoware detection msg (.core/.universe)

    Returns:
        List[Dict[str, Any]]: dict format
    """
    color_to_str: Dict[int, str] = {
        TrafficLight.GREEN: "green",
        TrafficLight.RED: "red",
        TrafficLight.AMBER: "yellow",
        TrafficLight.WHITE: "white",
    }
    shape_to_str: Dict[int, str] = {
        TrafficLight.CIRCLE: "circle",
        TrafficLight.DOWN_ARROW: "down",
        TrafficLight.DOWN_LEFT_ARROW: "down_left",
        TrafficLight.DOWN_RIGHT_ARROW: "down_right",
        TrafficLight.LEFT_ARROW: "left",
        TrafficLight.RIGHT_ARROW: "right",
        TrafficLight.CROSS: "cross",
        TrafficLight.UP_ARROW: "straight",
    }

    def get_category_name(signal: TrafficSignal):
        # list for saving the status of the lights
        blubs: List[int] = []

        category: str = ""
        for light in signal.lights:
            light: TrafficLight
            if light.color == TrafficLight.UNKNOWN:
                blubs.append(TrafficLight.UNKNOWN)
            elif light.shape == TrafficLight.CIRCLE:
                assert light.color in color_to_str
                # if the shape is circle, save the color
                blubs.append(light.color)
            else:
                # if the shape is not circle, the color must be green
                # in this case, save the shape
                assert light.color == TrafficLight.GREEN and light.shape in shape_to_str
                blubs.append(light.shape)
        # we want the category name to have the format "color-shape1",
        # and the color constants are smaller than shape constants,
        # we can simply achieve this by sort()
        blubs.sort()
        blubs_str: List[str] = []
        for blub in blubs:
            if blub == TrafficLight.UNKNOWN:
                blubs_str.append("unknown")
            elif blub in color_to_str:
                blubs_str.append(color_to_str[blub])
            else:
                blubs_str.append(shape_to_str[blub])
        category: str = "_".join(blubs_str)
        return category

    assert isinstance(
        roi_msg, TrafficLightRoiArray
    ), f"Invalid object message type: {type(roi_msg)}"
    assert isinstance(
        signal_msg, TrafficSignalArray
    ), f"Invalid object message type: {type(signal_msg)}"

    scene_annotation_list: List[Dict[str, Any]] = []
    for roi in roi_msg.rois:
        roi: TrafficLightRoi
        for signal in signal_msg.signals:
            signal: TrafficSignal
            if roi.id == signal._map_primitive_id:
                category_name = get_category_name(signal)
                label_dict: Dict[str, Any] = {
                    "category_name": category_name,
                    # this sensor_id would not be saved to the final dataset,
                    # considering traffic light would generally used camera_only mode and one camera,
                    # setting as "0" would be no problem
                    "sensor_id": 0,
                    "instance_id": str(signal.map_primitive_id),
                    "attribute_names": [],  # not available
                    "two_d_box": [
                        roi.roi.x_offset,
                        roi.roi.y_offset,
                        roi.roi.x_offset + roi.roi.width,
                        roi.roi.y_offset + roi.roi.height,
                    ],
                }
                scene_annotation_list.append(label_dict)
                break

    return scene_annotation_list
