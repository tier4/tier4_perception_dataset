from typing import Any, Dict, List, Optional, Union
import uuid

from autoware_auto_perception_msgs.msg import (
    DetectedObject,
    DetectedObjects,
    ObjectClassification,
    TrackedObject,
    TrackedObjects,
)
from tier4_perception_msgs.msg import (
    TrafficLight,
    TrafficLightArray,
    TrafficLightElement,
    TrafficLightRoi,
    TrafficLightRoiArray,
)

DEFAULT_ATTRIBUTES_BY_CATEGORY_NAME: Dict[str, List[str]] = {
    "unknown": [
        "object_state.still",
        "occlusion_state.none",
    ],
    "car": [
        "occlusion_state.none",
        "extremities_state.none",
        "vehicle_state.driving",
    ],
    "truck": [
        "occlusion_state.none",
        "extremities_state.none",
        "vehicle_state.driving",
    ],
    "bus": [
        "occlusion_state.none",
        "extremities_state.none",
        "vehicle_state.driving",
    ],
    "trailer": [
        "occlusion_state.none",
        "extremities_state.none",
        "vehicle_state.driving",
    ],
    "motorcycle": [
        "occlusion_state.none",
        "extremities_state.none",
        "vehicle_state.driving",
    ],
    "bicycle": [
        "occlusion_state.none",
        "extremities_state.none",
        "vehicle_state.driving",
    ],
    "pedestrian": [
        "occlusion_state.none",
        "extremities_state.none",
        "pedestrian_state.standing",
    ],
    "bicycle_without_rider": [],
    "motorcycle_without_rider": [],
    "personal_mobility_vehicle": [],
    "street_asset": [],
}


def semantic_type_to_class_name(semantic_type: int) -> str:
    """https://github.com/tier4/tier4_autoware_msgs/blob/tier4/universe/tier4_perception_msgs/msg/object_recognition/Semantic.msg"""
    semantic_to_category: Dict[int, str] = {
        0: "unknown",
        1: "car",
        2: "truck",
        3: "bus",
        4: "bicycle",
        5: "motorcycle",
        6: "pedestrian",
        7: "animal",
        11: "bicycle_without_rider",
        12: "motorcycle_without_rider",
        21: "street_asset",
    }

    return semantic_to_category.get(semantic_type, "unknown")


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
        11: "bicycle_without_rider",
        12: "motorcycle_without_rider",
        13: "personal_mobility_vehicle",
        14: "pedestrian",  # on wheelchair
        15: "pedestrian",  # with umbrella
        21: "street_asset",
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
            velocity: Dict[str, float] = {
                "x": obj.kinematics.twist_with_covariance.twist.linear.x,
                "y": obj.kinematics.twist_with_covariance.twist.linear.y,
                "z": obj.kinematics.twist_with_covariance.twist.linear.z,
            }
            acceleration: Optional[Dict[str, float]] = None
        elif isinstance(obj, TrackedObject):
            obj_uuid = uuid.UUID(bytes=obj.object_id.uuid.tobytes())
            velocity: Dict[str, float] = {
                "x": obj.kinematics.twist_with_covariance.twist.linear.x,
                "y": obj.kinematics.twist_with_covariance.twist.linear.y,
                "z": obj.kinematics.twist_with_covariance.twist.linear.z,
            }
            acceleration: Optional[Dict[str, float]] = {
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
            "attribute_names": (
                DEFAULT_ATTRIBUTES_BY_CATEGORY_NAME[category_name]
                if category_name in DEFAULT_ATTRIBUTES_BY_CATEGORY_NAME
                else []
            ),
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
    roi_msg: TrafficLightRoiArray, traffic_light_array_msg: TrafficLightArray
) -> List[Dict[str, Any]]:
    """
    https://github.com/tier4/tier4_autoware_msgs/tree/tier4/universe/tier4_perception_msgs
    https://github.com/autowarefoundation/autoware_msgs/tree/main/autoware_perception_msgs

    Args:
        roi_msg (tier4_perception_msgs.msg.TrafficLightRoiArray): tier4 perception msg
        signal_msg (tier4_perception_msgs.msg.TrafficLightArray): tier4 perception msg

    Returns:
        List[Dict[str, Any]]: dict format
    """
    color_to_str: Dict[int, str] = {
        TrafficLightElement.GREEN: "green",
        TrafficLightElement.RED: "red",
        TrafficLightElement.AMBER: "yellow",
        TrafficLightElement.WHITE: "white",
    }
    shape_to_str: Dict[int, str] = {
        TrafficLightElement.CIRCLE: "circle",
        TrafficLightElement.DOWN_ARROW: "down",
        TrafficLightElement.DOWN_LEFT_ARROW: "down_left",
        TrafficLightElement.DOWN_RIGHT_ARROW: "down_right",
        TrafficLightElement.LEFT_ARROW: "left",
        TrafficLightElement.RIGHT_ARROW: "right",
        TrafficLightElement.CROSS: "cross",
        TrafficLightElement.UP_ARROW: "straight",
    }

    def get_category_name(signal: TrafficLight):
        # list for saving the status of the lights
        blubs: List[int] = []

        category: str = ""
        for element in signal.elements:
            element: TrafficLightElement
            if element.color == TrafficLightElement.UNKNOWN:
                blubs.append(TrafficLightElement.UNKNOWN)
            elif element.shape == TrafficLightElement.CIRCLE:
                assert element.color in color_to_str
                # if the shape is circle, save the color
                blubs.append(element.color)
            else:
                # if the shape is not circle, the color must be green
                # in this case, save the shape
                assert element.color == TrafficLightElement.GREEN and element.shape in shape_to_str
                blubs.append(element.shape)
        # we want the category name to have the format "color-shape1",
        # and the color constants are smaller than shape constants,
        # we can simply achieve this by sort()
        blubs.sort()
        blubs_str: List[str] = []
        for blub in blubs:
            if blub == TrafficLightElement.UNKNOWN:
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
        traffic_light_array_msg, TrafficLightArray
    ), f"Invalid object message type: {type(traffic_light_array_msg)}"

    scene_annotation_list: List[Dict[str, Any]] = []
    for roi in roi_msg.rois:
        roi: TrafficLightRoi
        for signal in traffic_light_array_msg.signals:
            signal: TrafficLight
            if roi.traffic_light_id == signal.traffic_light_id:
                category_name = get_category_name(signal)
                label_dict: Dict[str, Any] = {
                    "category_name": category_name,
                    # this sensor_id would not be saved to the final dataset,
                    # considering traffic light would generally used camera_only mode and one camera,
                    # setting as "0" would be no problem
                    "sensor_id": 0,
                    "instance_id": str(signal.traffic_light_id),
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
