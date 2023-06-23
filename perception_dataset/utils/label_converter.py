from typing import Dict


class LabelConverter:
    def __init__(self):
        self.label_map: Dict[str, str] = LabelConverter._set_label_map()
        self.attribute_map: Dict[str, str] = LabelConverter._set_attribute_map()

    @staticmethod
    def _set_label_map() -> Dict[str, str]:
        label_map: Dict[str, str] = {
            "bicycle": "bicycle",
            "BICYCLE": "bicycle",
            "vehicle.bicycle": "bicycle",
            "bus": "bus",
            "BUS": "bus",
            "vehicle.bus (bendy & rigid)": "bus",
            "vehicle.bus": "bus",
            "car": "car",
            "CAR": "car",
            "vehicle.car": "car",
            "vehicle.construction": "car",
            "vehicle.emergency (ambulance & police)": "car",
            "motorbike": "motorcycle",
            "MOTORBIKE": "motorcycle",
            "vehicle.motorcycle": "motorcycle",
            "pedestrian": "pedestrian",
            "PEDESTRIAN": "pedestrian",
            "pedestrian.adult": "pedestrian",
            "pedestrian.child": "pedestrian",
            "pedestrian.construction_worker": "construction_worker",
            "pedestrian.personal_mobility": "pedestrian",
            "pedestrian.police_officer": "police_officer",
            "pedestrian.stroller": "stroller",
            "pedestrian.wheelchair": "wheelchair",
            "truck": "truck",
            "TRUCK": "truck",
            "vehicle.truck": "truck",
            "vehicle.trailer": "trailer",
            "vehicle.ambulance": "ambulance",
            "vehicle.fire": "fire_truck",
            "vehicle.police": "police_car",
            "animal": "animal",
            "ANIMAL": "animal",
            "unknown": "unknown",
            "UNKNOWN": "unknown",
            "movable_object.barrier": "barrier",
            "movable_object.debris": "debris",
            "movable_object.pushable_pullable": "pushable_pullable",
            "movable_object.trafficcone": "cone",
            "movable_object.traffic_cone": "cone",
            "static_object.bicycle rack": "unknown",
            "static_object.bicycle_rack": "unknown",
            "static_object.bollard": "bollard",
            "trailer": "trailer",
            "motorcycle": "motorcycle",
            "vehicle": "car",
            "construction_worker": "construction_worker",
            "stroller": "stroller",
            "police_officer": "police_officer",
            "wheelchair": "wheelchair",
            "police_car": "police_car",
            "fire_truck": "fire_truck",
            "ambulance": "ambulance",
            "forklift": "forklift",
            "barrier": "barrier",
            "pushable_pullable": "pushable_pullable",
            "traffic_cone": "traffic_cone",
            "bollard": "bollard",
            "protruding_object": "protruding_object",
        }
        return label_map

    @staticmethod
    def _set_attribute_map() -> Dict[str, str]:
        attribute_map: Dict[str, str] = {
            "pedestrian_state.siting_lying_down": "pedestrian_state.siting",
            "pedestrian_state.sitting": "pedestrian_state.siting",
            "pedestrian_state.standing": "pedestrian_state.standing",
            "pedestrian_state.moving": "pedestrian_state.standing",
            "vehicle_state.driving": "vehicle_state.driving",
            "vehicle_state.moving": "vehicle_state.driving",
            "vehicle_state.stopped": "vehicle_state.driving",
            "vehicle_state.driving": "vehicle_state.driving",
            "vehicle_state.parked": "vehicle_state.parked",
            "occlusion_state.none": "occlusion_state.none",
            "occlusion_state.partial": "occlusion_state.partial",
            "occlusion_state.most": "occlusion_state.most",
            "occlusion_state.full": "occlusion_state.full",
            "cycle_state.without_rider": "cycle_state.without_rider",
            "cycle_state.with_rider": "cycle_state.with_rider",
            "motorcycle_state.without_rider": "motorcycle_state.without_rider",
            "motorcycle_state.with_rider": "motorcycle_state.with_rider",
            "extremities_state.none": "extremities_state.none",
            "extremities_state.open_door": "extremities_state.protruding_object",
            "extremities_state.protruding_object": "extremities_state.protruding_object",
            "emergency_vehicle_lights_state.on": "emergency_vehicle_lights_state.on",
            "emergency_vehicle_lights_state.off": "emergency_vehicle_lights_state.off",
            "emergency_vehicle_lights_state.unknown": "emergency_vehicle_lights_state.unknown",
            "object_state.still": "object_state.still",
        }
        return attribute_map

    def convert_label(
        self,
        label: str,
    ) -> str:
        return_label: str = self.label_map[label]
        return return_label

    def convert_attribute(
        self,
        attribute: str,
    ) -> str:
        return_attribute: str = self.attribute_map[attribute]
        return return_attribute


class TrafficLightLabelConverter:
    def __init__(self):
        self.label_map: Dict[str, str] = TrafficLightLabelConverter._set_label_map()

    @staticmethod
    def _set_label_map() -> Dict[str, str]:
        label_map: Dict[str, str] = {
            "unknown": "unknown",
            "green": "green",
            "green_straight": "green_straight",
            "green_left": "green_left",
            "green_right": "green_right",
            "yellow": "yellow",
            "yellow_straight": "yellow_straight",
            "yellow_left": "yellow_left",
            "yellow_right": "yellow_right",
            "yellow_left_straight": "yellow_straight_left",
            "yellow_right_straight": "yellow_straight_right",
            "yellow_left_right_straight": "yellow_straight_left_right",
            "red": "red",
            "red_straight": "red_straight",
            "red_left": "red_left",
            "red_right": "red_right",
            "red_left_straight": "red_straight_left",
            "red_straight_left": "red_straight_left",
            "red_right_straight": "red_straight_right",
            "red_left_right_straight": "red_straight_left_right",
            "red_rightdiagonal": "red_rightdiagonal",
            "red_leftdiagonal": "red_leftdiagonal",
        }
        return label_map

    def convert_label(
        self,
        label: str,
    ) -> str:
        return_label: str = self.label_map[label]
        return return_label
