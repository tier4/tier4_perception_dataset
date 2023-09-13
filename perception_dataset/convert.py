import argparse

import yaml

from perception_dataset.rosbag2.converter_params import DataType, Rosbag2ConverterParams
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite files if exist in output directory",
    )
    parser.add_argument(
        "--without_compress",
        action="store_true",
        help="do NOT compress rosbag/non-annotated-t4",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="convert synthetic data",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    task = config_dict["task"]
    if task == "convert_rosbag2_to_non_annotated_t4":
        from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import (
            Rosbag2ToNonAnnotatedT4Converter,
        )

        param_args = {
            "task": config_dict["task"],
            "scene_description": config_dict["description"]["scene"],
            "overwrite_mode": args.overwrite,
            "without_compress": args.without_compress,
            **config_dict["conversion"],
        }
        params = Rosbag2ConverterParams(**param_args)
        converter = Rosbag2ToNonAnnotatedT4Converter(params)
        logger.info(
            f"[BEGIN] Converting Rosbag2 ({params.input_base}) to Non Annotated T4 data ({params.output_base})"
        )
        converter.convert()
        logger.info(
            f"[END] Converting Rosbag2 ({params.input_base}) to Non Annotated T4 data ({params.output_base})"
        )
    elif task == "convert_t4_to_deepen":
        from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
            NonAnnotatedT4ToDeepenConverter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        camera_sensors = config_dict["conversion"]["camera_sensors"]
        annotation_hz = config_dict["conversion"]["annotation_hz"]
        workers_number = config_dict["conversion"]["workers_number"]

        converter = NonAnnotatedT4ToDeepenConverter(
            input_base=input_base,
            output_base=output_base,
            camera_sensors=camera_sensors,
            annotation_hz=annotation_hz,
            workers_number=workers_number,
        )

        logger.info(
            f"[BEGIN] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
        converter.convert()
        logger.info(
            f"[Done] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
    elif task == "convert_deepen_to_t4":
        from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter

        input_base = config_dict["conversion"]["input_base"]
        input_anno_file = config_dict["conversion"]["input_anno_file"]
        output_base = config_dict["conversion"]["output_base"]
        dataset_corresponding = config_dict["conversion"]["dataset_corresponding"]
        description = config_dict["description"]
        input_bag_base = config_dict["conversion"]["input_bag_base"]
        topic_list_yaml_path = config_dict["conversion"]["topic_list"]
        ignore_interpolate_label = False
        if "ignore_interpolate_label" in config_dict["conversion"]:
            ignore_interpolate_label = config_dict["conversion"]["ignore_interpolate_label"]
        with open(topic_list_yaml_path) as f:
            topic_list_yaml = yaml.safe_load(f)

        converter = DeepenToT4Converter(
            input_base=input_base,
            output_base=output_base,
            input_anno_file=input_anno_file,
            dataset_corresponding=dataset_corresponding,
            overwrite_mode=args.overwrite,
            description=description,
            input_bag_base=input_bag_base,
            topic_list=topic_list_yaml,
            ignore_interpolate_label=ignore_interpolate_label,
        )

        logger.info(f"[BEGIN] Converting Deepen data ({input_base}) to T4 data ({output_base})")
        converter.convert()
        logger.info(f"[END] Converting Deepen data ({input_base}) to T4 data ({output_base})")
    elif task == "convert_rosbag2_to_t4":
        from perception_dataset.rosbag2.rosbag2_to_t4_converter import Rosbag2ToT4Converter

        param_args = {
            "task": config_dict["task"],
            "scene_description": config_dict["description"]["scene"],
            "overwrite_mode": args.overwrite,
            **config_dict["conversion"],
        }
        converter_params = Rosbag2ConverterParams(**param_args)
        if args.synthetic:
            converter_params.data_type = DataType.SYNTHETIC
        converter = Rosbag2ToT4Converter(converter_params)

        logger.info("[BEGIN] Converting ros2bag output by simulator --> T4 Format Data")
        converter.convert()
        logger.info("[END] Conversion Completed")

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
