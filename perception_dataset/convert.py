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
    parser.add_argument(
        "--generate-bbox-from-cuboid",
        action="store_true",
        help="generate 2d images annotations from cuboid annotations",
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
    elif task == "convert_non_annotated_t4_to_deepen":
        from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
            NonAnnotatedT4ToDeepenConverter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        camera_sensors = config_dict["conversion"]["camera_sensors"]
        annotation_hz = config_dict["conversion"]["annotation_hz"]
        workers_number = config_dict["conversion"]["workers_number"]
        drop_camera_token_not_found = config_dict["conversion"]["drop_camera_token_not_found"]
        save_intensity = config_dict["conversion"].get("save_intensity", False)

        converter = NonAnnotatedT4ToDeepenConverter(
            input_base=input_base,
            output_base=output_base,
            camera_sensors=camera_sensors,
            annotation_hz=annotation_hz,
            workers_number=workers_number,
            drop_camera_token_not_found=drop_camera_token_not_found,
            save_intensity=save_intensity,
        )

        logger.info(
            f"[BEGIN] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
        converter.convert()
        logger.info(
            f"[Done] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
    elif task == "convert_non_annotated_t4_tlr_to_deepen":
        from perception_dataset.deepen.non_annotated_t4_tlr_to_deepen_converter import (
            NonAnnotatedT4TlrToDeepenConverter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]

        converter = NonAnnotatedT4TlrToDeepenConverter(
            input_base=input_base,
            output_base=output_base,
        )

        logger.info(
            f"[BEGIN] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
        converter.convert()
        logger.info(
            f"[Done] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
    elif task == "convert_annotated_t4_to_deepen":
        from perception_dataset.deepen.annotated_t4_to_deepen_converter import (
            AnnotatedT4ToDeepenConverter,
        )
        from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
            NonAnnotatedT4ToDeepenConverter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        camera_sensors = config_dict["conversion"]["camera_sensors"]
        annotation_hz = config_dict["conversion"]["annotation_hz"]
        workers_number = config_dict["conversion"]["workers_number"]
        camera_position = config_dict["conversion"]["camera_position"]
        label_only = config_dict["conversion"]["label_only"]

        converter = AnnotatedT4ToDeepenConverter(
            input_base=input_base,
            output_base=output_base,
            camera_position=camera_position,
        )

        logger.info(
            f"[BEGIN] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )
        converter.convert()

        if not label_only:
            converter_non_anno = NonAnnotatedT4ToDeepenConverter(
                input_base=input_base,
                output_base=output_base,
                camera_sensors=camera_sensors,
                annotation_hz=annotation_hz,
                workers_number=workers_number,
            )
            converter_non_anno.convert()

        logger.info(
            f"[Done] Converting T4 dataset ({input_base}) to deepen format dataset ({output_base})"
        )

    elif task == "convert_deepen_to_t4":
        from perception_dataset.deepen.deepen_annotation import LabelInfo
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
        label_info = (
            LabelInfo(**config_dict["conversion"]["label_info"])
            if config_dict["conversion"].get("label_info")
            else None
        )

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
            label_info=label_info,
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
            "generate_bbox_from_cuboid": args.generate_bbox_from_cuboid,
        }
        converter_params = Rosbag2ConverterParams(**param_args)

        if args.synthetic:
            converter_params.data_type = DataType.SYNTHETIC

        converter = Rosbag2ToT4Converter(converter_params)

        logger.info("[BEGIN] Converting ros2bag output by simulator/autoware --> T4 Format Data")
        converter.convert()
        logger.info("[END] Conversion Completed")

    elif task == "convert_rosbag2_to_t4_tracking":
        from perception_dataset.rosbag2.converter_params import DataType
        from perception_dataset.rosbag2.rosbag2_to_t4_tracking_converter import (
            Rosbag2ToT4TrackingConverter,
        )

        param_args = {
            "task": config_dict["task"],
            "data_type": DataType.SYNTHETIC,
            "scene_description": config_dict["description"]["scene"],
            "overwrite_mode": args.overwrite,
            **config_dict["conversion"],
        }
        converter_params = Rosbag2ConverterParams(**param_args)
        converter = Rosbag2ToT4TrackingConverter(converter_params)

        logger.info(
            "[BEGIN] Converting ros2bag output by autoware scenario simulator --> T4 Format Data"
        )
        converter.convert()
        logger.info("[END] Conversion Completed")

    elif task == "convert_rosbag2_to_annotated_t4_tlr":
        from perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter import (
            Rosbag2ToAnnotatedT4TlrConverter,
        )

        param_args = {
            "task": task,
            **config_dict["conversion"],
        }
        if args.overwrite:
            param_args["overwrite_mode"] = args.overwrite
        logger.info("[BEGIN] Converting ros2bag output by simulator --> T4 Format Data")
        converter_params = Rosbag2ConverterParams(**param_args, with_camera=False)
        converter = Rosbag2ToAnnotatedT4TlrConverter(converter_params)
        converter.convert()
        logger.info("[END] Conversion Completed")

    elif task == "convert_rosbag2_with_gt_to_annotated_t4_tlr":
        from perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter import (
            Rosbag2ToAnnotatedT4TlrConverter,
        )

        param_args = {
            "task": task,
            **config_dict["conversion"],
        }
        if args.overwrite:
            param_args["overwrite_mode"] = args.overwrite
        logger.info("[BEGIN] Converting ros2bag with TLR GT --> T4 Format Data")
        topic_list_yaml = config_dict["conversion"]["topic_list"]
        with open(topic_list_yaml) as f:
            param_args["topic_list"] = yaml.safe_load(f)
        converter_params = Rosbag2ConverterParams(**param_args, with_camera=False)
        converter = Rosbag2ToAnnotatedT4TlrConverter(converter_params)
        converter.convert()
        logger.info("[END] Conversion Completed")

    elif task == "convert_annotated_t4_tlr_to_deepen":
        from perception_dataset.deepen.annotated_t4_tlr_to_deepen_converter import (
            AnnotatedT4TlrToDeepenConverter,
        )
        from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
            NonAnnotatedT4ToDeepenConverter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        camera_position = config_dict["conversion"]["camera_position"]
        camera_sensors = [{"channel": k} for k in camera_position.keys()]

        converter = AnnotatedT4TlrToDeepenConverter(
            input_base=input_base,
            output_base=output_base,
            camera_position=camera_position,
        )
        logger.info(
            f"[BEGIN] Converting T4 tlr dataset ({input_base}) to deepen format dataset ({output_base})"
        )
        converter.convert()
        if not config_dict["conversion"]["label_only"]:
            converter_non_anno = NonAnnotatedT4ToDeepenConverter(
                input_base=input_base,
                output_base=output_base,
                camera_sensors=camera_sensors,
            )
            converter_non_anno.convert()
        logger.info(
            f"[Done] Converting T4 tlr dataset ({input_base}) to deepen format dataset ({output_base})"
        )

    elif task == "add_2d_attribute":
        from perception_dataset.t4_dataset.attribute_merger import T4dataset2DAttributeMerger

        input_base = config_dict["conversion"]["input_base"]
        input_anno_base = config_dict["conversion"]["input_anno_base"]
        output_base = config_dict["conversion"]["output_base"]
        dataset_corresponding = config_dict["conversion"]["dataset_corresponding"]
        description = config_dict["description"]

        converter = T4dataset2DAttributeMerger(
            input_base=input_base,
            input_anno_base=input_anno_base,
            output_base=output_base,
            overwrite_mode=args.overwrite,
            dataset_corresponding=dataset_corresponding,
            description=description,
        )

        logger.info(f"[BEGIN] Merging T4 dataset ({input_base}) into T4 dataset ({output_base})")
        converter.convert()
        logger.info(f"[Done] Merging T4 dataset ({input_base}) into T4 dataset ({output_base})")

    elif task == "interpolate":
        from perception_dataset.t4_dataset.data_interpolator import DataInterpolator

        input_base: str = config_dict["conversion"]["input_base"]
        output_base: str = config_dict["conversion"]["output_base"]
        copy_excludes: list[str] | None = config_dict["conversion"].get("copy_excludes", None)

        converter = DataInterpolator(
            input_base=input_base,
            output_base=output_base,
            copy_excludes=copy_excludes,
            logger=logger,
        )

        logger.info(f"[BEGIN] Interpolating {input_base} into {output_base}")
        converter.convert()
        logger.info(f"[DONE] Interpolating {input_base} into {output_base}")

    elif task == "convert_fastlabel_2d_to_t4":
        from perception_dataset.fastlabel_to_t4.fastlabel_2d_to_t4_converter import (
            FastLabel2dToT4Converter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        input_anno_base = config_dict["conversion"]["input_anno_base"]
        description = config_dict["description"]
        input_bag_base = config_dict["conversion"]["input_bag_base"]
        topic_list_yaml_path = config_dict["conversion"]["topic_list"]
        tlr_mode = config_dict["conversion"].get("tlr_mode", False)

        with open(topic_list_yaml_path) as f:
            topic_list_yaml = yaml.safe_load(f)

        converter = FastLabel2dToT4Converter(
            input_base=input_base,
            output_base=output_base,
            input_anno_base=input_anno_base,
            overwrite_mode=args.overwrite,
            description=description,
            input_bag_base=input_bag_base,
            topic_list=topic_list_yaml,
            tlr_mode=tlr_mode,
        )
        logger.info(f"[BEGIN] Converting Fastlabel data ({input_base}) to T4 data ({output_base})")
        converter.convert()
        logger.info(f"[END] Converting Fastlabel data ({input_base}) to T4 data ({output_base})")

    elif task == "convert_fastlabel_2d_semantic_to_t4_tlr":
        from perception_dataset.fastlabel_to_t4.convert_fastlabel_2d_semantic_to_t4_tlr import (
            FastLabel2dSemanticToT4TlrConverter,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        input_anno_base = config_dict["conversion"]["input_anno_base"]
        input_semantic_anno_base = config_dict["conversion"]["input_semantic_anno_base"]
        description = config_dict["description"]
        input_bag_base = config_dict["conversion"]["input_bag_base"]
        topic_list_yaml_path = config_dict["conversion"]["topic_list"]
        arrow_angle_tolerance_deg = config_dict["conversion"].get("arrow_angle_tolerance_deg", 10.0)
        output_dataset_version = config_dict["conversion"].get("output_dataset_version")

        with open(topic_list_yaml_path) as f:
            topic_list_yaml = yaml.safe_load(f)

        converter = FastLabel2dSemanticToT4TlrConverter(
            input_base=input_base,
            output_base=output_base,
            input_anno_base=input_anno_base,
            input_semantic_anno_base=input_semantic_anno_base,
            overwrite_mode=args.overwrite,
            description=description,
            input_bag_base=input_bag_base,
            topic_list=topic_list_yaml,
            arrow_angle_tolerance_deg=arrow_angle_tolerance_deg,
            output_dataset_version=output_dataset_version,
        )
        logger.info(
            f"[BEGIN] Converting Fastlabel bulb + semantic data ({input_base}) to T4 data ({output_base})"
        )
        converter.convert()
        logger.info(
            f"[END] Converting Fastlabel bulb + semantic data ({input_base}) to T4 data ({output_base})"
        )

    elif task == "update_t4_with_fastlabel":
        from perception_dataset.fastlabel_to_t4.fastlabel_2d_to_t4_updater import (
            FastLabel2dToT4Updater,
        )

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        input_anno_base = config_dict["conversion"]["input_anno_base"]
        description = config_dict["description"]
        make_t4_dataset_dir = config_dict["conversion"]["make_t4_dataset_dir"]

        converter = FastLabel2dToT4Updater(
            input_base=input_base,
            output_base=output_base,
            input_anno_base=input_anno_base,
            overwrite_mode=args.overwrite,
            description=description,
            make_t4_dataset_dir=make_t4_dataset_dir,
        )
        logger.info(
            f"[BEGIN] Updating T4 dataset ({input_base}) with FastLabel {input_anno_base} into T4 data ({output_base})"
        )
        converter.convert()
        logger.info(
            f"[DONE] Updating T4 dataset ({input_base}) with FastLabel {input_anno_base} into T4 data ({output_base})"
        )

    elif task == "merge_2d_t4dataset_to_3d":
        from perception_dataset.t4_dataset.t4_dataset_2d3d_merger import T4dataset2D3DMerger

        input_base = config_dict["conversion"]["input_base"]
        output_base = config_dict["conversion"]["output_base"]
        dataset_corresponding = config_dict["conversion"]["dataset_corresponding"]

        converter = T4dataset2D3DMerger(
            input_base=input_base,
            output_base=output_base,
            dataset_corresponding=dataset_corresponding,
        )

        logger.info(f"[BEGIN] Merging T4 dataset ({input_base}) into T4 dataset ({output_base})")
        converter.convert()
        logger.info(f"[Done] Merging T4 dataset ({input_base}) into T4 dataset ({output_base})")

    elif task == "convert_fastlabel_to_t4":
        from perception_dataset.fastlabel_to_t4.fastlabel_to_t4_converter import (
            FastLabelToT4Converter,
        )

        make_t4_dataset_dir = config_dict["conversion"]["make_t4_dataset_dir"]
        input_base = config_dict["conversion"]["input_base"]
        input_anno_base = config_dict["conversion"]["input_anno_base"]
        output_base = config_dict["conversion"]["output_base"]
        description = config_dict["description"]
        input_bag_base = config_dict["conversion"]["input_bag_base"]
        if input_bag_base is not None:
            topic_list_yaml_path = config_dict["conversion"]["topic_list"]
            with open(topic_list_yaml_path) as f:
                topic_list_yaml = yaml.safe_load(f)
        else:
            topic_list_yaml = None

        converter = FastLabelToT4Converter(
            input_base=input_base,
            output_base=output_base,
            input_anno_base=input_anno_base,
            overwrite_mode=args.overwrite,
            description=description,
            make_t4_dataset_dir=make_t4_dataset_dir,
            input_bag_base=input_bag_base,
            topic_list=topic_list_yaml,
        )
        logger.info(f"[BEGIN] Converting Fastlabel data ({input_base}) to T4 data ({output_base})")
        converter.convert()
        logger.info(f"[END] Converting Fastlabel data ({input_base}) to T4 data ({output_base})")
    elif task == "convert_rosbag2_to_localization_evaluation":
        from perception_dataset.rosbag2.rosbag2_to_t4_loc_converter import Rosbag2ToT4LocConverter

        param_args = {
            "task": task,
            "scene_description": config_dict["description"]["scene"],
            **config_dict["conversion"],
        }
        if args.overwrite:
            param_args["overwrite_mode"] = args.overwrite
        logger.info("[BEGIN] Converting ros2bag --> T4 Localization Evaluation")
        converter_params = Rosbag2ConverterParams(**param_args, with_camera=False)
        converter = Rosbag2ToT4LocConverter(converter_params)
        converter.convert()
        logger.info("[END] Conversion Completed")
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
