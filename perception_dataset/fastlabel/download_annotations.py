import argparse
import json
import os
import os.path as osp
import time
from typing import Dict, List

import fastlabel

client = fastlabel.Client()

trial_project_slugs = [
    "202405-lidar-3d-bbox-trial",
    "202405-lidar-3d-bbox-trial-2",
    "202405-lidar-3d-bbox-trial-3",
    "202405-lidar-3d-bbox-trial-4",
    "202405-panoptic-segmentation-trial",
    "202405-panoptic-segmentation-trial-2",
    "202405-panoptic-segmentation-trial-3",
    "202405-panoptic-segmentation-trial-4",
    "202405-license-plate-person-head-bbox-trial",
    "202405-traffic-light-recognition-trial",
]


def get_large_size_tasks(project: Dict, limit: int = None) -> List[Dict]:
    # Iterate pages until new tasks are empty.
    large_size_tasks = []
    offset = 0
    while True:
        time.sleep(1)

        if limit is not None:
            donwload_limit = limit
        else:
            donwload_limit = 100
        print(f"Downloading label: {project['slug']}, offset: {offset:05d}. Please wait...")
        if "image" in project["type"]:
            tasks = client.get_image_tasks(
                project=project["slug"], offset=offset, limit=donwload_limit
            )
        elif "pcd" in project["type"]:
            tasks = client.get_pcd_tasks(
                project=project["slug"], offset=offset, limit=donwload_limit
            )
        else:
            raise ValueError(f"Unknown project type: {project['type']}")
        large_size_tasks.extend(tasks)

        if limit is not None and len(large_size_tasks) >= limit:
            break
        elif len(tasks) > 0:
            offset = len(large_size_tasks)  # Set the offset
        else:
            break
    return large_size_tasks


def split_project_pcd_image(projects: List[Dict]) -> Dict[str, List[Dict]]:
    cuboid_projects = []
    seg_projects = []
    bbox_projects = []

    for project in projects:
        if "pcd_cuboid" in project["type"]:
            cuboid_projects.append(project)
        elif "segmentation" in project["type"] or project["type"] == "image_all":
            seg_projects.append(project)
        elif "bbox" in project["type"]:
            bbox_projects.append(project)
    return {"cuboid": cuboid_projects, "segmentation": seg_projects, "bbox": bbox_projects}


def is_task_completed(task: Dict) -> bool:
    return task["status"] in ["approved", "completed"]


def rename_image_task_name(name: str) -> str:
    if "_CAM_" not in name:
        return name
    base_name, camera_info = name.split("CAM_")
    dataset_name, index = base_name.split("/")
    frame_index = index[:5]
    camera_name, ext = os.path.splitext(camera_info)
    new_file_name = f"{dataset_name}/CAM_{camera_name}/{frame_index}{ext}"
    return new_file_name


def download_completed_annotations(
    project: Dict, output_dir: str, save_each: bool = False
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    completed_tasks = []
    if project["slug"] in trial_project_slugs:
        tasks = get_large_size_tasks(project=project)
        for task in tasks:
            if is_task_completed(task):
                task["name"] = rename_image_task_name(task["name"])
                completed_tasks.append(task)
                each_task_file_name = task["name"].replace(" ", "_").replace("/", "_") + ".json"
                if save_each:
                    with open(osp.join(output_dir, each_task_file_name), "w") as f:
                        json.dump([task], f, indent=4)
            else:
                print(f"Task {task['name']} is not completed. Removing.")
    return completed_tasks


def get_labels(output_dir: str) -> None:
    projects = client.get_projects()
    project_dict = split_project_pcd_image(projects)
    os.makedirs(output_dir, exist_ok=True)

    cuboid_tasks = []
    pcd_out_dir = osp.join(output_dir, "pcd_annotation")
    for project in project_dict["cuboid"]:
        if project["slug"] in trial_project_slugs:
            cuboid_tasks.extend(
                download_completed_annotations(project, pcd_out_dir, save_each=True)
            )
    with open(osp.join(output_dir, "all_label_lidar_cuboid.json"), "w") as f:
        json.dump(cuboid_tasks, f, indent=4)

    segmentation_tasks = []
    pcd_out_dir = osp.join(output_dir, "segmentation_annotation")
    for project in project_dict["segmentation"]:
        if project["slug"] in trial_project_slugs:
            segmentation_tasks.extend(
                download_completed_annotations(project, pcd_out_dir, save_each=True)
            )
    with open(osp.join(output_dir, "all_label_segmentation.json"), "w") as f:
        json.dump(segmentation_tasks, f, indent=4)

    tlr_tasks = []
    pcd_out_dir = osp.join(output_dir, "tlr_annotation")
    for project in project_dict["bbox"]:
        if project["slug"] in trial_project_slugs and "traffic-light" in project["slug"]:
            tlr_tasks.extend(download_completed_annotations(project, pcd_out_dir, save_each=True))
    with open(osp.join(output_dir, "all_label_traffic_light.json"), "w") as f:
        json.dump(tlr_tasks, f, indent=4)

    anonymization_tasks = []
    pcd_out_dir = osp.join(output_dir, "tlr_annotation")
    for project in project_dict["bbox"]:
        if project["slug"] in trial_project_slugs and "license-plate" in project["slug"]:
            anonymization_tasks.extend(
                download_completed_annotations(project, pcd_out_dir, save_each=True)
            )
    with open(osp.join(output_dir, "all_label_anonymization.json"), "w") as f:
        json.dump(anonymization_tasks, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="label_data",
        help="the directory where the annotation file is saved.",
    )
    args = parser.parse_args()
    get_labels(args.output_dir)
