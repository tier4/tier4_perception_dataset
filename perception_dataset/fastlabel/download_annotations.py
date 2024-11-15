from pprint import pprint
import time
import json

import os
from typing import List, Dict
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
    "202405-traffic-light-recognition-trial    ",
]


def get_large_size_tasks(project: str) -> List[Dict]:
    # Iterate pages until new tasks are empty.
    large_size_tasks = []
    offset = None
    while True:
        time.sleep(1)

        tasks = client.get_image_tasks(project=project, offset=offset, limit=1000)
        large_size_tasks.extend(tasks)

        if len(tasks) > 0:
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


projects = client.get_projects()
project_dict = split_project_pcd_image(projects)

segmentation_tasks = []
for project in project_dict["segmentation"]:
    if project["slug"] in trial_project_slugs:
        print(project["slug"])
        print(project["type"])
        tasks = client.get_image_tasks(project=project["slug"], limit=10)
        segmentation_tasks.extend(tasks)
with open("annotation_segmentation.json", "w") as f:
    json.dump(segmentation_tasks, f, indent=4)

cuboid_tasks = []
for project in project_dict["cuboid"]:
    if project["slug"] in trial_project_slugs:
        print(project["slug"])
        print(project["type"])
        tasks = client.get_pcd_tasks(project=project["slug"], limit=10)
        cuboid_tasks.extend(tasks)
with open("annotation_cuboid.json", "w") as f:
    json.dump(cuboid_tasks, f, indent=4)

