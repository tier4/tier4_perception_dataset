import base64
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as cocomask
import skimage


def encode_polygons(polygons: List[List[List[float]]], width: int, height: int) -> Dict[str, Any]:
    flattened = [coord for polygon in polygons for point in polygon for coord in point]

    rle_objects = cocomask.frPyObjects(flattened, height, width)
    rle = cocomask.merge(rle_objects)

    rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")

    return rle


def encode_mask(mask: NDArray) -> List[Dict[str, Any]]:
    labeled = skimage.measure.label(mask, connectivity=1)
    num_instances = labeled.max()

    output = []
    for instance_id in range(1, num_instances + 1):
        instance_mask = np.asfortranarray(labeled == instance_id, dtype=np.uint8)
        rle = cocomask.encode(instance_mask)
        rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")
        output.append(rle)

    return output
