import json
from typing import Any, Dict, List


class BasicAiCameraConfig:
    """See https://docs.basic.ai/docs/data-type-and-format#camera-config"""

    def __init__(self):
        self._camera_config_dict: Dict[int, Dict[str, Any]] = {}

    def set_camera_config(
        self,
        camera_index: int,
        camera_internal: Dict[str, float],
        camera_external: List[float],
        row_major: bool,
        distortionK: List[float],
        distortionP: List[float],
        distortionInvP: List[float],
        width: int,
        height: int,
    ):
        """Set camera config
        Args:
            camera_index (int): camera index
            camera_internal (Dict[str, float]): camera internal parameters
            camera_external (List[float]): camera external parameters
            row_major (bool): row major
            distortionK (List[float]): distortion K
            distortionP (List[float]): distortion P
            distortionInvP (List[float]): distortion inverse P
            width (int): width
            height (int): height
        """
        assert isinstance(camera_index, int)
        assert isinstance(camera_internal, dict)
        assert "fx" in camera_internal
        assert "cx" in camera_internal
        assert "fy" in camera_internal
        assert "cy" in camera_internal

        assert isinstance(camera_external, list)
        assert len(camera_external) == 16

        assert isinstance(row_major, bool)
        assert isinstance(distortionK, list)
        assert isinstance(distortionP, list)
        assert isinstance(distortionInvP, list)
        assert isinstance(width, int)
        assert isinstance(height, int)

        self._camera_config_dict[camera_index] = {
            "camera_internal": camera_internal,
            "camera_external": camera_external,
            "rowMajor": row_major,
            "distortionK": distortionK,
            "distortionP": distortionP,
            "distortionInvP": distortionInvP,
            "width": width,
            "height": height,
            "camera_index": camera_index,
        }

    def dump_json(self, path):
        """Dump json
        Args:
            path ([type]): path
        """
        # save json while changing dict to list, and preserving the order by key

        camera_config_list = [val for key, val in sorted(self._camera_config_dict.items())]
        with open(path, "w") as f:
            json.dump(camera_config_list, f, indent=2)


def save_pcd(points, output_pcd_file_path, fields=["x", "y", "z", "i"], binary=False):
    num_points = len(points)
    with open(output_pcd_file_path, "wb" if binary else "w") as f:
        # heads
        headers = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            f'FIELDS {" ".join(fields)}',
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {num_points}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {num_points}",
            f'DATA {"binary" if binary else "ascii"}',
        ]
        header = "\n".join(headers) + "\n"
        if binary:
            header = bytes(header, "ascii")
        f.write(header)

        # points
        if binary:
            f.write(points.tobytes())
        else:
            for i in range(num_points):
                x, y, z, rgb = points[i]
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(rgb)}\n")
