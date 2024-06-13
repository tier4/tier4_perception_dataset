import copy

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo


def mkmat(rows, cols, L):
    mat = np.matrix(L, dtype="float64")
    mat.resize((rows, cols))
    return mat


class PinholeCameraModel:
    """Mainly taken from https://github.com/ros-perception/vision_opencv/blob/humble/image_geometry/image_geometry/cameramodels.py
    to address the mistaken usage of cv2.remap in the original code.
    """

    def __init__(self):
        self.K = None
        self.D = None
        self.R = None
        self.P = None
        self.full_K = None
        self.full_P = None
        self.width = None
        self.height = None
        self.binning_x = None
        self.binning_y = None
        self.raw_roi = None
        self.tf_frame = None
        self.stamp = None

    def fromCameraInfo(self, msg: CameraInfo):
        """
        :param msg: camera parameters
        :type msg:  sensor_msgs.msg.CameraInfo

        Set the camera parameters from the :class:`sensor_msgs.msg.CameraInfo` message.
        """
        self.K = mkmat(3, 3, msg.k)
        if msg.d:
            self.D = mkmat(len(msg.d), 1, msg.d)
        else:
            self.D = None
        self.R = mkmat(3, 3, msg.r)
        self.P = mkmat(3, 4, msg.p)
        self.full_K = mkmat(3, 3, msg.k)
        self.full_P = mkmat(3, 4, msg.p)
        self.width = msg.width
        self.height = msg.height
        self.binning_x = max(1, msg.binning_x)
        self.binning_y = max(1, msg.binning_y)
        self.resolution = (msg.width, msg.height)

        self.raw_roi = copy.copy(msg.roi)
        # ROI all zeros is considered the same as full resolution
        if (
            self.raw_roi.x_offset == 0
            and self.raw_roi.y_offset == 0
            and self.raw_roi.width == 0
            and self.raw_roi.height == 0
        ):
            self.raw_roi.width = self.width
            self.raw_roi.height = self.height
        self.tf_frame = msg.header.frame_id
        self.stamp = msg.header.stamp

        # Adjust K and P for binning and ROI
        self.K[0, 0] /= self.binning_x
        self.K[1, 1] /= self.binning_y
        self.K[0, 2] = (self.K[0, 2] - self.raw_roi.x_offset) / self.binning_x
        self.K[1, 2] = (self.K[1, 2] - self.raw_roi.y_offset) / self.binning_y
        self.P[0, 0] /= self.binning_x
        self.P[1, 1] /= self.binning_y
        self.P[0, 2] = (self.P[0, 2] - self.raw_roi.x_offset) / self.binning_x
        self.P[1, 2] = (self.P[1, 2] - self.raw_roi.y_offset) / self.binning_y

    def rectifyImage(self, raw: np.ndarray) -> np.ndarray:
        """Applies the rectification specified by camera parameters :math:`K` and and :math:`D` to image `raw` and writes the resulting image `rectified`.
        Args:
            raw: input image (np.ndarray)
        Return:
            rectified image (np.ndarray)
        """

        self.mapx = np.ndarray(shape=(self.height, self.width, 1), dtype="float32")
        self.mapy = np.ndarray(shape=(self.height, self.width, 1), dtype="float32")
        cv2.initUndistortRectifyMap(
            self.K,
            self.D,
            self.R,
            self.P,
            (self.width, self.height),
            cv2.CV_32FC1,
            self.mapx,
            self.mapy,
        )
        rectified = cv2.remap(raw, self.mapx, self.mapy, cv2.INTER_CUBIC)
        return rectified

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PinholeCameraModel):
            return False
        return (
            np.array_equal(self.K, other.K)
            and np.array_equal(self.D, other.D)
            and np.array_equal(self.R, other.R)
            and np.array_equal(self.P, other.P)
            and self.width == other.width
            and self.height == other.height
            and self.binning_x == other.binning_x
            and self.binning_y == other.binning_y
            and self.raw_roi == other.raw_roi
            and self.tf_frame == other.tf_frame
        )
