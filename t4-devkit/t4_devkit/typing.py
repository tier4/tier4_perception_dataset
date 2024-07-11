# flake8: noqa
from typing import NewType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyquaternion import Quaternion

__all__ = (
    "ArrayLike",
    "NDArray",
    "NDArrayF64",
    "NDArrayI64",
    "NDArrayI32",
    "NDArrayU8",
    "NDArrayBool",
    "NDArrayStr",
    "TranslationType",
    "VelocityType",
    "AccelerationType",
    "RotationType",
    "VelocityType",
    "SizeType",
    "CamIntrinsicType",
    "CamDistortionType",
    "RoiType",
    "MaskType",
    "KeypointType",
)

# numpy
NDArrayF64 = NDArray[np.float64]
NDArrayF32 = NDArray[np.float32]
NDArrayI64 = NDArray[np.int64]
NDArrayI32 = NDArray[np.int32]
NDArrayU8 = NDArray[np.uint8]
NDArrayBool = NDArray[np.bool_]
NDArrayStr = NDArray[np.str_]

# 3D
TranslationType = NewType("TranslationType", NDArrayF64)
VelocityType = NewType("VelocityType", NDArrayF64)
AccelerationType = NewType("AccelerationType", NDArrayF64)
RotationType = NewType("RotationType", Quaternion)
SizeType = NewType("SizeType", NDArrayF64)
CamIntrinsicType = NewType("CamIntrinsicType", NDArrayF64)
CamDistortionType = NewType("CamDistortionType", NDArrayF64)

# 2D
RoiType = NewType("RoiType", tuple[int, int, int, int])  # (xmin, ymin, xmax, ymax)
MaskType = NewType("MaskType", list[int])
KeypointType = NewType("KeypointType", NDArrayF64)
