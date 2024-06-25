# flake8: noqa
from typing import NewType

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = (
    "ArrayLike",
    "NDArray",
    "NDArrayF64",
    "NDArrayI64",
    "NDArrayI32",
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
    "BboxType",
    "MaskType",
    "KeypointType",
)

# numpy
NDArrayF64 = NDArray[np.float64]
NDArrayF32 = NDArray[np.float32]
NDArrayI64 = NDArray[np.int64]
NDArrayI32 = NDArray[np.int32]
NDArrayBool = NDArray[np.bool_]
NDArrayStr = NDArray[np.str_]

# 3D
TranslationType = NewType("TranslationType", tuple[float, float, float])
VelocityType = NewType("VelocityType", tuple[float, float, float] | None)
AccelerationType = NewType("AccelerationType", tuple[float, float, float] | None)
RotationType = NewType("RotationType", tuple[float, float, float, float])
SizeType = NewType("SizeType", tuple[float, float, float])
CamIntrinsicType = NewType(
    "CamIntrinsicType",
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
)
CamDistortionType = NewType("CamDistortionType", tuple[float, float, float, float, float])

# 2D
BboxType = NewType("BboxType", tuple[int, int, int, int])
MaskType = NewType("MaskType", list[int])
KeypointType = NewType("KeypointType", tuple[int, int])
