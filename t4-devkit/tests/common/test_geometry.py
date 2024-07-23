import numpy as np
from t4_devkit.common.geometry import view_points


def test_view_points_by_perspective_projection() -> None:
    points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )
    intrinsic = np.array([[1, 0, 0]])

    project = view_points(points, intrinsic)

    expect = np.array(
        [
            [0.14285714, 0.25, 0.33333333],
            [0.57142857, 0.625, 0.66666667],
            [1.0, 1.0, 1.0],
        ]
    )

    assert np.allclose(project, expect)


def test_view_points_by_orthographic_projection() -> None:
    points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )
    intrinsic = np.array([[1, 0, 0], [0, 1, 0]])

    project = view_points(points, intrinsic, normalize=False)

    expect = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    assert np.allclose(project, expect)


def test_view_points_with_distortion() -> None:
    points = np.array(
        [
            [0.5, -0.5],
            [0.5, -0.5],
            [1, 1],
        ]
    )
    intrinsic = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    distortion = np.array([0.1, 0.01, 0.01, 0.01, 0.001])

    project = view_points(points, intrinsic, distortion)

    print(project)

    expect = np.array(
        [
            [0.5413125, -0.5113125],
            [0.5413125, -0.5113125],
            [1.0, 1.0],
        ]
    )

    assert np.allclose(project, expect)
