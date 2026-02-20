import pytest

import perception_dataset.utils.misc as misc_utils


def test_unix_timestamp_to_nusc_timestamp():
    # TODO(yukke42): impl test_unix_timestamp_to_nusc_timestamp
    pass


def test_nusc_timestamp_to_unix_timestamp():
    # TODO(yukke42): impl test_unix_timestamp_to_nusc_timestamp
    pass


def test_get_sample_data_filename():
    # TODO(yukke42): impl test_unix_timestamp_to_nusc_timestamp
    pass


# Tests for get_frame_index_from_filename
class TestGetFrameIndexFromFilename:
    """Test suite for get_frame_index_from_filename function."""

    def test_valid_filename_with_typical_format(self):
        """Test extraction from standard filename format: data/SENSOR/12345.jpg"""
        assert misc_utils.get_frame_index_from_filename("data/CAM_FRONT/00123.jpg") == 123
        assert misc_utils.get_frame_index_from_filename("data/CAM_FRONT/12345.png") == 12345
        assert misc_utils.get_frame_index_from_filename("data/LIDAR_TOP/00001.bin") == 1

    def test_filename_with_non_numeric_frame_index(self):
        """Test filenames where the frame index is not numeric."""
        assert misc_utils.get_frame_index_from_filename("data/CAM_FRONT/frame_abc.jpg") is None
        assert misc_utils.get_frame_index_from_filename("data/CAM_FRONT/test123.jpg") is None
        assert misc_utils.get_frame_index_from_filename("data/CAM_FRONT/abc.jpg") is None
        assert misc_utils.get_frame_index_from_filename("") is None

    def test_filename_with_wrong_directory_structure(self):
        """Test filename with different numbers of directory levels."""
        assert misc_utils.get_frame_index_from_filename("sensor/00123.jpg") is None
        assert misc_utils.get_frame_index_from_filename("00123.jpg") == 123


def assert_synced_frame_info_list(expected, synced_frame_info_list):
    assert len(synced_frame_info_list) == len(expected)
    for i in range(len(synced_frame_info_list)):
        assert synced_frame_info_list[i][0] == expected[i][0]
        assert synced_frame_info_list[i][1] == expected[i][1]
        if expected[i][2] is None:  # If the timestamp is dummy
            assert synced_frame_info_list[i][2] is None
        else:
            synced_frame_info_list[i][2] == pytest.approx(expected[i][2])


def test_get_lidar_camera_synced_frame_info_1():
    # Basic test case
    image_timestamp_list = [0.07, 0.17, 0.27, 0.37, 0.47]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None], [3, 3, None], [4, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.07,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_2():
    # Test case with image drops
    image_timestamp_list = [0.07, 0.27, 0.47]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [None, 1, 0.17], [1, 2, None], [None, 3, 0.37], [2, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.07,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_3():
    # Test case with different latency
    image_timestamp_list = [0.11, 0.21, 0.31, 0.41, 0.51]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None], [3, 3, None], [4, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.11,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_4():
    # Test case with different latency
    image_timestamp_list = [0.14, 0.24, 0.34, 0.44, 0.54]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None], [3, 3, None], [4, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.14,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_5():
    # Test case with LiDAR drop
    image_timestamp_list = [0.07, 0.17, 0.27, 0.37, 0.47]
    lidar_timestamp_list = [0.0, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [2, 1, None], [3, 2, None], [4, 3, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.07,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_6():
    # Test case with successive Image drop
    image_timestamp_list = [0.07, 0.37, 0.47]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [None, 1, 0.17], [None, 2, 0.27], [1, 3, None], [2, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.07,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_7():
    # Test case with camera jitter
    image_timestamp_list = [0.07, 0.17, 0.289, 0.37, 0.47]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None], [3, 3, None], [4, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.07,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_8():
    # Test case with camera jitter
    image_timestamp_list = [-0.01, 0.09, 0.19, 0.29, 0.39]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None], [3, 3, None], [4, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=-0.01,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_9():
    # Test case with camera jitter
    image_timestamp_list = [-0.01, 0.04, 0.09, 0.14, 0.19]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [2, 1, None], [4, 2, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=-0.005,
        system_scan_period_sec=0.05,
        max_camera_jitter_sec=0.005,
        num_load_frames=len(lidar_timestamp_list),
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_num_load_frames():
    # Test case with num_load_frames
    image_timestamp_list = [0.07, 0.17, 0.27, 0.37, 0.47]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None]]
    num_load_frames = 3

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.07,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=num_load_frames,
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_get_lidar_camera_synced_frame_info_too_large_num_load_frames():
    # Test case where len(lidar_timestamp_list) >= num_load_frames > len(image_timestamp_list)
    image_timestamp_list = [0.552, 0.652]
    lidar_timestamp_list = [0.4, 0.499, 0.6]

    expected = [[0, 0, None], [1, 1, None]]
    num_load_frames = 3

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        lidar_to_camera_latency_sec=0.162,
        system_scan_period_sec=0.1,
        max_camera_jitter_sec=0.03,
        num_load_frames=num_load_frames,
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)


def test_camera_20fps_lidar_10fps():
    # Camera: 0.00, 0.05, 0.10, 0.15, 0.20
    image_ts = [0.00, 0.05, 0.10, 0.15, 0.20]
    # LiDAR: 0.00, 0.10, 0.20
    lidar_ts = [0.00, 0.10, 0.20]

    result = misc_utils.get_lidar_camera_frame_info_async(
        image_timestamp_list=image_ts,
        lidar_timestamp_list=lidar_ts,
        lidar_to_camera_latency=0.0,
        max_camera_jitter=0.01,
        camera_scan_period=0.05,
        num_load_image_frames=len(image_ts),
        num_load_lidar_frames=len(lidar_ts),
        msg_display_interval=10,
    )

    expected = [
        (0, 0, None),  # image 0 ↔ lidar 0
        (1, None, None),  # image 1 unmatched
        (2, 1, None),  # image 2 ↔ lidar 1
        (3, None, None),  # image 3 unmatched
        (4, 2, None),  # image 4 ↔ lidar 2
    ]

    assert result == expected, f"Expected {expected}, but got {result}"


def test_camera_20fps_lidar_2frames_10fps_camera_2frames():
    # Camera: 0.00, 0.05, 0.10, 0.15, 0.20
    image_ts = [0.00, 0.05, 0.10, 0.15, 0.20]
    # LiDAR: 0.00, 0.10, 0.20
    lidar_ts = [0.00, 0.10, 0.20]

    result = misc_utils.get_lidar_camera_frame_info_async(
        image_timestamp_list=image_ts,
        lidar_timestamp_list=lidar_ts,
        lidar_to_camera_latency=0.0,
        max_camera_jitter=0.01,
        camera_scan_period=0.05,
        num_load_image_frames=2,
        num_load_lidar_frames=2,
        msg_display_interval=10,
    )

    expected = [
        (0, 0, None),  # image 0 ↔ lidar 0
        (1, None, None),  # image 1 unmatched
        (None, 1, 0.1),  # lidar 1, no more image frames
    ]

    assert result == expected, f"Expected {expected}, but got {result}"


def test_camera_20fps_lidar_2frames_10fps_camera_4frames():
    # Camera: 0.00, 0.05, 0.10, 0.15, 0.20
    image_ts = [0.00, 0.05, 0.10, 0.15, 0.20]
    # LiDAR: 0.00, 0.10, 0.20
    lidar_ts = [0.00, 0.10, 0.20]

    result = misc_utils.get_lidar_camera_frame_info_async(
        image_timestamp_list=image_ts,
        lidar_timestamp_list=lidar_ts,
        lidar_to_camera_latency=0.0,
        max_camera_jitter=0.01,
        camera_scan_period=0.05,
        num_load_image_frames=4,
        num_load_lidar_frames=2,
        msg_display_interval=10,
    )

    expected = [
        (0, 0, None),  # image 0 ↔ lidar 0
        (1, None, None),  # image 1 unmatched
        (2, 1, None),  # image 2 ↔ lidar 1
        (3, None, None),  # image 3 unmatched
    ]

    assert result == expected, f"Expected {expected}, but got {result}"


def test_first_lidar_frame_dropped():
    # Camera: 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35
    # 1st and 2nd frames will be removed
    image_ts = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    # LiDAR: 0.10, 0.20
    lidar_ts = [0.10, 0.20]

    result = misc_utils.get_lidar_camera_frame_info_async(
        image_timestamp_list=image_ts,
        lidar_timestamp_list=lidar_ts,
        lidar_to_camera_latency=0.0,
        max_camera_jitter=0.005,
        camera_scan_period=0.05,
        num_load_image_frames=100,
        num_load_lidar_frames=50,
        msg_display_interval=1,
    )

    expected = [
        (2, 0, None),
        (3, None, None),
        (4, 1, None),
        (5, None, None),
        (6, None, None),
        (7, None, None),
    ]

    print(f"image_ts: {image_ts}, lidar_ts: {lidar_ts}")
    print(f"result: {result}")
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "image_ts, lidar_ts, expected",
    [
        # missing LiDAR frames
        (
            [0.00, 0.05, 0.10, 0.15, 0.20],  # Camera: 0.00, 0.05, 0.10, 0.15, 0.20
            [0.00, 0.10],  # LiDAR: 0.00, 0.10
            [
                (0, 0, None),
                (1, None, None),
                (2, 1, None),
                (3, None, None),
                (4, None, None),  # LiDAR 2 is missing
            ],
        ),
        # missing the first Camera frame
        (
            [0.05, 0.10, 0.15, 0.20],  # Camera: 0.05, 0.10, 0.15, 0.20
            [0.00, 0.10, 0.20],  # LiDAR: 0.00, 0.10, 0.20
            [
                (None, 0, 0.00),  # Camera 0 is missing
                (0, None, None),
                (1, 1, None),
                (2, None, None),
                (3, 2, None),
            ],
        ),
        # missing Camera frames
        (
            [0.00, 0.05, 0.15, 0.20],  # Camera: 0.00, 0.05, 0.15, 0.20
            [0.00, 0.10, 0.20],  # LiDAR: 0.00, 0.10, 0.20
            [
                (0, 0, None),
                (1, None, None),
                (None, 1, 0.10),
                (2, None, None),
                (3, 2, None),
            ],
        ),
        # missing Camera and LiDAR frames
        (
            [0.00, 0.05, 0.15],  # Camera: 0.00, 0.05, 0.15
            [0.00, 0.20],  # LiDAR: 0.00, 0.20
            [
                (0, 0, None),
                (1, None, None),
                (2, None, None),
                (None, 1, 0.20),
            ],
        ),
    ],
)
def test_lidar_camera_edge_cases(image_ts, lidar_ts, expected):
    result = misc_utils.get_lidar_camera_frame_info_async(
        image_timestamp_list=image_ts,
        lidar_timestamp_list=lidar_ts,
        lidar_to_camera_latency=0.0,
        max_camera_jitter=0.01,
        camera_scan_period=0.05,
        num_load_image_frames=len(image_ts),
        num_load_lidar_frames=len(lidar_ts),
        msg_display_interval=1,
    )

    assert result == expected, f"Expected {expected}, but got {result}"
