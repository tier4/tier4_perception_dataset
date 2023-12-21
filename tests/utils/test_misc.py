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
