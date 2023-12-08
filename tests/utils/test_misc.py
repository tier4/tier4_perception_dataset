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

def test_get_lidar_camera_synced_frame_info_1():
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    image_timestamp_list = [0.0, 0.2, 0.4]
    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        lidar_timestamp_list=lidar_timestamp_list,
        image_timestamp_list=image_timestamp_list,
        start_timestamp=0.0,
        camera_latency_sec=0.0,
        accept_frame_drop=False,
        timestamp_diff=0.15,
        num_load_frames=0,
    )

    expected = [[0, 0, None], [2, 1, None], [4, 2, None]]
    assert synced_frame_info_list == expected
