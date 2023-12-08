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
        if expected[i][2] is None: # If the timestamp is dummy
            assert synced_frame_info_list[i][2] is None
        else:
            assert abs(synced_frame_info_list[i][2] - expected[i][2]) < 1e-6

def test_get_lidar_camera_synced_frame_info_1():
    image_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [1, 1, None], [2, 2, None], [3, 3, None], [4, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        accept_frame_drop=True,
        num_load_frames=5,
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)

def test_get_lidar_camera_synced_frame_info_2():
    image_timestamp_list = [0.0, 0.2, 0.4]
    lidar_timestamp_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected = [[0, 0, None], [None, 1, 0.1], [1, 2, None], [None, 3, 0.3], [2, 4, None]]

    synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
        image_timestamp_list=image_timestamp_list,
        lidar_timestamp_list=lidar_timestamp_list,
        accept_frame_drop=True,
        num_load_frames=5,
    )
    assert_synced_frame_info_list(expected, synced_frame_info_list)
