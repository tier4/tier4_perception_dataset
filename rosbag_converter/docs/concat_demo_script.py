from autoware_pointcloud_preprocessor.concatenate_pointclouds import (
    CollectorStatus,
    Concatenator,
)

# tf_static: {sensor_frame -> geometry_msgs/TransformStamped}, each with
#   header.frame_id = output_frame, child_frame_id = sensor_frame
# (offline_concat.py builds these by composing the /tf_static chain; see build_static_transforms)

concat = Concatenator(
    input_topics=lidar_topics,           # list[str], order matters for advanced offsets/noise
    output_frame="base_link",
    tf_static=tf_static,                 # dict[str, TransformStamped]
    timeout_sec=0.2,                     # per-group timeout, in arrival time
    is_motion_compensated=True,
    matching_strategy="advanced",        # or "naive"
    lidar_timestamp_offsets=offsets,     # advanced only: one per input topic (seconds), same order
    lidar_timestamp_noise_window=noise,  # advanced only: one per input topic (seconds), same order
)

# Feed messages in ARRIVAL order. arrival_time is in seconds (e.g. bag record ts * 1e-9).
for topic, arrival_time_sec, msg in stream:        # your iterator, in record order
    if topic == twist_topic:
        concat.process_twist(twist_msg)            # TwistWithCovarianceStamped
        continue
    # msg is a sensor_msgs/PointCloud2 (rclpy object)
    for frame in concat.process_cloud(topic, msg, arrival_time=arrival_time_sec):
        cloud = frame.result.concatenated_cloud    # sensor_msgs/PointCloud2 (or None)
        if frame.status == CollectorStatus.COMPLETE:
            ...  # all input topics contributed
        elif frame.status == CollectorStatus.TIMEOUT:
            ...  # group closed before all topics arrived
        # frame.result also carries: concatenation_info, no_twist_available,
        # twist_time_gap_too_large, topic_to_original_stamp, dropped_frames_missing_transform

# Once the stream is exhausted, flush any still-open groups (emitted as TIMEOUT):
for frame in concat.flush():
    ...