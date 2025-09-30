# Tier4 Perception Dataset Converter
FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    ros-humble-sensor-msgs-py \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-radar-msgs \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Build ROS2 custom messages
WORKDIR /workspace
COPY build_depends.repos .
RUN mkdir -p src && \
    vcs import src < build_depends.repos && \
    . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install \
                 --cmake-args -DCMAKE_BUILD_TYPE=Release \
                 --packages-up-to \
                   autoware_auto_perception_msgs \
                   autoware_perception_msgs \
                   tier4_perception_msgs \
                   oxts_msgs \
                   vehicle_msgs

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry poetry-plugin-export && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY perception_dataset/ ./perception_dataset/
COPY config/ ./config/

# Setup environment
ENV PYTHONPATH="/workspace/install/lib/python3.10/site-packages:${PYTHONPATH}"

# Create entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/ros/humble/setup.bash\n\
source /workspace/install/setup.bash\n\
exec python3 -m perception_dataset.convert "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]