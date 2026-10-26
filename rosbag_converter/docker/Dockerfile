ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libboost-dev \
    libcgal-dev \
    libeigen3-dev \
    libopencv-dev \
    libpcl-dev \
    libpng-dev \
    nlohmann-json3-dev \
    pybind11-dev \
    python3-colcon-common-extensions \
    python3-dev \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-venv \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN <<'EOF'
set -euo pipefail

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  rosdep init
fi

for attempt in 1 2 3; do
  if rosdep update --rosdistro "${ROS_DISTRO}"; then
    exit 0
  fi
  echo "rosdep update failed on attempt ${attempt}; retrying..." >&2
  sleep $((attempt * 5))
done

rosdep update --rosdistro "${ROS_DISTRO}" --verbose
EOF

RUN python3 -m pip install --no-cache-dir --break-system-packages uv || \
    python3 -m pip install --no-cache-dir uv

COPY workspace_build_smoke.sh /usr/local/bin/workspace_build_smoke
RUN chmod +x /usr/local/bin/workspace_build_smoke

WORKDIR /workspace
CMD ["/bin/bash"]
