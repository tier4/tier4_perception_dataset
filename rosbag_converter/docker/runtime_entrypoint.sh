#!/usr/bin/env bash
set -euo pipefail

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "/opt/t4_ws/install/setup.bash"
set -u

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/t4_ws/.venv}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export PATH="/opt/t4_ws/.venv/bin:${PATH}"
cd /opt/t4_ws

if [[ "${1:-}" == "uv" ]]; then
  exec "$@"
fi

exec "$@"
