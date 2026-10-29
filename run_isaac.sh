#!/usr/bin/env bash
# Run a script with Isaac Sim's bundled python and this workspace on PYTHONPATH.
#   ./run_isaac.sh scripts/isaac_nav.py --planner hybrid_astar
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/isaacsim}"
if [[ ! -x "$ISAACSIM_PATH/python.sh" ]]; then
  echo "Isaac Sim not found at ISAACSIM_PATH=$ISAACSIM_PATH" >&2
  exit 1
fi
# Isaac's python.sh must not run inside an activated conda env.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONHOME

# This Isaac install lost some bundled packages (numpy, torch, PIL, ...). Keep private copies for this
# workspace only, so neither the Isaac install nor the Isaac Lab conda env is modified.
PYDEPS="$WS/.isaac_pydeps"
if [[ ! -d "$PYDEPS/numpy" || ! -d "$PYDEPS/torch" || ! -d "$PYDEPS/packaging" ]]; then
  echo "[run_isaac] installing isaac_requirements.txt into $PYDEPS (one-time)"
  "$ISAACSIM_PATH/python.sh" -m pip install -q --no-deps --target "$PYDEPS" -r "$WS/isaac_requirements.txt"
fi

export PYTHONPATH="$WS:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
cd "$WS"
exec "$ISAACSIM_PATH/python.sh" "$@"
