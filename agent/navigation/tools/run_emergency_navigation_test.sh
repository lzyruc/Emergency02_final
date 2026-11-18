#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xrocs_root="$(cd "${script_dir}/../../../.." && pwd)"

cd "${xrocs_root}"
exec env PYTHONPATH=. python3 \
  xrocs/agent/navigation/tools/demo_navigation_graph.py \
  xrocs/agent/navigation/data/logs/emergency.navigation.scene.json \
  --navigation-mode move_by \
  --linear-speed 0.2 \
  --angular-speed 0.3 \
  --nav-timeout 30 \
  --no-compress-path \
  "$@"
