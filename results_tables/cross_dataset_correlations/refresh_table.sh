#!/bin/zsh
set -euo pipefail

artifact_dir="${0:A:h}"
exec '/Users/lailajohnston/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node' \
  "$artifact_dir/build_table.mjs" "$@"
