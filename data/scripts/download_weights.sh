#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Download latest models from https://github.com/ultralytics/assets/releases
# Example usage: bash ultralytics/data/scripts/download_weights.sh
# parent
# └── weights
#     ├── ← downloads here
#     ├──
#     └── ...

python - <<EOF
from ultralytics.utils.downloads import attempt_download_asset

assets = [f'yolov8{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '-cls', '-seg', '-pose')]
for x in assets:
    attempt_download_asset(f'weights/{x}')

EOF
