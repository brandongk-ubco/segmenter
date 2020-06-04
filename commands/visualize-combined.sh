#!/usr/bin/env bash

conda activate segmenter

set -euo pipefail

launch visualize --visualizer combined-layer-output
launch visualize --visualizer combined-f1