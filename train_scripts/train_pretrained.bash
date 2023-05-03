#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

ns-train nesf --load_config /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/tmp/nesf/2023-04-29_175231/auto_semantic_config_small_10_1.yml