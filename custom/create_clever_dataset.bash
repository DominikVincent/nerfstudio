#!/bin/bash

sudo docker run --rm --interactive --gpus '"device=0"' \
            --env KUBRIC_USE_GPU=1 \
            --volume "$(pwd):/kubric" \
            --volume "/data/vision/polina/scratch/clintonw/datasets/kubric-public:/kubric_data" \
            --volume "/data/vision/polina/projects/wmh/dhollidt/datasets:/out_dir" \
            --user $(id -u):$(id -g) \
            --volume "$PWD:/kubric" \
            kubricdockerhub/kubruntu_gpu \
            python3 static_clevr.py