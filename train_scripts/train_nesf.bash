#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio3

# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_100.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_5.json"
DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"

ns-train nesf --data /data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf/0  \
	--output-dir /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/ \
	--vis wandb \
	--steps-per-eval-batch 100 \
    --steps-per-eval-image 500 \
    --steps-per-save 10000 \
    --max-num-iterations 5000000 \
	--pipeline.datamanager.steps-per-model 1 \
	--pipeline.datamanager.train-num-images-to-sample-from 4 \
	--pipeline.datamanager.train-num-times-to-repeat-images 4 \
	--pipeline.datamanager.eval-num-images-to-sample-from 4 \
	--pipeline.datamanager.eval-num-times-to-repeat-images 4 \
	--pipeline.datamanager.train-num-rays-per-batch 40962 \
	--pipeline.datamanager.eval-num-rays-per-batch 40962 \
	--pipeline.model.eval-num-rays-per-chunk 40962 \
	--pipeline.model.surface_sampling True \
	--pipeline.model.mode semantics \
	--pipeline.model.pretrain False  \
	--pipeline.model.use-feature-rgb True \
	--pipeline.model.use-feature-dir True \
	--pipeline.model.use-feature-pos True \
	--pipeline.model.use-feature-density False \
	--pipeline.model.rgb-feature-dim 16 \
	--pipeline.model.density-feature-dim 8 \
	--pipeline.model.rot-augmentation True \
	--pipeline.model.space-partitioning "evenly" \
	--pipeline.model.feature_transformer pointnet \
	--pipeline.model.feature-transformer-num-layers 8 \
	--pipeline.model.feature-transformer-num-heads 8 \
	--pipeline.model.feature-transformer-dim-feed-forward 256 \
	--pipeline.model.feature-transformer-dropout-rate 0.2 \
	--pipeline.model.feature-transformer-feature-dim 128 \
	--pipeline.model.decoder-feature-transformer-num-layers 2 \
	--pipeline.model.decoder-feature-transformer-num-heads 2 \
	--pipeline.model.decoder-feature-transformer-dim-feed-forward 32 \
	--pipeline.model.decoder-feature-transformer-dropout-rate 0.2 \
	--pipeline.model.decoder-feature-transformer-feature-dim 32 \
	--pipeline.model.batching-mode "off" \
	--pipeline.model.batch_size 4096 \
	--pipeline.model.samples_per_ray 10 \
	nesf-data \
	--data-config $DATA_CONFIG 
