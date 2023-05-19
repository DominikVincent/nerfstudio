#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

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
	--pipeline.datamanager.train-num-images-to-sample-from 6 \
	--pipeline.datamanager.train-num-times-to-repeat-images 4 \
	--pipeline.datamanager.eval-num-images-to-sample-from 6 \
	--pipeline.datamanager.eval-num-times-to-repeat-images 4 \
	--pipeline.images-to-sample-during-eval-image 6 \
	--pipeline.datamanager.train-num-rays-per-batch 131072 \
	--pipeline.datamanager.eval-num-rays-per-batch 131072 \
	--pipeline.model.eval-num-rays-per-chunk 131072 \
	--pipeline.model.sampler.surface-sampling True \
	--pipeline.model.sampler.samples-per-ray 10 \
	--pipeline.model.batching-mode "random" \
	--pipeline.model.batch_size 10000 \
	--pipeline.model.mode semantics \
	--pipeline.model.pretrain False  \
	--pipeline.model.feature-generator-config.use-rgb False \
	--pipeline.model.feature-generator-config.use-dir-encoding False \
	--pipeline.model.feature-generator-config.use-pos-encoding False \
	--pipeline.model.feature-generator-config.use-density True \
	--pipeline.model.feature-generator-config.out-rgb-dim 16 \
	--pipeline.model.feature-generator-config.out-density-dim 1 \
	--pipeline.model.feature-generator-config.rot-augmentation True \
	--pipeline.model.space-partitioning "random" \
	--pipeline.model.feature-transformer-model "pointnet" \
	--pipeline.model.feature-transformer-pointnet-config.out_feature_channels 64 \
	--pipeline.model.feature-transformer-pointnet-config.radius_scale 0.2 \
	nesf-data \
	--data-config $DATA_CONFIG 
