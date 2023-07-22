#!/bin/bash

source /data/vision/polina/projects/wmh/dhollidt/conda/bin/activate
conda activate nerfstudio2

# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_100.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_10.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_100_270.json"
DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_500_270.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_100_270.json"
# DATA_CONFIG="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_2_train_100_10.json"


# NP_SURFACE=True
# Q_SURFACE=True
# SAMPLES_PER_NP_RAY=24
# SAMPLES_PER_QUERY_RAY=24
# RAYS_NP=65536
# RAYS_QUERY=32768

# NP_SURFACE=True
# Q_SURFACE=False
# SAMPLES_PER_NP_RAY=24
# SAMPLES_PER_QUERY_RAY=8
# RAYS_NP=65536
# RAYS_QUERY=5120

NP_SURFACE=False
Q_SURFACE=True
SAMPLES_PER_NP_RAY=8
SAMPLES_PER_QUERY_RAY=24
RAYS_NP=24576
RAYS_QUERY=32768

# NP_SURFACE=False
# Q_SURFACE=False
# SAMPLES_PER_NP_RAY=8
# SAMPLES_PER_QUERY_RAY=8
# RAYS_NP=24576
# RAYS_QUERY=8192

ns-train nesf --data /data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf/0  \
	--output-dir /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/ \
	--vis wandb \
	--machine.num-gpus 1 \
	--steps-per-eval-batch 100 \
    --steps-per-eval-image 500 \
    --steps-per-save 5000 \
    --max-num-iterations 5000000 \
	--pipeline.datamanager.steps-per-model 1 \
	--pipeline.datamanager.train-num-images-to-sample-from 8 \
	--pipeline.datamanager.train-num-times-to-repeat-images 4 \
	--pipeline.datamanager.eval-num-images-to-sample-from 8 \
	--pipeline.datamanager.eval-num-times-to-repeat-images 4 \
	--pipeline.datamanager.num-rays-per-neural-pointcloud $RAYS_NP \
	--pipeline.datamanager.num-rays-per-query $RAYS_QUERY \
	--pipeline.model.eval-num-rays-per-chunk $RAYS_QUERY \
	--pipeline.model.sampler.surface-sampling $NP_SURFACE \
	--pipeline.model.sampler.samples-per-ray $SAMPLES_PER_NP_RAY \
	--pipeline.model.sampler.get-normals False \
	--pipeline.model.sampler.ground_removal_mode "ransac" \
	--pipeline.model.sampler.ground-points-count 500000 \
	--pipeline.model.sampler.ground-tolerance 0.004 \
	--pipeline.model.sampler.surface-threshold 0.5 \
	--pipeline.model.batching-mode "off" \
	--pipeline.model.batch_size 16384 \
	--pipeline.model.mode semantics \
	--pipeline.model.proximity-loss True \
	--pipeline.model.feature-generator-config.jitter 0.000 \
	--pipeline.model.pretrain False  \
	--pipeline.model.feature-generator-config.use-rgb True \
	--pipeline.model.feature-generator-config.use-dir-encoding True \
	--pipeline.model.feature-generator-config.use-pos-encoding True \
	--pipeline.model.feature-generator-config.pos-encoder "sin" \
	--pipeline.model.feature-generator-config.use-normal-encoding False \
	--pipeline.model.feature-generator-config.use-density True \
	--pipeline.model.feature-generator-config.out-rgb-dim 16 \
	--pipeline.model.feature-generator-config.out-density-dim 8 \
	--pipeline.model.feature-generator-config.rot-augmentation True \
	--pipeline.model.space-partitioning "random" \
	--pipeline.model.feature-transformer-model "stratified" \
	--pipeline.model.feature-transformer-stratified-config.grid_size 0.005 \
	--pipeline.model.feature-transformer-stratified-config.quant_size 0.0001 \
	--pipeline.model.feature-transformer-stratified-config.window_size 4 \
	--pipeline.model.feature-transformer-stratified-config.load_dir "/data/vision/polina/projects/wmh/dhollidt/documents/Stratified-Transformer/weights/s3dis_model_best.pth" \
	--pipeline.model.use_field2field True \
	--pipeline.model.field2field_sampler.surface_sampling $Q_SURFACE \
	--pipeline.model.field2field_sampler.samples_per_ray $SAMPLES_PER_QUERY_RAY \
	--pipeline.model.field2field_sampler.ground_removal_mode "ransac" \
	--pipeline.model.field2field_sampler.ground_points_count 500000 \
	--pipeline.model.field2field_sampler.ground_tolerance 0.004 \
	--pipeline.model.field2field_sampler.surface_threshold 0.5 \
	--pipeline.model.field2field_sampler.max_points 12288 \
	--pipeline.model.field2field_config.knn 64 \
	--pipeline.model.field2field_config.transformer.num_layers 3 \
	--pipeline.model.field2field_config.transformer.num_heads 4 \
	--pipeline.model.field2field_config.transformer.dim_feed_forward 96 \
	--pipeline.model.field2field_config.mode "transformer" \
	nesf-data \
	--data-config $DATA_CONFIG 

# needed if we want to use 100% of the pretrained model weights. Just use rgb
# --pipeline.model.feature-transformer-stratified-config.grid_size 0.0054 \
# --pipeline.model.feature-transformer-stratified-config.quant_size 0.001 \
# --pipeline.model.feature-generator-config.out-rgb-dim 3 \
# /data/vision/polina/projects/wmh/dhollidt/documents/Stratified-Transformer/weights/s3dis_model_best.pth
