import os
import sys
from pathlib import Path
from typing import cast

import torch_geometric

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.models.nesf import NeuralSemanticFieldConfig
from nerfstudio.pipelines.nesf_pipeline import NesfPipelineConfig
from scripts.train import main as train_main


def run_nesf(vis: str = "wandb"):
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_5.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_100.json")
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_1.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_1_normals.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_10.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"
    # )

    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_normal_nesf_train_10.json"
    # )    
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_10_10.json"
    # )
    data_config_path = Path(
        "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_10_270.json"
    )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_1_270.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_100_270.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_200_270.json"
    # )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/toybox-5_nesf_train_500_270.json"
    # )

    OUTPUT_DIR = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/")
    DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11")

    trainConfig = method_configs["nesf"]
    trainConfig.pipeline.model = cast(NeuralSemanticFieldConfig, trainConfig.pipeline.model)
    trainConfig.pipeline = cast(NesfPipelineConfig, trainConfig.pipeline)
    # trainConfig = method_configs["nesf_density"]
    trainConfig.vis = vis
    trainConfig.data = DATA_PATH
    trainConfig.output_dir = OUTPUT_DIR
    trainConfig.machine.num_gpus = 1
    trainConfig.pipeline.datamanager.dataparser.data_config = data_config_path
    trainConfig.steps_per_eval_batch = 10
    trainConfig.steps_per_eval_image = 20
    
    trainConfig.pipeline.datamanager.use_sample_mask = False
    trainConfig.pipeline.datamanager.sample_mask_ground_percentage = 0.2
    trainConfig.pipeline.datamanager.steps_per_model = 1
    trainConfig.pipeline.datamanager.train_num_images_to_sample_from = 8
    trainConfig.pipeline.datamanager.train_num_times_to_repeat_images = 1
    trainConfig.pipeline.datamanager.eval_num_images_to_sample_from = 8
    trainConfig.pipeline.datamanager.eval_num_times_to_repeat_images = 1
    
    trainConfig.pipeline.model.pretrain = True
    trainConfig.pipeline.model.mode = "rgb"
    trainConfig.pipeline.model.batching_mode = "off"
    trainConfig.pipeline.model.batch_size = 2048
    
    trainConfig.pipeline.model.sampler.surface_sampling = True
    trainConfig.pipeline.model.sampler.get_normals = False
    trainConfig.pipeline.model.sampler.samples_per_ray = 16
    trainConfig.pipeline.model.sampler.ground_removal_mode = "ransac"
    trainConfig.pipeline.model.sampler.ground_tolerance = 0.01
    trainConfig.pipeline.model.sampler.surface_threshold = 0.5
    
    trainConfig.pipeline.model.masker_config.mode = "patch"
    trainConfig.pipeline.model.masker_config.mask_ratio = 0.5
    trainConfig.pipeline.model.rgb_prediction = "integration"
    trainConfig.pipeline.model.density_prediction = "direct"
    
    trainConfig.pipeline.model.feature_generator_config.use_rgb = True
    trainConfig.pipeline.model.feature_generator_config.use_density = True
    trainConfig.pipeline.model.feature_generator_config.use_pos_encoding = True
    trainConfig.pipeline.model.feature_generator_config.use_dir_encoding = True
    trainConfig.pipeline.model.feature_generator_config.use_normal_encoding = False
    
    trainConfig.pipeline.model.feature_transformer_model = "stratified"
    trainConfig.pipeline.model.feature_transformer_custom_config.num_layers = 6
    trainConfig.pipeline.model.feature_transformer_custom_config.num_heads = 8
    trainConfig.pipeline.model.feature_transformer_custom_config.dim_feed_forward = 128
    trainConfig.pipeline.model.feature_transformer_custom_config.feature_dim = 128
    
    trainConfig.pipeline.model.feature_transformer_stratified_config.grid_size = 0.006
    trainConfig.pipeline.model.feature_transformer_stratified_config.window_size = 5
    trainConfig.pipeline.model.feature_transformer_stratified_config.quant_size = 0.005
    trainConfig.pipeline.model.feature_transformer_stratified_config.num_layers = 4
    
    
    trainConfig.pipeline.model.feature_decoder_model = "stratified"
    trainConfig.pipeline.model.feature_decoder_custom_config.num_layers = 2
    trainConfig.pipeline.model.feature_decoder_custom_config.num_heads = 4
    trainConfig.pipeline.model.feature_decoder_custom_config.dim_feed_forward = 128
    trainConfig.pipeline.model.feature_decoder_custom_config.feature_dim = 128
    
    trainConfig.pipeline.model.feature_decoder_stratified_config.grid_size = 0.006
    trainConfig.pipeline.model.feature_decoder_stratified_config.window_size = 5
    trainConfig.pipeline.model.feature_decoder_stratified_config.quant_size = 0.005
    trainConfig.pipeline.model.feature_decoder_stratified_config.num_layers = 3
    trainConfig.pipeline.model.feature_decoder_stratified_config.depths = [2,2,4]
    
    trainConfig.set_timestamp()
    trainConfig.pipeline.datamanager.dataparser.data = trainConfig.data
    trainConfig.pipeline.datamanager.dataparser.train_split_percentage = trainConfig.data
    # trainConfig.pipeline.model.feature_generator_config.visualize_point_batch = True
    # trainConfig.pipeline.model.debug_show_image = True
    trainConfig.save_config()
        
    train_main(trainConfig)
    
    # trainer = trainConfig.setup(local_rank=0, world_size=1)
    # trainer.setup()
    # trainer.train()


if __name__ == "__main__":
    print("PID: ", os.getpid())
    args = sys.argv
    run_nesf(args[1])
