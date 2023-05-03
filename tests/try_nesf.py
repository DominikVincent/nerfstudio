import sys
from pathlib import Path
from typing import cast

import torch_geometric

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.models.nesf import NeuralSemanticFieldConfig
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
    data_config_path = Path(
        "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_10.json"
    )
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_100.json"
    # )

    OUTPUT_DIR = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/")
    DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11")

    trainConfig = method_configs["nesf"]
    trainConfig.pipeline.model = cast(NeuralSemanticFieldConfig, trainConfig.pipeline.model)
    # trainConfig = method_configs["nesf_density"]
    trainConfig.vis = vis
    trainConfig.data = DATA_PATH
    trainConfig.output_dir = OUTPUT_DIR
    trainConfig.machine.num_gpus = 1
    trainConfig.pipeline.datamanager.dataparser.data_config = data_config_path
    trainConfig.steps_per_eval_batch = 10
    trainConfig.steps_per_eval_image = 20
    
    trainConfig.pipeline.model.pretrain = False
    trainConfig.pipeline.model.mode = "semantics"
    trainConfig.pipeline.model.batching_mode = "off"
    trainConfig.pipeline.model.batch_size = 6144
    
    trainConfig.pipeline.model.sampler.surface_sampling = True
    trainConfig.pipeline.model.sampler.samples_per_ray = 10
    
    trainConfig.pipeline.model.masker_config.mask_ratio = 0.75
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
    trainConfig.pipeline.model.feature_transformer_custom_config.num_heads = 8
    trainConfig.pipeline.model.feature_transformer_custom_config.dim_feed_forward = 128
    trainConfig.pipeline.model.feature_transformer_custom_config.feature_dim = 128
    
    trainConfig.pipeline.model.feature_transformer_stratified_config.grid_size = 0.02
    trainConfig.pipeline.model.feature_transformer_stratified_config.window_size = 4
    trainConfig.pipeline.model.feature_transformer_stratified_config.quant_size = 0.005
    
    trainConfig.pipeline.model.feature_decoder_model = "custom"
    trainConfig.pipeline.model.feature_decoder_custom_config.num_layers = 2
    trainConfig.pipeline.model.feature_decoder_custom_config.num_heads = 8
    trainConfig.pipeline.model.feature_decoder_custom_config.dim_feed_forward = 128
    trainConfig.pipeline.model.feature_decoder_custom_config.feature_dim = 128
    
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
    args = sys.argv
    run_nesf(args[1])
    run_nesf(args[1])
    run_nesf(args[1])
