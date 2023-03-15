import sys
from pathlib import Path

import wandb
from nerfstudio.configs.method_configs import method_configs


def main():
    wandb.init(project="nesf-sweep")
    config = wandb.config

    data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config.json")

    OUTPUT_DIR = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/")
    DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11")

    trainConfig = method_configs["nesf"]
    trainConfig.vis = "wandb"
    trainConfig.data = DATA_PATH
    trainConfig.output_dir = OUTPUT_DIR
    trainConfig.pipeline.datamanager.dataparser.data_config = data_config_path
    trainConfig.steps_per_eval_all_images = 25000

    trainConfig.set_timestamp()
    trainConfig.pipeline.datamanager.dataparser.data = trainConfig.data
    trainConfig.save_config()

    # update config with dict
    for key, value in config.items():
        if key.startswith("model"):
            key = key.replace("model_", "")
            if hasattr(trainConfig.pipeline.model, key):
                setattr(trainConfig.pipeline.model, key, value)
                print(f"Set {key} to {value} in model config")
            else:
                print(f"WARNING: {key} not found in model config")

    trainConfig.optimizers["feature_network"]["optimizer"].lr = config["lr"]
    trainConfig.optimizers["feature_transformer"]["optimizer"].lr = config["lr"]
    trainConfig.optimizers["learned_low_density_params"]["optimizer"].lr = config["lr"]
    trainer = trainConfig.setup(local_rank=0, world_size=1)
    trainer.setup()
    trainer.train()


# rgb
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "Eval Loss"},
#     "parameters": {
#         "lr": {"values": [1e-3]},
#         "model_rgb": {"value": True},
#         "model_rgb_feature_dim": {"values": [4, 8, 16]},
#         "model_use_feature_pos": {"values": [True, False]},
#         "model_use_feature_dir": {"values": [True, False]},
#         "model_feature_transformer_num_layers": {"values": [2, 4, 8]},
#         "model_feature_transformer_num_heads": {"values": [2, 4, 8]},
#         "model_feature_transformer_dim_feed_forward": {"values": [16, 32, 64, 128]},
#         "model_feature_transformer_feature_dim": {"values": [4, 8, 16, 32, 64, 128]},
#     },
# }

# semantic
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "Eval Loss"},
    "parameters": {
        "lr": {"value": 1e-3},
        "model_rgb": {"value": False},
        "model_rgb_feature_dim": {"values": [8, 16]},
        "model_use_feature_pos": {"values": [True, False]},
        "model_use_feature_dir": {"values": [True, False]},
        "model_feature_transformer_num_layers": {"values": [2, 4, 8]},
        "model_feature_transformer_num_heads": {"values": [2, 4, 8]},
        "model_feature_transformer_dim_feed_forward": {"values": [32, 64, 128]},
        "model_feature_transformer_feature_dim": {"values": [32, 64, 128, 256]},
    },
}

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        sweep_id = args[1]
        wandb.agent(sweep_id, function=main, count=30)
    else:
        sweep_id = wandb.sweep(sweep_configuration, project="nerfstudio-project")
        print("sweep_id:", sweep_id)
