import sys
from pathlib import Path

from nerfstudio.configs.method_configs import method_configs


def run_nesf(vis: str = "wandb"):
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_5.json")
    # data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_nesf_train_100.json")
    # data_config_path = Path(
    #     "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_1.json"
    # )
    data_config_path = Path(
        "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/klever_depth_nesf_train_10.json"
    )

    OUTPUT_DIR = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/")
    DATA_PATH = Path("/data/vision/polina/projects/wmh/dhollidt/datasets/klevr/11")

    trainConfig = method_configs["nesf"]
    # trainConfig = method_configs["nesf_density"]
    trainConfig.vis = vis
    trainConfig.data = DATA_PATH
    trainConfig.output_dir = OUTPUT_DIR
    trainConfig.pipeline.datamanager.dataparser.data_config = data_config_path
    trainConfig.set_timestamp()
    trainConfig.pipeline.datamanager.dataparser.data = trainConfig.data
    trainConfig.save_config()

    trainer = trainConfig.setup(local_rank=0, world_size=1)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    args = sys.argv
    run_nesf(args[1])
