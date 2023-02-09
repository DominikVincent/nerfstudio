from pathlib import Path

from nerfstudio.configs.method_configs import method_configs


def run_nesf():
    data_config_path = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/playground/nesf_test_config.json")

    OUTPUT_DIR = Path("/data/vision/polina/projects/wmh/dhollidt/documents/nerf/playground/own_loading/")
    DATA_PATH = Path("/data/vision/polina/scratch/clintonw/datasets/kubric/klevr/0")

    trainConfig = method_configs["nesf"]
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
    run_nesf()