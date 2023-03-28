#!/usr/bin/env python

import argparse
import multiprocessing
import subprocess
import time
from pathlib import Path
from typing import Union

import yaml

import wandb

# from nerfstudio.utils import writer

# from scripts.eval import ComputePSNR

parser = argparse.ArgumentParser()
parser.add_argument("--proj_name", help="The wandb proj name", type=str, default="dhollidt/mae-models-project")
parser.add_argument("--sweep_id", help="The wandb sweep id", type=str, default="kfsdevg7")
parser.add_argument(
    "--eval_config",
    help="The nesf eval config",
    type=str,
    default="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/data/nesf_test_config_test.json",
)

FIX_CONFIG = True


def get_sweep_runs(sweep_id, project_name):
    api = wandb.Api()
    sweep = api.sweep(project_name + "/" + sweep_id)
    return sweep.runs


def sweep_run_to_path(run):
    return (
        Path(run.config["output_dir"])
        / run.config["experiment_name"].strip("/")
        / run.config["method_name"]
        / run.config["timestamp"]
    )


def ns_eval(config_path, output_path, name=""):
    command = f"ns-eval --load-config {config_path} --output-path {output_path} --use-wandb --name {name}"

    # Execute the command and capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Stream the output in real-time
    while True:
        output = process.stdout.readline().decode()
        error = process.stderr.readline().decode()
        if output == "" and error == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
        if error:
            print(error.strip())

    # Wait for the command to finish and get the return code
    return_code = process.wait()
    return return_code


# def eval_run(path: Path, eval_set: Union[None, Path] = None, run, gpu_index: int = 0):
#   wandb_config = wandb_run.config
#   wandb_name = wandb_run.name + "_eval"
def eval_run(path: Path, eval_set: Union[None, Path] = None, wandb_config=None, wandb_name=None, gpu_index: int = 0):
    # print("GPU: ", gpu_index, "Path: ", path, "writers: ", writer.EVENT_WRITERS)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    out_path = path / "auto_eval_config.yml"
    input_path = path / "config.yml"

    config = yaml.load(input_path.read_text(), Loader=yaml.Loader)

    if eval_set is not None:
        config.pipeline.datamanager.dataparser.data_config = eval_set

    if FIX_CONFIG:
        for key, value in wandb_config["pipeline"]["model"].items():
            if key.startswith("_"):
                continue
            if hasattr(config.pipeline.model, key):
                should_type = type(getattr(config.pipeline.model, key))
                cast_value = should_type(value)
                # print("Should be type: ", should_type, "cast to: ", type(cast_value))
                # print(f"Set {key} to {value} in model config from: {getattr(config.pipeline.model, key)}")
                setattr(config.pipeline.model, key, cast_value)
            else:
                print(f"WARNING: {key} not found in model config")
    # save config as yaml
    out_path.write_text(yaml.dump(config), "utf8")

    # eval = ComputePSNR(load_config=out_path,
    #                    output_path=path / "auto_eval.json",
    #                    save_images=False,
    #                    use_wandb=True)
    # eval.main()
    # print(gpu_index, ": Reseting writers: ", writer.EVENT_WRITERS)
    # writer.reset_writer()
    # print(gpu_index, ": Reset writers", writer.EVENT_WRITERS)
    ns_eval(out_path, path / "auto_eval.json", name=wandb_name)
    print(" ######################### Done with: ", path, " ######################### ")
    return path


def main(project_name: str, sweep_id: str, eval_config: Union[None, Path] = None):
    runs = get_sweep_runs(sweep_id, project_name)
    pool = multiprocessing.Pool(processes=2, maxtasksperchild=1)

    results = []
    for i, run in enumerate(runs):
        gpu_index = i % 3 + 1  # assign the next GPU index
        path = sweep_run_to_path(run)
        print(path)
        result = pool.apply_async(eval_run, args=(path, eval_config, run.config, run.name + "_test", gpu_index))
        results.append(result)
        # eval_run(path, eval_config, run, gpu_index)

    pool.close()
    print("###### CLOSE POOL ######")
    pool.join()
    print("###### JOIN POOL ######")

    # get the results from the async calls
    final_results = [result.get() for result in results]
    print(final_results)


if __name__ == "__main__":
    args = parser.parse_args()
    proj_name = args.proj_name
    sweep_id = args.sweep_id
    eval_config = Path(args.eval_config) if args.eval_config is not None or args.eval_config != "" else None
    main(proj_name, sweep_id, eval_config)
