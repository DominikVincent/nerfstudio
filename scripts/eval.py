#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from rich.console import Console

from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    save_images: bool = False
    use_wandb: bool = False

    def main(self) -> None:
        """Main function."""
        writer.setup_event_writer(self.use_wandb, False, log_dir="logs")

        writer.setup_local_writer(
            LoggingConfig(local_writer=LocalWriterConfig(enable=False)), max_iter=100000, banner_messages=["HERRO"]
        )

        config, pipeline, checkpoint_path = eval_setup(self.load_config)
        writer.put_config("config", dataclasses.asdict(config), step=0)
        assert self.output_path.suffix == ".json"
        if self.save_images:
            metrics_dict = pipeline.get_average_eval_image_metrics(save_path=self.output_path.parent)
        else:
            metrics_dict = pipeline.get_average_eval_image_metrics()

        # log the final results
        print(metrics_dict)
        writer.put_dict("test", metrics_dict, step=1000000)
        writer.write_out_storage()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
