#!/bin/bash

set -x  # Enable debug mode to print each command


ns-train nerfacto \
	--vis wandb \
	--pipeline.datamanager.camera-optimizer.mode off \
	--save-only-latest-checkpoint True \
	--max-num-iterations 8000 \
	--experiment-name $2 \
	--output-dir $3 \
	--data $1 \
	--timestamp "17_03_23" \
	--viewer.websocket-port 7006  
	# --optimizers.proposal-networks.optimizer.weight-decay 0.0001 \
