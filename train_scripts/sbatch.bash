#!/bin/bash

sbatch -p QRTX5000 --gres=gpu:1 -t 96:60:00 --mem-per-cpu 2000 -o logs/$(date +"%Y_%m_%d_%I_%M_%p").out -e logs/$(date +"%Y_%m_%d_%I_%M_%p").err $1