#!/bin/bash

sbatch -p A6000 --gres=gpu:1 -t 96:60:00 --mem=80G -o logs/$(date +"%Y_%m_%d_%H_%M_%p").out -e logs/$(date +"%Y_%m_%d_%H_%M_%p").err $1
