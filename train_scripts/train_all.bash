#!/bin/bash

set -x  # Enable debug mode to print each command

# Set the path to the directory containing the folders
directory_path="/data/vision/polina/projects/wmh/dhollidt/datasets/klevr_nesf"

# Set ouput directory
output_dir="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/klever_models_nesf"

# check or create output directory
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Set the path to the bash script you want to execute
script_path="/data/vision/polina/projects/wmh/dhollidt/documents/nerf/train_scripts/train.bash"

# Set the number of folders to process
num_folders=300

# Iterate through each folder in the directory
count=0
for folder in $directory_path/*; do
  if [ $count -eq $num_folders ]; then
    break
  fi

  # get the folder name
  folder_name=$(basename "$folder")
  if [ -d "$folder" ] && [ "$(ls -A "$folder")" ]; then
    # Execute the script with the folder name as an argument
    $script_path "$folder" "$folder_name" "$output_dir"
    ((count++))
  fi
done

