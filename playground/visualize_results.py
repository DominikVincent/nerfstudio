import plotly.graph_objs as go
import os
from collections import defaultdict
import plotly
import time

# title = "KLEVR mIoU of best Models"
# run_values = {
#     "Pointnet++": {"value": 75.2, "group": "group_1"},
#     "Custom": {"value": 83.5, "group": "group_1"},
#     "Stratified (S3DIS)": {"value": 93.2, "group": "group_1"},
#     "Stratified (S3DIS) + Field Head": {"value": 92.8, "group": "group_1"},
#     "NeSF": {"value": 92.7, "group": "group_1"},
#     "DeepLab": {"value": 97.1, "group": "group_1"},
# }

# run_values = {
#     "Ours: SPT\uFE61": {"value": 93.2, "group": "group_1"},
#     "Ours: SPT\uFE61 + Field Head": {"value": 92.8, "group": "group_1"},
#     "NeSF": {"value": 92.7, "group": "group_1"},
#     "DeepLab": {"value": 97.1, "group": "group_1"},
# }

# run_values = {
#     "Ours: PointNet++": {"value": 75.2, "group": "group_1"},
#     "Ours: Custom": {"value": 83.5, "group": "group_1"},
#     "Ours: SPT\uFE61": {"value": 93.2, "group": "group_1"},
#     "NeSF": {"value": 92.7, "group": "group_1"},
#     "DeepLab": {"value": 97.1, "group": "group_1"},
# }


# Proximity Loss
# title = "KLEVR Impact of Proximity Loss on mIoU"
# run_values = {
#     "Pointnet++": {"value": 75.2, "group": "proximity loss"},
#     "Pointnet++*": {"value": 65.5, "group": "no proximity loss"},
#     "Custom": {"value": 80.6, "group": "proximity loss"},
#     "Custom*": {"value": 79.7, "group": "no proximity loss"},
#     "Stratified": {"value": 88.7, "group": "proximity loss"},
#     "Stratified*": {"value": 88, "group": "no proximity loss"},
# }

# ground removal
# title = "KLEVR Impact of Ground Removal on mIoU"
# run_values = {
#     "Pointnet++": {"value": 75.2, "group": "ground removal"},
#     "Pointnet++*": {"value": 42., "group": "no ground removal"},
#     "Stratified": {"value": 88.7, "group": "ground removal"},
#     "Stratified*": {"value": 81.3, "group": "no ground removal"},
# }

# Klevr no surface sampling 8 samples
# title = "KLEVR Impact of Surface Sampling on mIoU"
# run_values = {
#     "Pointnet++": {"value": 65.5, "group": "surface sampling"}, #no proxmiity loss
#     "Pointnet++*": {"value": 54.04, "group": "no surface sampling"},
#     "Stratified": {"value": 88., "group": "surface sampling"}, # no proximity no pretrained
#     "Stratified*": {"value": 88.73, "group": "no surface sampling"},
# }

# Klevr pretrain
# title="KLEVR Impact of Pretraining on mIoU, 10 scenes"
# run_values = {
#     "scratch": {"value": 54.0, "group": "no pretrain"},
#     "s3dis": {"value": 66.9, "group": "pretrained"},
#     "rgb random p=0.5": {"value": 66.6, "group": "pretrained"},
#     "rgb random p=0.75": {"value": 48.4, "group": "pretrained"},
#     "rgb patch, p=0.5, k=100": {"value": 68.2, "group": "pretrained"},
#     "rgb patch, p=0.5, k=400": {"value": 68.1, "group": "pretrained"},
#     "rgb patch-fp, p=0.5, k=400": {"value": 67.7, "group": "pretrained"},
#     "normals+rgb patch-fp, p=0.5, k=400": {"value": 74.6, "group": "pretrained"},
#     "normals": {"value": 76.1, "group": "pretrained"},
# }   

# Klevr Normal Eval
# title = "KUBASIC-10 normal evaluation"
# run_values = {
#     "analytic normals": {"value": 0.4672, "group": "0"},
#     "predicted normals": {"value": 0.5875, "group": "0"},
# }

# Toybox-5 best models
title = "Toybox-5 mIoU of best Models"
# run_values = {
#     "Pointnet++": {"value": 63.0, "group" :"group_1"}, 
#     "Custom": {"value": 69.94, "group": "group_1"},
#     "Stratified (S3DIS)": {"value": 84.16, "group": "group_1"},
#     "Stratified (S3DIS) + Field Head": {"value": 80.7, "group": "group_1"},
#     "NeSF": {"value": 81.7, "group": "group_1"},
#     "Deep Lab": {"value": 81.6, "group": "group_1"},
# }
run_values = {
    "Ours: SPT\uFE61": {"value": 84.16, "group": "group_1"},
    "Ours: SPT\uFE61 + Field Head": {"value": 80.7, "group": "group_1"},
    "NeSF": {"value": 81.7, "group": "group_1"},
    "Deep Lab": {"value": 81.6, "group": "group_1"},
}

# run_values = {
#     "Ours: PointNet++": {"value": 63.0, "group": "group_1"},
#     "Ours: Custom": {"value": 69.94, "group": "group_1"},
#     "Ours: SPT\uFE61": {"value": 84.16, "group": "group_1"},
#     "NeSF": {"value": 81.7, "group": "group_1"},
#     "Deep Lab": {"value": 81.6, "group": "group_1"},
# }

# Toybox-5
# run_values = {
#     "NeSF": {"value": 0.817, "group": "*"},
#     "Stratified_best": {"value": 0.810, "group": "*"},
#     "Stratified_100_270_pretrain_rgb_patch-0.5-100": {"value": 0.6438, "group": "*"},
#     "Stratified_100_270_pretrain_rgb_patch-0.5-400": {"value": 0.7289, "group": "*"},
#     "Stratified_100_270_pretrain_rgb-random-0.75": {"value": 0.7303, "group": "*"},
#     "Stratified_100_270_pretrain_rgb-random-0.5": {"value": 0.7264, "group": "*"},
#     "Stratified_100_270": {"value": 0.7484, "group": "*"},
#     "Stratified_100_270_s3dis": {"value": 0.7742, "group": "*"},
#     "Stratified_100_10_pretrain_rgb_patch-0.5-100": {"value": 0.602, "group": "*"},
#     "Stratified_100_10_pretrain_rgb_random_0.5": {"value": 0.638, "group": "*"},
#     "Stratified_100_10_pretrain_rgb_patch_0.5-400": {"value": 0.6412, "group": "*"},
#     "Stratified_100_10_pretrain_normals": {"value": 0.642, "group": "*"},
#     "Stratified_100_10_pretrain_rgb_random_0.75": {"value": 0.6815, "group": "*"},
#     "Stratified_100_10": {"value": 0.7526, "group": "*"},
#     "Stratified_100_10_s3dis": {"value": 0.7458, "group": "*"},
# }

# Toybox-5 impact of pretraining on mIoU 100 scenes 270 images
# title = "Toybox5 Impact of Pretraining on mIoU, 100 scenes 270 images"
# run_values = {
#     "scratch": {"value": 75.4, "group": "no pretrain"},
#     "S3DIS": {"value": 76.82, "group": "pretrained"},
#     "rgb random p=0.5": {"value": 65.63, "group": "pretrained"},
#     "rgb random p=0.75": {"value": 66.62, "group": "pretrained"},
#     "rgb patch p=0.5 k=100": {"value": 68.08, "group": "pretrained"}, 
#     "rgb patch p=0.5 k=400": {"value": 64.21, "group": "pretrained"},
#     "rgb patch-fp p=0.5 k=400": {"value": 63.78, "group": "pretrained"},
#     "normals+rgb patch-fp p=0.5 k=400": {"value": 75.11, "group": "pretrained"},
#     "normals": {"value": 75.96, "group": "pretrained"},
#     "density": {"value": 65.21, "group": "pretrained"},
# }

# Toybox5 impact of pretraining on mIoU 100 scenes 10 images
# title = "Toybox5 Impact of Pretraining on mIoU, 100 scenes 10 images"
# run_values = {
#     "scratch": {"value": 74.58, "group": "no pretrain"},
#     "S3DIS pretrained all parameters": {"value": 75.26, "group": "pretrained"},
#     "rgb random p=0.5": {"value": 66.04, "group": "pretrained"},
#     "rgb random p=0.75": {"value": 64.27, "group": "pretrained"},
#     "rgb patch p=0.5 k=100": {"value": 65.61, "group": "pretrained"},
#     "rgb patch p=0.5 k=400": {"value": 64.22, "group": "pretrained"},
#     "rgb patch-fp p=0.5 k=400": {"value": 65.63, "group": "pretrained"},
#     "normals+rgb patch-fp p=0.5 k=400": {"value": (73.96+75.24)/2, "group": "pretrained"},
#     "normals": {"value": 75.29, "group": "pretrained"},
#     "density": {"value": 61.17, "group": "pretrained"},
# }


show_legend = not all(run_data["group"] == next(iter(run_values.values()))["group"] for run_data in run_values.values())


y_axis_label = "mIoU"
x_axis_label = "Models"

group_data = {group: {"x": [], "y": []} for group in set(run_data["group"] for run_data in run_values.values())}

for run_name, run_data in run_values.items():
    group_data[run_data["group"]]["x"].append(run_name.replace("*", ""))
    group_data[run_data["group"]]["y"].append(run_data["value"])

data = []
# colors = ["blue", "red", "green", "orange", "purple", "yellow"]
colors = plotly.colors.qualitative.Plotly
sorted_group_data = sorted(group_data.items(), key=lambda item: item[0], reverse=True)
for i, (group, values) in enumerate(sorted_group_data):
    text = ["{:.1f}".format(y) for y in values["y"]]
    data.append(
        go.Bar(
            x=values["x"],
            y=values["y"],
            name=group,
            text=text,
            textposition="auto",
            textfont=dict(size=124),
            hovertemplate="%{y:.1f}",
            marker=dict(color=colors[i % len(colors)]),
        )
    )

layout = go.Layout(
    title=title,
    xaxis=dict(
        title=x_axis_label, 
        categoryorder='array', 
        categoryarray=[run_value for run_value in run_values.keys() if "*" not in run_value ]  # Preserving the order of runs here
    ),
    yaxis=dict(title=y_axis_label, range=[0, 100]),
    font=dict(family="Arial", size=110),
    margin=dict(l=50, r=50, b=50, t=240, pad=4),
    barmode="group",
    showlegend=show_legend,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

width = 8  # inches
height = 6  # inches
dpi = 600
fig = go.Figure(data=data, layout=layout)
fig.update_layout(width=int(width*dpi), height=int(height*dpi))

folder_path = "/data/vision/polina/projects/wmh/dhollidt/documents/nerf/results"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_name = "{}.png".format(title)
file_name_pdf = "{}.pdf".format(title)
if file_name in os.listdir(folder_path):
    file_name = "{}_{}.png".format(title, len(os.listdir(folder_path)))
    file_name_pdf = "{}_{}.pdf".format(title, len(os.listdir(folder_path)))
file_path = os.path.join(folder_path, file_name)
file_path_pdf = os.path.join(folder_path, file_name_pdf)
time.sleep(1)
fig.write_image(file_path)
time.sleep(1)
fig.write_image(file_path_pdf)
