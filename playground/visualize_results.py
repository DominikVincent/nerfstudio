import plotly.graph_objs as go
import os
from collections import defaultdict
import plotly

# title = "KLEVR mIoU of best Models"
# run_values = {
#     "Pointnet++": {"value": 0.752, "group": "group_1"},
#     "Custom": {"value": 0.835, "group": "group_1"},
#     "Stratified": {"value": 0.932, "group": "group_1"},
#     "NeSF": {"value": 0.927, "group": "group_1"},
#     "DeepLab": {"value": 0.971, "group": "group_1"},
# }

# Proximity Loss
# title = "KLEVR Impact of Proximity Loss on mIoU"
# run_values = {
#     "Pointnet++": {"value": 0.752, "group": "proximity loss"},
#     "Pointnet++ ": {"value": 0.655, "group": "no proximity loss"},
#     "Custom": {"value": 0.806, "group": "proximity loss"},
#     "Custom ": {"value": 0.797, "group": "no proximity loss"},
#     "Stratified": {"value": 0.887, "group": "proximity loss"},
#     "Stratified ": {"value": 0.88, "group": "no proximity loss"},
# }

# ground removal
# title = "KLEVR Impact of Ground Removal on mIoU"
# run_values = {
#     "Pointnet++": {"value": 0.752, "group": "ground removal"},
#     "Pointnet++ ": {"value": 0.42, "group": "no ground removal"},
#     "Stratified": {"value": 0.887, "group": "ground removal"},
#     "Stratified ": {"value": 0.813, "group": "no ground removal"},
# }

# Klevr pretrain
title="KLEVR Impact of Pretraining on mIoU"
run_values = {
    "Custom*": {"value": 0.8826, "group": "10 scenes, all views"},
    "pretrained  Custom*": {"value": 0.8826, "group": "10 scenes, all views"}, # yes really the same
    "Custom**": {"value": 0.3789, "group": "3 scenes, all views"},
    "Pretrained rgb 0.75 Custom**": {"value": 0.8826, "group": "3 scenes, all views"}, 
    "Custom***": {"value": 0.333, "group": "3 scenes, 5 views"},
    "Pretrained rgb 0.75 Custom***": {"value": 0.332, "group": "3 scenes, 5 views"}, 
}   

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
for i, (group, values) in enumerate(group_data.items()):
    text = ["{:.3f}".format(y) for y in values["y"]]

    data.append(
        go.Bar(
            x=values["x"],
            y=values["y"],
            name=group,
            text=text,
            textposition="auto",
            textfont=dict(size=124),
            hovertemplate="%{y:.3f}",
            marker=dict(color=colors[i % len(colors)]),
        )
    )

layout = go.Layout(
    title=title,
    xaxis=dict(
        title=x_axis_label, 
        categoryorder='array', 
        categoryarray=list(run_values.keys())  # Preserving the order of runs here
    ),
    yaxis=dict(title=y_axis_label, range=[0, 1]),
    font=dict(family="Arial", size=72),
    margin=dict(l=50, r=50, b=50, t=200, pad=4),
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
fig.write_image(file_path)
fig.write_image(file_path_pdf)
