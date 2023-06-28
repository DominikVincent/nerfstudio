import plotly.graph_objs as go
import os
from collections import defaultdict


# run_values = {
#     "Pointnet++": {"value": 0.752, "group": "group_1"},
#     "Custom": {"value": 0.835, "group": "group_1"},
#     "Stratified": {"value": 0.932, "group": "group_1"},
#     "NeSF": {"value": 0.927, "group": "group_1"},
#     "DeepLab": {"value": 0.971, "group": "group_1"},
# }

# Proximity Loss
# run_values = {
#     "Pointnet++": {"value": 0.752, "group": "proximity loss"},
#     "Pointnet++ ": {"value": 0.655, "group": "no proximity loss"},
#     "Custom": {"value": 0.806, "group": "proximity loss"},
#     "Custom ": {"value": 0.797, "group": "no proximity loss"},
#     "Stratified": {"value": 0.887, "group": "proximity loss"},
#     "Stratified ": {"value": 0.88, "group": "no proximity loss"},
# }

# ground removal
run_values = {
    "Pointnet++": {"value": 0.752, "group": "ground removal"},
    "Pointnet++ ": {"value": 0.42, "group": "no ground removal"},
    "Stratified": {"value": 0.887, "group": "ground removal"},
    "Stratified ": {"value": 0.813, "group": "no ground removal"},
}

show_legend = not all(run_data["group"] == next(iter(run_values.values()))["group"] for run_data in run_values.values())

# title = "mIoU of best Models"
# title = "Impact of Proximity Loss on mIoU"
title = "Impact of Ground Removal on mIoU"

y_axis_label = "mIoU"
x_axis_label = "Models"

group_data = {group: {"x": [], "y": []} for group in set(run_data["group"] for run_data in run_values.values())}

for run_name, run_data in run_values.items():
    group_data[run_data["group"]]["x"].append(run_name)
    group_data[run_data["group"]]["y"].append(run_data["value"])

data = []
colors = ["blue", "red", "green", "orange", "purple", "yellow"]
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
