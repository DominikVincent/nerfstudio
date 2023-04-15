from math import ceil, sqrt
from typing import Union

import plotly.graph_objs as go
import torch
from rich.console import Console

import wandb

CONSOLE = Console(width=120)


def filter_points(outs, density_mask, misc: dict):
    Z_FILTER_VALUE = -2.49
    Z_FILTER_VALUE = -1.0
    XY_DISTANCE = 1.0
    points_dense = misc["ray_samples"].frustums.get_positions()[density_mask]
    points_dense_mask = (points_dense[:, 2] > Z_FILTER_VALUE) & (torch.norm(points_dense[:, :2], dim=1) <= XY_DISTANCE)
    points_dense_mask = points_dense_mask.to(density_mask.device)
    outs = outs[:, points_dense_mask, :]

    points = misc["ray_samples"].frustums.get_positions()
    points_mask = (points[:, :, 2] > Z_FILTER_VALUE) & (torch.norm(points[:, :, :2], dim=2) <= XY_DISTANCE).to(
        density_mask.device
    )
    density_mask = torch.logical_and(density_mask, points_mask)
    MAX_COUNT_POINTS = 16384
    if density_mask.sum() > MAX_COUNT_POINTS:
        CONSOLE.print("There are too many points, we are limiting to ", MAX_COUNT_POINTS, " points.")
        # randomly mask points such that we have MAX_COUNT_POINTS points
        outs = outs[:, :MAX_COUNT_POINTS, :]
        true_indices = torch.nonzero(density_mask.flatten()).squeeze()
        true_indices = true_indices[:MAX_COUNT_POINTS]
        density_mask = torch.zeros_like(density_mask)
        density_mask.view(-1, 1)[true_indices] = 1
    return outs, density_mask


def sequential_batching(outs: torch.Tensor, density_mask: torch.Tensor, misc: dict, batch_size: int):
    device = outs.device
    W = outs.shape[1]
    padding = batch_size - (W % batch_size) if W % batch_size != 0 else 0
    outs = torch.nn.functional.pad(outs, (0, 0, 0, padding))
    masking_top = torch.full((W,), False, dtype=torch.bool)
    masking_bottom = torch.full((padding,), True, dtype=torch.bool)
    masking = torch.cat((masking_top, masking_bottom)).to(device)
    ids_shuffle = torch.arange(outs.shape[1])
    ids_restore = ids_shuffle

    points = misc["ray_samples"].frustums.get_positions()[density_mask].view(-1, 3)
    points_pad = torch.nn.functional.pad(points, (0, 0, 0, padding))
    colors_pad = torch.nn.functional.pad(misc["rgb"][density_mask], (0, 0, 0, padding))

    return outs, masking, ids_shuffle, ids_restore, points_pad, colors_pad


def spatial_sliced_batching(outs: torch.Tensor, density_mask: torch.Tensor, misc: dict, batch_size: int, device="cpu"):
    """
    Given the features, density, points and batch size, order the spatially in such a away that each batch is a box on an irregualar grid.
    Each cell contains batch size points.
    In:
     - outs: [1, points, feature_dim]
     - density_mask: [N, used_samples]
     - misc["ray_samples]: [N, used_samples]

    Out:
     - outs: [1, points + padding_of_batch, feature_dim]
     - masking: [points + padding_of_batch]
     - ids_shuffle: [points + padding_of_batch]
     - ids_restore: [points + padding_of_batch]
     - points_pad: [points + padding_of_batch, 3]
     - colors_pad: [points + padding_of_batch, 3]
    """
    device = outs.device
    W = outs.shape[1]

    columns_per_row = max(int(sqrt(W / batch_size)), 1)
    row_size = columns_per_row * batch_size
    row_count = ceil(W / row_size)

    print(
        "Row count",
        row_count,
        "columns per row",
        columns_per_row,
        "Grid size",
        row_count * columns_per_row,
        "Takes points: ",
        row_count * columns_per_row * batch_size,
        "At point count: ",
        W,
    )

    padding = (
        (columns_per_row * batch_size) - (W % (columns_per_row * batch_size))
        if W % (columns_per_row * batch_size) != 0
        else 0
    )

    outs = torch.nn.functional.pad(outs, (0, 0, 0, padding))
    masking_top = torch.full((W,), False, dtype=torch.bool)
    masking_bottom = torch.full((padding,), True, dtype=torch.bool)
    masking = torch.cat((masking_top, masking_bottom)).to(device)

    points = misc["ray_samples"].frustums.get_positions()[density_mask].view(-1, 3)
    rand_points = torch.randn((padding, 3), device=points.device)
    points_padded = torch.cat((points, rand_points))
    ids_shuffle = torch.argsort(points_padded[:, 0])

    for i in range(row_count):
        # for each row seperate into smaller cells by y axis
        row_idx = ids_shuffle[i * row_size : (i + 1) * row_size]
        points_row_y = points_padded[row_idx, 1]
        ids_shuffle[i * row_size : (i + 1) * row_size] = row_idx[torch.argsort(points_row_y)]

    ids_restore = torch.argsort(ids_shuffle)

    colors_pad = torch.nn.functional.pad(misc["rgb"][density_mask], (0, 0, 0, padding))

    return outs, masking, ids_shuffle, ids_restore, points_padded, colors_pad


def visualize_point_batch(points_pad: torch.Tensor, ids_shuffle: torch.Tensor, batch_size: int):
    """ """
    points_pad_reordered = points_pad[ids_shuffle, :].to("cpu")
    points_pad_reordered = points_pad_reordered.reshape(-1, batch_size, 3)
    points = torch.empty((0, 3))
    colors = torch.empty((0, 3))
    for i in range(points_pad_reordered.shape[0]):
        points = torch.cat((points, points_pad_reordered[i, :, :]))
        colors = torch.cat((colors, torch.randn((1, 3)).repeat(batch_size, 1) * 255))
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(color=colors, size=1, opacity=0.8),
    )

    # create a layout with axes labels
    layout = go.Layout(
        title=f"RGB points: {points.shape}",
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
    )
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def visualize_points(points: torch.Tensor):
    """ """
    points = points.squeeze(0).to("cpu").numpy()
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(color="red", size=1, opacity=0.8),
    )

    # create a layout with axes labels
    layout = go.Layout(
        title=f"RGB points: {points.shape}",
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z")),
    )
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def log_points_to_wandb(points_pad: torch.Tensor, ids_shuffle: Union[None, torch.Tensor], batch_size: int):
    if wandb.run is None:
        CONSOLE.print("[yellow] Wandb run is None but tried to log points. Skipping!")
        return
    if ids_shuffle is not None:
        points_pad_reordered = points_pad[ids_shuffle, :].to("cpu")
        points_pad_reordered = points_pad_reordered.reshape(-1, batch_size, 3)
    else:
        points_pad_reordered = points_pad.to("cpu")
    points = torch.empty((0, 3))
    colors = torch.empty((0, 3))
    for i in range(points_pad_reordered.shape[0]):
        points = torch.cat((points, points_pad_reordered[i, :, :]))
        colors = torch.cat((colors, torch.randn((1, 3)).repeat(batch_size, 1) * 255))
    point_cloud = torch.cat((points, colors), dim=1).detach().cpu().numpy()
    wandb.log({"point_samples": wandb.Object3D(point_cloud)}, step=wandb.run.step)
