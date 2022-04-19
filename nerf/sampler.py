import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase

from render_functions import get_device

# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):

        device = get_device()

        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        pts_step_size = (self.max_depth - self.min_depth) / self.n_pts_per_ray
        z_points = torch.arange(self.min_depth, self.max_depth, pts_step_size)

        N = ray_bundle.origins.size(0)
        d = z_points.size(0)

        # Both min and max inclusive. Resulting shape - (N, d, 1)
        all_z_vals = z_points.view(1, -1, 1).repeat(N, 1, 1).to(device)
        ray_bundle = ray_bundle.to(device)

        # TODO (1.4): Sample points from z values
        # Directions are converted to (N, d, 3) from (N, 3). These are multiplied
        # with all_z_vals (N, d, 1) to get the points. The origins are repeated
        # accordingly and added to get the final points as (N, d, 3).
        sample_points = ray_bundle.directions.view(-1, 1, 3).repeat(1, d, 1) * \
            all_z_vals + ray_bundle.origins.view(-1, 1, 3).repeat(1, d, 1)

        # Note to self - sample_points - (N, d, 3), lengths - (N, d, 1)
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=all_z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}