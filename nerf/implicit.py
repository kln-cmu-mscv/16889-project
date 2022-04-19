import torch
import torch.nn as nn
import torch.nn.functional as F

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)

class NeuralRadianceField(torch.nn.Module):

    def __init__(self, cfg):
        """
        """
        super(NeuralRadianceField, self).__init__()

        self.D = cfg.n_layers_xyz
        self.W = cfg.n_hidden_neurons_xyz
        self.input_ch = 3
        self.input_ch_views = 3
        self.skips = cfg.append_xyz
        self.use_viewdirs = cfg.use_viewdirs

        self.harmonic_embedding_xyz = HarmonicEmbedding(3,
                                                        cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3,
                                                        cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        pts_linears = [nn.Linear(embedding_dim_xyz, self.W)]

        for i in range(self.D - 1):
            if i not in self.skips:
                pts_linears.append(nn.Linear(self.W, self.W))

            else:
                pts_linears.append(nn.Linear(self.W + embedding_dim_xyz, self.W))

        self.pts_linears = nn.ModuleList(pts_linears)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(embedding_dim_dir + self.W,
                                                      self.W//2)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(self.W, self.W)
            self.alpha_linear = nn.Linear(self.W, 1)
            self.rgb_linear = nn.Linear(self.W//2, 3)
        else:
            self.output_linear = nn.Linear(self.W, 1 + 3)

    def forward(self, ray_bundle):

        # Putting it in words for my stupid brain
        # Without View Dependence
        # xyz -> embed(xyz) -> pts_linears -> output_lin() -> 1 + 3
        # With View Dependence
        # xyz -> embed(xyz) -> pts_linears() -> alpha_linear() -> 1
        #                                    -> feature_lin() + embed(dir) -> rgb_lin() -> 3

        points = ray_bundle.sample_points
        directions = ray_bundle.directions.unsqueeze(1).repeat(1,
                                                               points.size(1),
                                                               1)

        embed_points = self.harmonic_embedding_xyz(points.view(-1, 3))
        embed_dirs = self.harmonic_embedding_dir(directions.view(-1, 3))

        h = embed_points
        for i, _ in enumerate(self.pts_linears):

            h = self.pts_linears[i](h)
            h = F.relu(h)

            if i in self.skips:
                h = torch.cat([embed_points, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, embed_dirs], -1)

            for i, _ in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)

            out = {
                'density': alpha,
                'feature': torch.sigmoid(rgb)
            }

        else:
            outputs = self.output_linear(h)
            out = {
                'density': outputs[..., 0].view(-1, 1),
                'feature': torch.sigmoid(outputs[..., 1:])
            }

        return out

volume_dict = {
    'nerf': NeuralRadianceField,
}