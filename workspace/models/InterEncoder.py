import torch
import torch.nn as nn

from models.PointNet2SemSegSSG import PointNet2SemSegSSG


class InterEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=2,
        input_dim=6,
        pn_feat_dim=128,
        hidden_feat_dim=128,
        use_normals=True,
    ):
        super(InterEncoder, self).__init__()
        self.use_normals = use_normals
        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )
        self.mlp_m = nn.Sequential(
            nn.Linear(12, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_feat_dim * 2 + pn_feat_dim, hidden_feat_dim),
            nn.ReLU(),
        )
        self.hidden_info_encoder = nn.Linear(hidden_feat_dim, hidden_dim)

        self.pointnet2 = PointNet2SemSegSSG(pn_feat_dim, use_normals=use_normals)
        self.AttentionNet = nn.Sequential(
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, interact):
        cp2, dir2, pcs, move = interact
        pcs = jitter_zero_pcs(pcs)
        batch_size = cp2.shape[0]
        horizon_set = cp2.shape[1]
        cp2 = cp2.view(batch_size * horizon_set, -1)
        dir2 = dir2.view(batch_size * horizon_set, -1)
        move = move.view(batch_size * horizon_set, -1)
        pcs = pcs.view(batch_size * horizon_set, -1, 6 if self.use_normals else 3)

        whole_feats = self.pointnet2(pcs)
        pc_feat = whole_feats[:, :, 0]

        # pc_feat=pc_feat.view(batch_size,horizon_set,-1)

        x = torch.cat([cp2, dir2], dim=1)
        hidden_feat = self.mlp1(x)
        hidden_m = self.mlp_m(move)
        hidden_feat = torch.cat([hidden_feat, hidden_m, pc_feat], dim=1)
        hidden_feat = self.mlp2(hidden_feat)
        hidden_info = self.hidden_info_encoder(hidden_feat)
        hidden_info_attention = self.AttentionNet(hidden_info)

        hidden_info = hidden_info * hidden_info_attention
        hidden_info = hidden_info.view(batch_size, horizon_set, -1).sum(dim=1)
        hidden_info_attention = hidden_info_attention.view(
            batch_size, horizon_set, -1
        ).sum(dim=1)

        eps = 1e-8
        hidden_info_attention = hidden_info_attention.clamp(min=eps)
        mean_hidden_info = hidden_info / hidden_info_attention

        return mean_hidden_info


def jitter_zero_pcs(pcs, sigma=1e-4):
    """
    pcs: (B,horizon,npoints,6 or 3)
    """
    B = pcs.shape[0]
    is_all_zero = pcs.abs().sum(dim=(1, 2, 3)) == 0
    if is_all_zero.any():
        noise = torch.randn_like(pcs) * sigma
        pcs[is_all_zero] += noise[is_all_zero]
    return pcs
