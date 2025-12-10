import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2SemSegSSG import PointNet2SemSegSSG
from models.InterEncoder import InterEncoder


class ActionScore(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(ActionScore, self).__init__()

        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    # feats B x F
    # output: B
    def forward(self, inputs):
        feats = torch.cat(inputs, dim=-1)
        net = F.leaky_relu(self.mlp1(feats))
        net = self.mlp2(net)
        return net


class Network(nn.Module):
    def __init__(
        self,
        feat_dim,
        cp_feat_dim,
        inter_input=9,
        inter_hidden_dim=128,
        hidden_feat_dim=128,
        use_normals=True,
    ):
        super(Network, self).__init__()

        # self.topk = topk
        self.use_normals = use_normals
        self.pointnet2 = PointNet2SemSegSSG(feat_dim, use_normals=use_normals)
        self.interencoder = InterEncoder(
            hidden_dim=inter_hidden_dim,
            input_dim=inter_input,
            pn_feat_dim=feat_dim,
            hidden_feat_dim=hidden_feat_dim,
            use_normals=use_normals,
        )

        # self.mlp_dir = nn.Linear(3 , dir_feat_dim)
        self.mlp_cp = nn.Linear(6 if use_normals else 3, cp_feat_dim)  # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction="none")
        self.L1Loss = nn.L1Loss(reduction="none")

        self.action_score = ActionScore(
            input_dim=feat_dim * 2 + cp_feat_dim * 2 + inter_hidden_dim,
            hidden_dim=hidden_feat_dim,
        )

    def forward(self, pcs, cp1, cp2, interaction):
        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]
        net2 = whole_feats[:, :, 1]

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        # dir1_feats = self.mlp_dir(dir1)
        inter_info = self.interencoder(interaction)

        pred_result_logits = self.action_score(
            [net1, net2, cp1_feats, cp2_feats, inter_info]
        )
        pred_score = torch.sigmoid(pred_result_logits)
        return pred_score

    def get_loss(self, pred_score, gt_score):
        loss = self.L1Loss(pred_score, gt_score).mean()
        return loss

    def inference_whole_pc(self, pcs, cp1, interaction):
        batch_size = pcs.shape[0]
        num_pts = pcs.shape[1]

        pcs[:, 0] = cp1

        cp2 = pcs.view(batch_size * num_pts, -1)
        cp2_feats = self.mlp_cp(cp2)

        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]  # B * dim
        net2 = whole_feats.permute(0, 2, 1).reshape(
            batch_size * num_pts, -1
        )  # (B * num_pts) * dim

        cp1_feats = self.mlp_cp(cp1)
        # dir1_feats = self.mlp_dir(dir1)
        inter_info = self.interencoder(interaction)

        expanded_net1 = (
            net1.unsqueeze(dim=1)
            .repeat(1, num_pts, 1)
            .reshape(batch_size * num_pts, -1)
        )
        expanded_cp1_feats = (
            cp1_feats.unsqueeze(dim=1)
            .repeat(1, num_pts, 1)
            .reshape(batch_size * num_pts, -1)
        )
        # expanded_dir1_feats = dir1_feats.unsqueeze(dim=1).repeat(1, num_pts, 1).reshape(batch_size * num_pts, -1)
        expanded_inter_info = (
            inter_info.unsqueeze(dim=1)
            .repeat(1, num_pts, 1)
            .reshape(batch_size * num_pts, -1)
        )

        pred_result_logits = self.action_score(
            [expanded_net1, net2, expanded_cp1_feats, cp2_feats, expanded_inter_info]
        )
        pred_score = torch.sigmoid(pred_result_logits).reshape(batch_size, num_pts)
        return pred_score
