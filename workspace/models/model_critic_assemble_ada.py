import torch.nn as nn
import torch
import torch.nn.functional as F

from models.PointNet2SemSegSSG import PointNet2SemSegSSG
from models.InterEncoder import InterEncoder


class Critic(nn.Module):

    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(Critic, self).__init__()

        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, inputs):
        input_net = torch.cat(inputs, dim=-1)
        hidden_net = F.leaky_relu(self.mlp1(input_net))
        net = self.mlp2(hidden_net)
        return net


class Network(nn.Module):
    def __init__(
        self,
        feat_dim,
        cp_feat_dim,
        dir_feat_dim,
        inter_input_dim=9,
        inter_hidden_dim=128,
        hidden_feat_dim=128,
        use_normals=True,
    ):
        super(Network, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG(feat_dim, use_normals=use_normals)

        self.interencoder = InterEncoder(
            hidden_dim=inter_hidden_dim,
            input_dim=inter_input_dim,
            pn_feat_dim=feat_dim,
            hidden_feat_dim=hidden_feat_dim,
            use_normals=use_normals,
        )

        self.critic = Critic(
            input_dim=feat_dim * 2 + cp_feat_dim * 2 + dir_feat_dim + inter_hidden_dim,
            hidden_dim=hidden_feat_dim,
        )

        self.mlp_dir = nn.Linear(3, dir_feat_dim)
        self.mlp_cp = nn.Linear(6 if use_normals else 3, cp_feat_dim)  # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction="none")
        self.sigmoid = nn.Sigmoid()
        self.BCELoss_withoutSigmoid = nn.BCELoss(reduction="none")
        self.L1Loss = nn.L1Loss(reduction="none")
        self.MSELoss = nn.MSELoss(reduction="none")

    def forward(self, pcs, cp1, cp2, dir2, interaction, idx=None):
        pcs[:, 0, :] = cp1
        pcs[:, 1, :] = cp2
        whole_feats = self.pointnet2(pcs)

        # feature for contact point
        net1 = whole_feats[:, :, 0]
        if torch.isnan(net1).any():
            torch.save(pcs, "pcs_nan.pt")
            print(idx)
            raise ("net1 has nan")
        net2 = whole_feats[:, :, 1]
        if torch.isnan(net2).any():
            raise ("net2 has nan")

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        # dir1_feats = self.mlp_dir(dir1)
        dir2_feats = self.mlp_dir(dir2)
        inter_info = self.interencoder(interaction)

        pred_result_logits = self.critic(
            [net1, net2, cp1_feats, cp2_feats, dir2_feats, inter_info]
        )
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores

    def forward_n(self, pcs, cp1, cp2, dir2, interaction, rvs=100):
        batch_size = pcs.shape[0]
        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]
        net2 = whole_feats[:, :, 1]

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        # dir1_feats = self.mlp_dir(dir1)     # B * 3
        dir2_feats = self.mlp_dir(dir2)  # (B * rvs) * 3

        inter_info = self.interencoder(interaction)

        expanded_net1 = (
            net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        )
        expanded_net2 = (
            net2.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        )
        expanded_cp1_feats = (
            cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        )
        expanded_cp2_feats = (
            cp2_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        )
        # expanded_dir1_feats = dir1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_inter_info = (
            inter_info.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        )
        pred_result_logits = self.critic(
            [
                expanded_net1,
                expanded_net2,
                expanded_cp1_feats,
                expanded_cp2_feats,
                dir2_feats,
                expanded_inter_info,
            ]
        )
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores

    def forward_n_finetune(self, pcs, cp1, cp2, dir2, interaction, rvs_ctpt=10, rvs=10):
        batch_size = pcs.shape[0]

        cp1_feats = self.mlp_cp(cp1)  #  B * dim
        cp2_feats = self.mlp_cp(cp2)  # (B * rvs_ctpt) * dim

        dir2_feats = self.mlp_dir(dir2)  # (B * rvs_ctpt * rvs) * dim

        inter_info = self.interencoder(interaction)

        cp2 = cp2.reshape(batch_size, rvs_ctpt, -1)
        pcs[:, 0] = cp1  # B * N * C
        pcs[:, 1 : 1 + rvs_ctpt] = cp2
        whole_feats = self.pointnet2(pcs)  # B * feats_dim * num_pts
        net1 = (
            whole_feats[:, :, 0:1]
            .permute(0, 2, 1)
            .repeat(1, rvs_ctpt, 1)
            .reshape(batch_size * rvs_ctpt, -1)
        )  # (B * rvs_ctpt) * dim
        net2 = (
            whole_feats[:, :, 1 : 1 + rvs_ctpt]
            .permute(0, 2, 1)
            .reshape(batch_size * rvs_ctpt, -1)
        )  # (B * rvs_ctpt) * dim

        expanded_net1 = (
            net1.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_net2 = (
            net2.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_cp1_feats = (
            cp1_feats.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_cp2_feats = (
            cp2_feats.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_inter_info = (
            inter_info.unsqueeze(dim=1)
            .repeat(1, rvs_ctpt * rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )

        pred_result_logits = self.critic(
            [
                expanded_net1,
                expanded_net2,
                expanded_cp1_feats,
                expanded_cp2_feats,
                dir2_feats,
                expanded_inter_info,
            ]
        )  # (B * rvs_ctpt * rvs) * 1
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores

    def forward_n_diffCtpts(
        self, pcs, cp1, cp2, dir2, interaction, rvs_ctpt=10, rvs=10
    ):  # topk = num_ctpt2
        batch_size = pcs.shape[0]
        channel_size = pcs.shape[-1]
        pcs[:, 0] = cp1  # B * N * C
        pcs = (
            pcs.unsqueeze(dim=1)
            .repeat(1, 1, rvs_ctpt, 1)
            .reshape(batch_size * rvs_ctpt, -1, channel_size)
        )  # (B * rvs_ctpt) * N * C
        pcs[:, 1] = cp2  # (B * rvs_ctpt) * N * C
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]  # (B * rvs_ctpt) * dim
        net2 = whole_feats[:, :, 1]

        cp1_feats = self.mlp_cp(cp1)  # B * dim
        cp2_feats = self.mlp_cp(cp2)  # (B * rvs_ctpt) * dim
        dir2_feats = self.mlp_dir(dir2)  # (B * rvs_ctpt * rvs) * dim
        inter_info = self.interencoder(interaction)

        expanded_net1 = (
            net1.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_net2 = (
            net2.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_cp1_feats = (
            cp1_feats.unsqueeze(dim=1)
            .repeat(1, rvs_ctpt * rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_cp2_feats = (
            cp2_feats.unsqueeze(dim=1)
            .repeat(1, rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )
        expanded_inter_info = (
            inter_info.unsqueeze(dim=1)
            .repeat(1, rvs_ctpt * rvs, 1)
            .reshape(batch_size * rvs_ctpt * rvs, -1)
        )

        pred_result_logits = self.critic(
            [
                expanded_net1,
                expanded_net2,
                expanded_cp1_feats,
                expanded_cp2_feats,
                dir2_feats,
                expanded_inter_info,
            ]
        )  # (B * rvs_ctpt * rvs) * 1
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores

    def get_ce_loss_total(self, pred_logits, gt_labels):
        loss = self.BCELoss_withoutSigmoid(pred_logits, gt_labels.float())
        return loss

    def get_mse_loss_total(self, pred_scores, gt_scores):
        loss = self.MSELoss(pred_scores, gt_scores)
        return loss
