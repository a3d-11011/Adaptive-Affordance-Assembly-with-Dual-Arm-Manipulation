import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointNet2SemSegSSG import PointNet2SemSegSSG
from models.InterEncoder import InterEncoder


class ActorEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorEncoder, self).__init__()

        self.hidden_dim = 128
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        # self.bn1=nn.BatchNorm1d(self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)
        # self.bn2=nn.BatchNorm1d(output_dim)
        self.mlp3 = nn.Linear(output_dim, output_dim)
        self.get_mu = nn.Linear(output_dim, output_dim)
        self.get_logvar = nn.Linear(output_dim, output_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction="none")
        self.L1Loss = nn.L1Loss(reduction="none")

    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = self.mlp1(net)
        # net = self.bn1(net)
        net = F.leaky_relu(net)
        net = self.mlp2(net)
        # net = self.bn2(net)
        net = F.leaky_relu(net)
        net = self.mlp3(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar


class ActorDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(ActorDecoder, self).__init__()

        self.hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = self.mlp(net)
        return net


class Network(nn.Module):
    def __init__(
        self,
        feat_dim,
        cp_feat_dim,
        dir_feat_dim,
        inter_input_dim=9,
        inter_hidden_dim=128,
        z_dim=128,
        lbd_kl=1.0,
        lbd_dir=1.0,
        use_normals=True,
    ):
        super(Network, self).__init__()

        self.feat_dim = feat_dim
        self.z_dim = z_dim

        self.lbd_kl = lbd_kl
        self.lbd_dir = lbd_dir

        self.pointnet2 = PointNet2SemSegSSG(feat_dim, use_normals=use_normals)
        self.interencoder = InterEncoder(
            hidden_dim=inter_hidden_dim,
            input_dim=inter_input_dim,
            pn_feat_dim=feat_dim,
            use_normals=use_normals,
        )

        self.mlp_dir = nn.Linear(3, dir_feat_dim)
        self.mlp_cp = nn.Linear(
            (6 if use_normals else 3), cp_feat_dim
        )  # contact point position+normal

        self.BCELoss = nn.BCEWithLogitsLoss(reduction="none")
        self.L1Loss = nn.L1Loss(reduction="none")
        self.MSELoss = nn.MSELoss(reduction="none")
        self.COS_SiM = nn.CosineSimilarity()

        self.all_encoder = ActorEncoder(
            input_dim=feat_dim * 2 + cp_feat_dim * 2 + dir_feat_dim + inter_hidden_dim,
            output_dim=z_dim,
        )
        self.decoder = ActorDecoder(
            input_dim=feat_dim * 2 + cp_feat_dim * 2 + inter_hidden_dim + z_dim,
            output_dim=3,
        )

    def KL(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        # ipdb.set_trace()
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        # high star implementation
        # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        loss = torch.mean(loss)
        return loss

    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(
            a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1,
            p=2,
            dim=1,
        )
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        # [bug fixed]
        # pred_Rs = self.bgs(pred_6d.reshape(-1, 3, 2))
        # gt_Rs = self.bgs(gt_6d.reshape(-1, 3, 2))
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta

    def get_dir_loss(self, pred_dir, gt_dir):
        # return self.MSELoss(pred_dir, gt_dir).mean()
        return (1 - self.COS_SiM(pred_dir, gt_dir)).mean()

    def forward(self, pcs, cp1, cp2, dir2, interaction):
        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        whole_feats = self.pointnet2(pcs)

        net1 = whole_feats[:, :, 0]
        net1 = whole_feats[:, :, 0]
        if torch.isnan(net1).any():
            torch.save(pcs, "pcs_nan.pt")
            raise ("net1 has nan")
        net2 = whole_feats[:, :, 1]
        if torch.isnan(net2).any():
            raise ("net2 has nan")

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        dir2_feats = self.mlp_dir(dir2)
        inter_info = self.interencoder(interaction)

        z_all, mu, logvar = self.all_encoder(
            [net1, net2, cp1_feats, cp2_feats, dir2_feats, inter_info]
        )
        recon_dir2 = self.decoder([net1, net2, cp1_feats, cp2_feats, inter_info, z_all])

        return recon_dir2, mu, logvar

    def get_loss(self, pcs, cp1, cp2, dir2, interaction):
        batch_size = pcs.shape[0]
        recon_dir2, mu, logvar = self.forward(pcs, cp1, cp2, dir2, interaction)
        dir_loss = self.get_dir_loss(recon_dir2, dir2)
        kl_loss = self.KL(mu, logvar)
        losses = {}
        losses["kl"] = kl_loss
        losses["dir"] = dir_loss
        losses["tot"] = self.lbd_kl * kl_loss + self.lbd_dir * dir_loss

        return losses, recon_dir2

    def actor_sample(self, pcs, cp1, cp2, interaction):
        batch_size = pcs.shape[0]

        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]
        net2 = whole_feats[:, :, 1]

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        inter_info = self.interencoder(interaction)

        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        recon_dir2 = self.decoder([net1, net2, cp1_feats, cp2_feats, inter_info, z_all])

        recon_dir2 = F.normalize(recon_dir2, p=2, dim=-1)

        return recon_dir2

    def actor_sample_n(self, pcs, cp1, cp2, interaction, rvs=10):
        batch_size = pcs.shape[0]

        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]
        net2 = whole_feats[:, :, 1]

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        # dir1_feats = self.mlp_dir(dir1)
        inter_info = self.interencoder(interaction)

        z_all = torch.Tensor(torch.randn(batch_size, rvs, self.z_dim)).to(net1.device)

        expanded_rvs = z_all.reshape(batch_size * rvs, -1)
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
        recon_dir2 = self.decoder(
            [
                expanded_net1,
                expanded_net2,
                expanded_cp1_feats,
                expanded_cp2_feats,
                expanded_inter_info,
                expanded_rvs,
            ]
        )

        recon_dir2 = F.normalize(recon_dir2, p=2, dim=-1)
        return recon_dir2

    def actor_sample_n_diffCtpts(self, pcs, cp1, cp2, interaction, rvs_ctpt=10, rvs=10):
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

        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        inter_info = self.interencoder(interaction)

        z_all = torch.Tensor(torch.randn(batch_size * rvs_ctpt, rvs, self.z_dim)).to(
            net1.device
        )

        expanded_rvs = z_all.reshape(batch_size * rvs_ctpt * rvs, -1)

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
        recon_dir2 = self.decoder(
            [
                expanded_net1,
                expanded_net2,
                expanded_cp1_feats,
                expanded_cp2_feats,
                expanded_inter_info,
                expanded_rvs,
            ]
        )

        recon_dir2 = F.normalize(recon_dir2, p=2, dim=-1)
        return recon_dir2

    def actor_sample_n_finetune(
        self, pcs, cp1, cp2, interaction, rvs_ctpt=100, rvs=100
    ):
        batch_size = pcs.shape[0]

        cp1_feats = self.mlp_cp(cp1)  # (B , -1)
        cp2_feats = self.mlp_cp(cp2)  # (B * rvs_ctpt, -1)
        inter_info = self.interencoder(interaction)  # (B , -1)

        cp1 = cp1.reshape(batch_size, rvs_ctpt, -1)
        cp2 = cp2.reshape(batch_size, rvs_ctpt, -1)
        pcs[:, 0:1] = cp1
        pcs[:, 1 : 1 + rvs_ctpt] = cp2

        whole_feats = self.pointnet2(pcs)
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
        )

        z_all = torch.Tensor(torch.randn(batch_size * rvs_ctpt, rvs, self.z_dim)).to(
            net1.device
        )

        expanded_rvs = z_all.reshape(batch_size * rvs_ctpt * rvs, -1)
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

        recon_dir2 = self.decoder(
            [
                expanded_net1,
                expanded_net2,
                expanded_cp1_feats,
                expanded_cp2_feats,
                expanded_inter_info,
                expanded_rvs,
            ]
        )
        recon_dir2 = F.normalize(recon_dir2, p=2, dim=-1)

        return recon_dir2[:, :2, :]
