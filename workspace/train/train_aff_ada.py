import os
import time
import sys
import yaml
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm, trange

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from datasets.dataset_ada import AssembleDataset
from models.model_aff_assemble_ada import Network as AffordanceNetwork
from models.model_actor_assemble_ada import Network as ActorNetwork
from models.model_critic_assemble_ada import Network as CriticNetwork
from utils.wandb import log_init, log_writer
from utils.train_utils import optimizer_to_device, get_unique_dirname
import argparse


def train(task, config, data_dir="data", base_dir="logs"):

    val_every_epoch = config["val_every_epoch"] if "val_every_epoch" in config else 1
    save_every_epoch = (
        config["save_every_epoch"] if "save_every_epoch" in config else 10
    )

    save_dir_base = f"{base_dir}/{task}/aff"
    save_dir = get_unique_dirname(save_dir_base)

    # actor_path= os.path.join(base_dir,task,'actor','ckpts',f"{config['actor_eval_epoch']}-network.pth")
    # critic_path = os.path.join(base_dir,task,'critic','ckpts',f"{config['critic_eval_epoch']}-network.pth")
    
    actor_path = config["actor_path"]
    critic_path = config["critic_path"]

    train_num = config["train_num"]
    val_num = config["val_num"]

    feat_dim = config["feat_dim"]
    cp_feat_dim = config["cp_feat_dim"]
    dir_feat_dim = config["dir_feat_dim"]
    z_dim = config["z_dim"]
    lbd_kl = config["lbd_kl"]
    lbd_dir = config["lbd_dir"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    lr_decay_every = config["lr_decay_every"]
    lr_decay_by = config["lr_decay_by"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    if torch.cuda.device_count() <= 1:
        device = [
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ]
    else:
        device = [torch.device("cuda:0"), torch.device("cuda:1")]

    seed = random.randint(1, 10000) if "seed" not in config else config["seed"]
    config["seed"] = seed

    use_normals = config["use_normals"] if "use_normals" in config else True

    log_init(run_name=task + "_aff", cfg=config, mode="online")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Creating network ...... ")
    network = AffordanceNetwork(
        feat_dim=feat_dim,
        cp_feat_dim=cp_feat_dim,
        inter_input=9 if use_normals else 6,
        use_normals=use_normals,
    )
    network_opt = torch.optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        network_opt, step_size=lr_decay_every, gamma=lr_decay_by
    )

    # Continue_to_play
    #     network.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % conf.saved_epoch)))
    #     network_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % conf.saved_epoch)))
    #     network_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % conf.saved_epoch)))

    if "finetune" in config.keys() and config["finetune"]:
        print("Finetuning ...... ")
        network.load_state_dict(torch.load(config["finetune"], weights_only=True))
        finetune_dir_base = f"{save_dir_base}_finetune"
        save_dir = get_unique_dirname(finetune_dir_base)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    network.to(device[0])
    optimizer_to_device(network_opt, device[0])

    print("Loading actor and critic ...... ")
    actor = ActorNetwork(
        feat_dim,
        cp_feat_dim,
        dir_feat_dim,
        inter_input_dim=9 if use_normals else 6,
        z_dim=z_dim,
        lbd_kl=lbd_kl,
        lbd_dir=lbd_dir,
        use_normals=use_normals,
    )
    actor.load_state_dict(
        torch.load(
            actor_path,
            weights_only=True,
        )
    )
    actor.to(device[1])
    actor.eval()

    critic = CriticNetwork(
        feat_dim,
        cp_feat_dim,
        dir_feat_dim,
        inter_input_dim=9 if use_normals else 6,
        use_normals=use_normals,
    )
    critic.load_state_dict(
        torch.load(
            critic_path,
            weights_only=True,
        )
    )
    critic.to(device[1])
    critic.eval()

    # Load dataset
    print("Loading dataset ...... ")
    train_dataset = AssembleDataset(
        os.path.join(data_dir, task),
        "train",
        train_num=train_num,
        val_num=val_num,
        use_normals=use_normals,
    )
    val_dataset = AssembleDataset(
        os.path.join(data_dir, task),
        "val",
        train_num=train_num,
        val_num=val_num,
        use_normals=use_normals,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    train_num_batch = len(train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )
    val_num_batch = len(val_dataloader)

    print(f"train_num_batch: {train_num_batch},val_num_batch: {val_num_batch}")

    # start training
    start_epoch = 1
    print("Start training ...... ")
    for epoch in tqdm(
        range(start_epoch, num_epochs + 1),
        desc="Epoch",
        initial=start_epoch,
        total=num_epochs,
        leave=True,
    ):
        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        # ep_loss, ep_cnt = 0, 0
        train_ep_loss, train_cnt = 0, 0

        total_all_acc, positive_acc, negative_acc, fail_acc = 0, 0, 0, 0
        percent_positive, percent_negative, percent_fail = 0, 0, 0

        for train_batch_ind, batch in tqdm(
            train_batches, desc="Batch", total=train_num_batch, leave=False
        ):
            network.train()

            total_loss, all_acc, positive, negative, fail = aff_forward(
                batch=batch,
                network=network,
                actor=actor,
                critic=critic,
                device=device,
                is_val=False,
            )

            total_all_acc += all_acc
            positive_acc += positive[0]
            percent_positive += positive[1]
            negative_acc += negative[0]
            percent_negative += negative[1]
            fail_acc += fail[0]
            percent_fail += fail[1]

            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            train_ep_loss += total_loss
            train_cnt += 1

        content = {
            "loss": train_ep_loss / train_cnt,
            "lr": network_opt.param_groups[0]["lr"],
            "all_acc": total_all_acc / train_cnt,
            "positive_acc": positive_acc / train_cnt,
            "negative_acc": negative_acc / train_cnt,
            "fail_acc": fail_acc / train_cnt,
            "percent_positive": percent_positive / train_cnt,
            "percent_negative": percent_negative / train_cnt,
            "percent_fail": percent_fail / train_cnt,
        }
        log_writer(epoch, content, is_val=False)

        if epoch % val_every_epoch == 0:
            val_ep_loss, val_cnt = 0, 0
            total_all_acc, positive_acc, negative_acc, fail_acc = 0, 0, 0, 0
            percent_positive, percent_negative, percent_fail = 0, 0, 0
            for val_batch_ind, batch in tqdm(
                val_batches, desc="Val", total=val_num_batch, leave=False
            ):
                network.eval()
                total_loss, all_acc, positive, negative, fail = aff_forward(
                    batch=batch,
                    network=network,
                    actor=actor,
                    critic=critic,
                    device=device,
                    is_val=True,
                )
                total_all_acc += all_acc
                positive_acc += positive[0]
                percent_positive += positive[1]
                negative_acc += negative[0]
                percent_negative += negative[1]
                fail_acc += fail[0]
                percent_fail += fail[1]
                val_ep_loss += total_loss
                val_cnt += 1
            content = {
                "loss": val_ep_loss / val_cnt,
                "all_acc": total_all_acc / val_cnt,
                "positive_acc": positive_acc / val_cnt,
                "negative_acc": negative_acc / val_cnt,
                "fail_acc": fail_acc / val_cnt,
                "percent_positive": percent_positive / val_cnt,
                "percent_negative": percent_negative / val_cnt,
                "percent_fail": percent_fail / val_cnt,
            }
            log_writer(epoch, content, is_val=True)

        if (
            epoch % save_every_epoch == 0
            or epoch == num_epochs
            or network_opt.param_groups[0]["lr"] < 5e-7
        ):
            with torch.no_grad():
                tqdm.write(f"Saving checkpoint {epoch}...... ")
                if not os.path.exists(os.path.join(save_dir, "ckpts")):
                    os.makedirs(os.path.join(save_dir, "ckpts"))
                torch.save(
                    network.state_dict(),
                    os.path.join(save_dir, "ckpts", "%d-network.pth" % epoch),
                )
                torch.save(
                    network_opt.state_dict(),
                    os.path.join(save_dir, "ckpts", "%d-optimizer.pth" % epoch),
                )
                torch.save(
                    network_lr_scheduler.state_dict(),
                    os.path.join(save_dir, "ckpts", "%d-lr_scheduler.pth" % epoch),
                )
                tqdm.write("DONE")
            if network_opt.param_groups[0]["lr"] < 5e-7:
                tqdm.write(
                    "Epoch %d : Learning rate is too small, stop training" % epoch
                )
                torch.save(
                    network.state_dict(),
                    os.path.join(save_dir, "ckpts", "%d-network.pth" % epoch),
                )
                torch.save(
                    network_opt.state_dict(),
                    os.path.join(save_dir, "ckpts", "%d-optimizer.pth" % epoch),
                )
                torch.save(
                    network_lr_scheduler.state_dict(),
                    os.path.join(save_dir, "ckpts", "%d-lr_scheduler.pth" % epoch),
                )
                tqdm.write("DONE")
                break


def get_acc(reward, pred_score):
    device = reward.device
    reward_succ = reward > 0.9
    pred_score_succ = pred_score > 0.9
    all_acc = (pred_score_succ == reward_succ).float().mean()

    # Positive accuracy
    positive_mask = reward > 0.9
    positive_pred = pred_score_succ[positive_mask]
    positive_acc = (
        positive_pred.float().mean()
        if positive_pred.numel() > 0
        else torch.tensor(0.0).to(device)
    )
    percent_positive = positive_mask.sum() / len(reward)

    # Negative accuracy
    negative_mask = (reward <= 0.9) & (reward > 0.5)
    pred_score_neg = (pred_score <= 0.9) & (pred_score > 0.5)
    negative_pred = pred_score_neg[negative_mask]
    negative_acc = (
        negative_pred.float().mean()
        if negative_pred.numel() > 0
        else torch.tensor(0.0).to(device)
    )
    percent_negative = negative_mask.sum() / len(reward)

    # Fail accuracy
    fail_mask = reward < 0.5
    pred_score_fail = pred_score < 0.5
    fail_pred = pred_score_fail[fail_mask]
    fail_acc = (
        fail_pred.float().mean()
        if fail_pred.numel() > 0
        else torch.tensor(0.0).to(device)
    )
    percent_fail = fail_mask.sum() / len(reward)

    return (
        all_acc,
        (positive_acc, percent_positive),
        (negative_acc, percent_negative),
        (fail_acc, percent_fail),
    )


def aff_forward(
    batch, network, actor, critic, device, rv_cnt=100, topk=100, is_val=False
):
    points, cp1, cp2, dir2, interaction, reward = batch

    batch_size = points.shape[0]
    pcs = points.to(device[0])
    cp1 = cp1.to(device[0])
    cp2 = cp2.to(device[0])
    a, b, c, d = interaction
    interaction = (a.to(device[0]), b.to(device[0]), c.to(device[0]), d.to(device[0]))

    pred_score = network.forward(pcs, cp1, cp2, interaction)
    with torch.no_grad():
        batch_size = points.shape[0]
        pcs = points.to(device[1])
        cp1 = cp1.to(device[1])

        cp2 = cp2.to(device[1])
        a, b, c, d = interaction
        interaction = (
            a.to(device[1]),
            b.to(device[1]),
            c.to(device[1]),
            d.to(device[1]),
        )
        recon_dir2 = (
            actor.actor_sample_n(pcs, cp1, cp2, interaction, rvs=rv_cnt)
            .contiguous()
            .view(batch_size * rv_cnt, -1)
        )

        gt_scores = critic.forward_n(pcs, cp1, cp2, recon_dir2, interaction, rvs=rv_cnt)
        gt_score = (
            gt_scores.view(batch_size, rv_cnt, 1).topk(k=topk, dim=1)[0].mean(dim=1)
        )
    gt_score = gt_score.to(device[0])
    loss = network.get_loss(
        pred_score.view(batch_size, -1), gt_score.view(batch_size, -1)
    )
    all_acc, positive, negative, fail = get_acc(gt_score, pred_score)
    return loss, all_acc, positive, negative, fail


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="data", help="data directory")
    args.add_argument("--save_dir", type=str, default="logs", help="directory to save logs")
    args.add_argument("--task", type=str, default="drawer", help="task name")
    args.add_argument("--actor", type=int, help="actor eval epoch")
    args.add_argument("--critic", type=int, help="critic eval epoch")
    args.add_argument(
        "--train_num", type=int, default=128, help="number of training samples"
    )
    args.add_argument(
        "--val_num", type=int, default=128, help="number of validation samples"
    )
    args.add_argument(
        "--use_normals",
        action="store_true",
        default=True,
        help="whether to use normals in point cloud",
    )
    args.add_argument(
        "--finetune", type=str, default=None, help="path to finetune model"
    )
    args = args.parse_args()

    task = args.task

    finetune = args.finetune

    val_every_epoch = 10000
    save_every_epoch = 1

    # actor_eval_epoch = 180
    # critic_eval_epoch = 30
    actor_path = args.actor
    critic_path = args.critic

    train_num = args.train_num
    val_num = args.val_num

    feat_dim = 128
    cp_feat_dim = 32
    dir_feat_dim = 32
    z_dim = 128
    lbd_kl = 1.0
    lbd_dir = 1.0
    lr = 0.001
    weight_decay = 1e-5
    lr_decay_every = 500
    lr_decay_by = 0.9
    batch_size = 64 if finetune is None else 32
    num_epochs = 1000 if finetune is None else 30
    use_normals = args.use_normals

    config = {
        "val_every_epoch": val_every_epoch,
        "save_every_epoch": save_every_epoch,
        # "actor_eval_epoch": actor_eval_epoch,
        # "critic_eval_epoch": critic_eval_epoch,
        "actor_path": actor_path,
        "critic_path": critic_path,
        "train_num": train_num,
        "val_num": val_num,
        "feat_dim": feat_dim,
        "cp_feat_dim": cp_feat_dim,
        "dir_feat_dim": dir_feat_dim,
        "z_dim": z_dim,
        "lbd_kl": lbd_kl,
        "lbd_dir": lbd_dir,
        "lr": lr,
        "weight_decay": weight_decay,
        "lr_decay_every": lr_decay_every,
        "lr_decay_by": lr_decay_by,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "use_normals": use_normals,
        "finetune": finetune,
    }
    data_dir = args.data_dir
    base_dir = args.save_dir
    train(task, config, data_dir, base_dir)
