import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet, MedianTripletHead, SmoothTripletHead, TripletNNPULoss, HeadNNPU
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        #negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

###new
def get_mask_classes(batch_size, labels):
    mat = torch.ones((batch_size, 2 * batch_size), dtype=int)
    for i in range(batch_size):
        for j in range(batch_size):
            if (labels[i] == 1) and (labels[j] == 1):
                mat[i, j] = 1
                mat[i, j + batch_size] = 1
            elif (labels[i] == 0) and (labels[j] == 0):
                mat[i, j] = 0
                mat[i, j + batch_size] = 0
            else:
                mat[i, j] = -1
                mat[i, j + batch_size] = -1
        mat[i, i] = 99

    mat = torch.cat((mat, mat), 0)
    return mat

def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), y) in enumerate(train_loader):

        if y.sum() == 0:
            print('Skip batch: No Positive Samples available')
        else:
            optimizer.zero_grad()
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)



            # new
            _, _, out_1, out_2 = model(x_i, x_j)
            out_1, out_2 = F.normalize(out_1, dim=-1), F.normalize(out_2, dim=-1)
            labels = torch.cat([y, y], dim = 0)

            # neg score
            out = torch.cat([out_1, out_2], dim=0)
            exp_sim = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)

            aug = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
            aug_pos = torch.cat([aug, aug], dim=0)[labels==1]
            aug_unl = torch.cat([aug, aug], dim=0)[labels==0]

            mask_sample = get_negative_mask(args.batch_size).cuda()
            anchor = exp_sim.masked_select(mask_sample).view(2 * args.batch_size, -1)

            mask_classes = get_mask_classes(args.batch_size, y).cuda()


            n_pos = (labels==1).sum()
            n_unl = (labels==0).sum()

            # sim_unl = exp_sim.masked_select(mask_classes==0).view(n_unl, -1)

            sim_pos = exp_sim.masked_select(mask_classes==1).view(n_pos, -1)

            sim_pos_inv = exp_sim[labels==1].masked_select((mask_classes==-1)[labels==1]).view(n_pos, -1)


            # anchor_pos = torch.where(mask_classes==-1, exp_sim, 0)[labels==1].sum(dim=1).unsqueeze(1)
            # anchor_unl = torch.where(mask_classes==-1, exp_sim, 0)[labels==0].sum(dim=1).unsqueeze(1)

            # anchor_pos = exp_sim[labels==1].masked_select((mask_classes==-1)[labels==1]).view(n_pos, -1).sum(dim=1).unsqueeze(1)
            # anchor_unl = exp_sim[labels==0].masked_select((mask_classes==-1)[labels==0]).view(n_unl, -1).sum(dim=1).unsqueeze(1)

            # anchor_pos = anchor[labels==1].sum(dim=1).unsqueeze(1)
            anchor_unl = anchor[labels==0].sum(dim=1).unsqueeze(1)

            # positive class augmentations
            Ng_pos = (-args.tau_plus * n_unl * sim_pos.mean(dim=1) + sim_pos_inv.sum(dim=1)) / (1 - args.tau_plus)

            if any(Ng_pos <= (n_unl * np.e**(-1 / args.temperature))):
                 print(f"--\n WARNING Step {step}: Too low negative loss in PU: {loss_neg}\n")

            # constrain (optional)
            Ng_pos = torch.clamp(Ng_pos, min = n_unl * np.e**(-1 / args.temperature))
            # contrastive loss
            loss_pos = (- torch.log(aug_pos / (aug_pos + Ng_pos) )).mean()

            # unlabeled class augmentations
            loss_neg = (- torch.log(aug_unl / (anchor_unl) )).mean()

            loss = loss_pos + loss_neg



            print(f"--\n WARNING Step {step}: Possible Overfitting, negative loss: {loss_neg}\n")



            # extension: bring in prior prime just for unncessary fun

    ### new end
            # loss = criterion(z_i, z_j)
            loss.backward()

            optimizer.step()
            

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            if args.nr == 0 and step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            if args.nr == 0:
                writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
                args.global_step += 1

            loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
        if args.data_pretrain == "imbalanced":
            idxs = []
            idxtargets_up = []
            for cls in range(100):
                idxs_cls = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i]==cls]
                if cls not in [23]: # cloud, keep this size and shrink all other classes to 20 %
                    if args.data_pretrain == "imbalanced":
                        idxs_cls = idxs_cls[:int(len(idxs_cls) * 0.2)]
                idxs.extend(idxs_cls)
                idxs.sort()
            idxtargets_up = torch.tensor(idxtargets_up)
            train_dataset.targets = torch.tensor(train_dataset.targets)
            train_datasubset_pu = torch.utils.data.Subset(train_dataset, idxs)
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
        
        # just take 750 instead of 5000 of the 4 vehicle (positive) classes --> P:U ratio 3k : 30k
        if args.data_pretrain == "imbalanced" or args.data_classif == "PU":
            idxs = []
            idxtargets_up = []
            for cls in range(10):
                idxs_cls = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i]==cls]
                if cls in [0, 1, 8, 9]:
                    if args.data_pretrain == "imbalanced":
                        idxs_cls = idxs_cls[:750]
                    if args.data_classif == "PU":  
                        idxtargets_up_cls = idxs_cls[:int((1-args.PU_ratio)*len(idxs_cls))] # change here 0.2 for any other prop of labeled positive / all positives
                idxs.extend(idxs_cls)
                idxs.sort()
                if args.data_classif == "PU":  
                    idxtargets_up.extend(idxtargets_up_cls)
                    idxtargets_up.sort()
            idxtargets_up = torch.tensor(idxtargets_up)

            train_dataset.targets = torch.tensor(train_dataset.targets)
            if args.data_classif == "PU":
                train_dataset.targets = torch.where(torch.isin(train_dataset.targets, torch.tensor([0, 1, 8, 9])), 1, 0)  
                train_dataset.targets[idxtargets_up] = 0
            train_datasubset_pu = torch.utils.data.Subset(train_dataset, idxs) 
    else:
        raise NotImplementedError

    if args.data_pretrain == "all":
            train_datasubset_pu = train_dataset

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_datasubset_pu, num_replicas=args.world_size, rank=rank, shuffle=True #train_dataset,
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_datasubset_pu, #train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler
    )
    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    if args.loss_pretrain == "NT_Xent":
        criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)
    elif args.loss_pretrain == "MedianTripletHead":
        criterion = MedianTripletHead()
    elif args.loss_pretrain == "SmoothTripletHead":
        criterion = SmoothTripletHead(k=args.batch_size-1)
    elif args.loss_pretrain == "TripletNNPULoss":
        criterion = TripletNNPULoss(prior=args.prior, k = args.batch_size//2, C=args.C)
    elif args.loss_pretrain == "HeadNNPU":
        criterion = HeadNNPU(prior=args.prior, latent_size=args.projection_dim)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    # config = yaml_config_hook("./config/config_tripnnpu.yaml")
    config = yaml_config_hook(f"./config/{args.config}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
    # print(args.prior)