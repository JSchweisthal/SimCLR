import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from model import load_optimizer, save_model

from utils import yaml_config_hook

from sklearn.metrics import f1_score, roc_auc_score


def train(args, loader, model, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    f1_epoch = 0

    log_softmax = nn.LogSoftmax(dim=1)
    p_loader = loader[0]
    x_loader = loader[1]
    model_phi = model
    opt_phi = optimizer
    model_phi.train()


    for batch_idx in range(len(p_loader)):

        try:
            data_x, _ = x_iter.next()
        except:
            x_iter = iter(x_loader)
            data_x, _ = x_iter.next()

        try:
            data_p, _ = p_iter.next()
        except:
            p_iter = iter(p_loader)
            data_p, _ = p_iter.next()

        if torch.cuda.is_available():
            data_p, data_x = data_p.cuda(), data_x.cuda()

        # calculate the variational loss
        data_all = torch.cat((data_p, data_x))
        output_phi_all = model_phi(data_all)
        output_phi_all = log_softmax(output_phi_all)
        log_phi_all = output_phi_all[:, 1]
        idx_p = slice(0, len(data_p))
        idx_x = slice(len(data_p), len(data_all))
        log_phi_x = log_phi_all[idx_x]
        log_phi_p = log_phi_all[idx_p]
        output_phi_x = output_phi_all[idx_x]
        var_loss = torch.logsumexp(log_phi_x, dim=0) - np.log(len(log_phi_x)) - 1 * torch.mean(log_phi_p)

        # perform Mixup and calculate the regularization
        target_x = output_phi_x[:, 1].exp()
        target_p = torch.ones(len(data_p), dtype=torch.float32)
        target_p = target_p.cuda() if torch.cuda.is_available() else target_p
        rand_perm = torch.randperm(data_p.size(0))
        data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
        m = torch.distributions.beta.Beta(config.mix_alpha, config.mix_alpha)
        lam = m.sample()
        data = lam * data_x + (1 - lam) * data_p_perm
        target = lam * target_x + (1 - lam) * target_p_perm
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        out_log_phi_all = model_phi(data)
        reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

        # calculate gradients and update the network
        phi_loss = var_loss + config.lam * reg_mix_log
        opt_phi.zero_grad()
        phi_loss.backward()
        opt_phi.step()

        # # update the utilities for analysis of the model
        # reg_avg.update(reg_mix_log.item())
        # phi_loss_avg.update(phi_loss.item())
        # var_loss_avg.update(var_loss.item())
        # phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
        # phi_p_avg.update(phi_p.mean().item(), len(phi_p))
        # phi_x_avg.update(phi_x.mean().item(), len(phi_x))

        loss_epoch += phi_loss.item()


    return loss_epoch # phi_loss_avg.avg, var_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg



def test(args, loader, model, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    f1_epoch = 0
    auc_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()
        with torch.no_grad():

            x = x.to(args.device)
            y = y.to(args.device).float()

            output = torch.flatten(model(x)).detach()
            # loss = criterion(output, y)

            predicted = ((torch.sign(output)+1)/2).int()
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            f1 = f1_score(y.cpu().numpy(), predicted.cpu().numpy())
            f1_epoch += f1
            auc = roc_auc_score(y.cpu().numpy(), output.cpu().numpy())
            auc_epoch += auc

            # loss_epoch += loss.item()

    return  accuracy_epoch, f1_epoch, auc_epoch # loss_epoch,



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    # config = yaml_config_hook("./config/config_tripnnpu.yaml")
    config = yaml_config_hook(f"./config/{args.config}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
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
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )

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
        
        if args.data_classif == "PU":
            test_dataset.targets = torch.tensor(test_dataset.targets)
            test_dataset.targets = torch.where(torch.isin(test_dataset.targets, torch.tensor([0, 1, 8, 9])), 1, 0)  
    else:
        raise NotImplementedError

    if args.data_classif == "PU":
        idx_pos = [i for i in idxs if (train_dataset.targets[i]==1)]
        idx_unl = [i for i in idxs if (train_dataset.targets[i]==0)]

        train_datasubset_pos = torch.utils.data.Subset(train_dataset, idx_pos)
        train_datasubset_unl = torch.utils.data.Subset(train_dataset, idx_unl)

        train_loader_pos = torch.utils.data.DataLoader(
            train_datasubset_pos, #train_dataset,
            batch_size= int(args.batch_size/2),
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            sampler=None)
        train_loader_unl = torch.utils.data.DataLoader(
            train_datasubset_unl, #train_dataset,
            batch_size=int(args.batch_size/2),
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            sampler=None)

        train_loader = (train_loader_pos, train_loader_unl)

    # if args.data_pretrain == "all":
    #         train_datasubset_pu = train_dataset

    # train_loader = torch.utils.data.DataLoader(
    #     train_datasubset_pu, #train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=args.workers,
    # )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # encoder = get_resnet(args.resnet, pretrained=False)
    # n_features = encoder.fc.in_features  # get dimensions of fc layer

    # # load pre-trained model from checkpoint
    # simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    # model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    # simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    # simclr_model = simclr_model.to(args.device)
    # simclr_model.eval()

 
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # learning rate changed!
    # criterion = torch.nn.CrossEntropyLoss()

    writer = None
    if args.nr == 0:
        writer = SummaryWriter('runs/' + args.config)

    args.global_step = 0
    args.current_epoch = 0

    print(f"File: {os.path.basename(__file__)}\nConfig: {args.config}\nStart Training...")
    for epoch in range(args.start_epoch, args.epochs):
        # if train_sampler is not None:
        #     train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, optimizer)

        # if args.nr == 0 and scheduler:
        #     scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            # writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}" #\t lr: {round(lr, 5)}
            )

            loss_epoch, accuracy_epoch, f1_epoch, auc_epoch  = test(
                args, test_loader, model, optimizer
            )
            writer.add_scalar("Loss/test", loss_epoch / len(test_loader), epoch)
            writer.add_scalar("TestScore/accuracy", accuracy_epoch / len(test_loader), epoch)
            writer.add_scalar("TestScore/F1", f1_epoch / len(test_loader), epoch)
            writer.add_scalar("TestScore/auc", auc_epoch / len(test_loader), epoch)

            print(
                f"[TEST]\t Loss: {round(loss_epoch / len(test_loader), 4)}\t Accuracy: {round(accuracy_epoch / len(test_loader), 4)}\t F1: {round(f1_epoch / len(test_loader), 4)}\t AUC: {round(auc_epoch / len(test_loader), 4)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)




