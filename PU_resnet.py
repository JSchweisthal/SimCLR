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


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    f1_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        # perm = torch.randperm(len(y))
        # x = x[perm]
        # y = y[perm]
        # #
        x = x.to(args.device)
        y = y.to(args.device).float()
       
        output = torch.flatten(model(x))
        loss = criterion(output, y)

        predicted = ((torch.sign(output)+1)/2).int()
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        f1 = f1_score(y.cpu().numpy(), predicted.cpu().numpy())
        f1_epoch += f1

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch #, accuracy_epoch, f1_epoch


def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    f1_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device).float()

        output = torch.flatten(model(x))
        loss = criterion(output, y)

        predicted = ((torch.sign(output)+1)/2).int()
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        f1 = f1_score(y.cpu().numpy(), predicted.cpu().numpy())
        f1_epoch += f1
        auc = roc_auc_score(y.cpu().numpy(), output.cpu().numpy())

        loss_epoch += loss.item()

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")

    return loss_epoch, accuracy_epoch, f1_epoch, auc


class OversampledPULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, prior_prime=0.5, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=True):
        super(OversampledPULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.prior_prime = prior_prime
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss#lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)
    
    def forward(self, inp, target, prior=None, prior_prime=None, test=False):  
        # assert(inp.shape == target.shape)
        if prior is None:
            prior=self.prior
        if prior_prime is None:
            prior_prime=self.prior_prime
        target = target*2 - 1 # else : target -1 == self.unlabeled in next row #!!!! -1 instead of 0!!

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        # if inp.is_cuda:
        #     self.min_count = self.min_count.cuda()
        #     prior = torch.tensor(prior).cuda()
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))
        
        y_positive = self.loss_func(positive*inp) * positive
        y_positive_inv = self.loss_func(-positive*inp) * positive
        y_unlabeled = self.loss_func(-unlabeled*inp) * unlabeled

        positive_risk = prior_prime/n_positive * torch.sum(y_positive)
        negative_risk =  (1-prior_prime)/(n_unlabeled*(1-prior)) * torch.sum(y_unlabeled) - ((1-prior_prime)*prior/(n_positive*(1-prior))) *torch.sum(y_positive_inv)

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk 
        else:
            return positive_risk + negative_risk


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

    if args.data_pretrain == "all":
            train_datasubset_pu = train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_datasubset_pu, #train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

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
    model.fc = nn.Linear(2048, 1)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    prior = ((1-args.PU_ratio)*3/33)/(1-args.PU_ratio*3/33) if args.data_pretrain == "imbalanced" else ((1-args.PU_ratio)*2/5)/(1-args.PU_ratio*2/5)

    criterion = OversampledPULoss(prior=prior, prior_prime=0.5, nnPU=True) 

    # print("### Creating features from pre-trained context model ###")
    # (train_X, train_y, test_X, test_y) = get_features(
    #     simclr_model, train_loader, test_loader, args.device
    # )

    # arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
    #     train_X, train_y, test_X, test_y, args.logistic_batch_size
    # )

    writer = None
    if args.nr == 0:
        writer = SummaryWriter(args.config)

    args.global_step = 0
    args.current_epoch = 0

    print(f"File: {os.path.basename(__file__)}\nConfig: {args.config}\nStart Training...")
    for epoch in range(args.start_epoch, args.epochs):
        # if train_sampler is not None:
        #     train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer)

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
                args, test_loader, model, criterion, optimizer
            )
            writer.add_scalar("Loss/test", loss_epoch / len(test_loader), epoch)
            writer.add_scalar("TestScore/accuracy", accuracy_epoch / len(test_loader), epoch)
            writer.add_scalar("TestScore/F1", f1_epoch / len(test_loader), epoch)
            writer.add_scalar("TestScore/auc", auc_epoch / len(test_loader), epoch)

            print(
                f"[TEST]\t Loss: {round(loss_epoch / len(test_loader), 4)}\t Accuracy: {round(accuracy_epoch / len(test_loader), 4)}\t F1\
                    : {round(f1_epoch / len(test_loader), 4)}\t AUC\: {round(auc_epoch / len(test_loader), 4)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)

    # for epoch in range(args.logistic_epochs):
    #     loss_epoch, model = train(
    #         args, arr_train_loader, model, criterion, optimizer
    #     )
    #     print(
    #         f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}\t F1: {f1_epoch / len(arr_train_loader)}"
    #     )
    #     # final testing
    #     loss_epoch, accuracy_epoch, f1_epoch  = test(
    #         args, arr_test_loader, model, criterion, optimizer
    #     )
    #     print(
    #         f"[TEST]:\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}\t F1: {f1_epoch / len(arr_test_loader)}"
    #     )
    # # final testing
    # loss_epoch, accuracy_epoch, f1_epoch  = test(
    #     args, arr_test_loader, model, criterion, optimizer
    # )
    # print(
    #     f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}\t F1: {f1_epoch / len(arr_test_loader)}"
    # )


