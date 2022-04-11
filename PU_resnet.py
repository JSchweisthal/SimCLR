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

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    f1_epoch = 0
    model.train()
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
        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}")

    return loss_epoch #, accuracy_epoch, f1_epoch


def test(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    # accuracy_epoch = 0
    # f1_epoch = 0
    # auc_epoch= 0
    model.eval()
    
    output = np.array([])
    predicted = np.array([])
    labels = np.array([])

    for step, (x, y) in enumerate(loader):
        model.zero_grad()
        with torch.no_grad():

            x = x.to(args.device)
            y = y.to(args.device).float()

            output_step = torch.flatten(model(x)).detach()
            loss = criterion(output_step, y)
            predicted_step = ((torch.sign(output_step)+1)/2).int()

            output = np.append(output, output_step.cpu().numpy())
            predicted = np.append(predicted, predicted_step.cpu().numpy())
            labels = np.append(labels, y.cpu().numpy())

            # acc = (predicted == y).sum().item() / y.size(0)
            # accuracy_epoch += acc
            # f1 = f1_score(y.cpu().numpy(), predicted.cpu().numpy())
            # f1_epoch += f1
            # auc = roc_auc_score(y.cpu().numpy(), output.cpu().numpy())
            # auc_epoch += auc

            loss_epoch += loss.item()
    accuracy_epoch = (predicted == labels).sum()/ len(labels)
    f1_epoch = f1_score(labels, predicted)
    auc_epoch = roc_auc_score(labels, output)

    conf_mat = confusion_matrix(labels, predicted).ravel()
    print(f"TN: {conf_mat[0]}, FP: {conf_mat[1]}, FN: {conf_mat[2]}, TP: {conf_mat[3]}")


    return loss_epoch, accuracy_epoch, f1_epoch, auc_epoch


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
        if args.data_classif == "PU":
            idxtargets_up = []
            for cls in range(100):
                idxs_cls = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i]==cls]
                # vehicles_1 = ["bicycle", "bus", "motorcycle", "pickup_truck", "train"]
                # vehicles_2 = ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
                if cls in [8, 13, 48, 58, 90, 41, 69, 81, 85, 89]:
                    idxtargets_up_cls = idxs_cls[:int((1-args.PU_ratio)*len(idxs_cls))] # change here 0.2 for any other prop of labeled positive / all positives
                    idxtargets_up.extend(idxtargets_up_cls)
                    idxtargets_up.sort()
            idxtargets_up = torch.tensor(idxtargets_up)

        train_dataset.targets = torch.tensor(train_dataset.targets)
        train_dataset.targets = torch.where(torch.isin(train_dataset.targets, torch.tensor([8, 13, 48, 58, 90, 41, 69, 81, 85, 89])), 1, 0)  

        if args.data_classif == "PU":
            train_dataset.targets[idxtargets_up] = 0

        test_dataset.targets = torch.tensor(test_dataset.targets)
        test_dataset.targets = torch.where(torch.isin(test_dataset.targets, torch.tensor([8, 13, 48, 58, 90, 41, 69, 81, 85, 89])), 1, 0)
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform= TransformsSimCLR(size=args.image_size).train_transform if args.augment_data else TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )

        if args.data_pretrain == "imbalanced":
            idxs = []
            idxtargets_up = []
            for cls in range(10):
                idxs_cls = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i]==cls]
                if cls in [0, 1, 8, 9]:
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
            train_dataset.targets = torch.where(torch.isin(train_dataset.targets, torch.tensor([0, 1, 8, 9])), 1, 0)
            if args.data_classif == "PU":  
                train_dataset.targets[idxtargets_up] = 0
            train_datasubset_pu = torch.utils.data.Subset(train_dataset, idxs)

            test_dataset.targets = torch.tensor(test_dataset.targets)
            test_dataset.targets = torch.where(torch.isin(test_dataset.targets, torch.tensor([0, 1, 8, 9])), 1, 0)

        elif "2class" in args.data_pretrain :
            idxs = []
            idxtargets_up = []
            for cls in [args.class_pos, args.class_neg]:
                idxs_cls = [i for i in range(len(train_dataset.targets)) if train_dataset.targets[i]==cls]
                if cls == args.class_pos:
                    if args.data_pretrain == "2class_imbalanced":
                        idxs_cls = idxs_cls[:750]
                    if args.data_classif == "PU":  
                        idxtargets_up_cls = idxs_cls[:int((1-args.PU_ratio)*len(idxs_cls))] # change here 0.2 for any other prop of labeled positive / all positives
                        idxtargets_up.extend(idxtargets_up_cls)
                idxs.extend(idxs_cls)
            idxs.sort()
            idxtargets_up.sort()        
            idxtargets_up = torch.tensor(idxtargets_up)

            train_dataset.targets = torch.tensor(train_dataset.targets)
            train_dataset.targets = torch.where(torch.isin(train_dataset.targets, torch.tensor([args.class_pos])), 1, 0)  

            if args.data_classif == "PU":  
                train_dataset.targets[idxtargets_up] = 0
            train_datasubset_pu = torch.utils.data.Subset(train_dataset, idxs)

            idxs_test = [i for i in range(len(test_dataset.targets)) if test_dataset.targets[i] in [args.class_pos, args.class_neg]]

            # idxs_test = [i for i in range(len(test_dataset.targets)) if test_dataset.targets[i] == args.class_pos]
            # idxs_test = idxs_test[:100]
            # idxs_test_neg = [i for i in range(len(test_dataset.targets)) if test_dataset.targets[i] == args.class_neg]
            # idxs_test.extend(idxs_test_neg)
            # idxs_test.sort()
            

            test_dataset.targets = torch.tensor(test_dataset.targets)
            test_dataset.targets = torch.where(torch.isin(test_dataset.targets, torch.tensor([args.class_pos])), 1, 0)
            test_datasubset = torch.utils.data.Subset(test_dataset, idxs_test)


    elif args.dataset == "GLAUCOMA":
        from glaucoma import GLAUCOMA
        train_dataset = GLAUCOMA(
            args.dataset_dir,
            train=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )  
        if args.data_classif == "PU":
            train_dataset.labels = torch.tensor(train_dataset.labels)
            idxs_pos = [i for i in range(len(train_dataset.labels)) if train_dataset.labels[i]==1]
            idxs_pos_unl = idxs_pos[:int((1-args.PU_ratio)*len(idxs_pos))]
            idxs_pos_unl = torch.tensor(idxs_pos_unl)
            train_dataset.labels[idxs_pos_unl] = 0

        test_dataset = GLAUCOMA(
            args.dataset_dir,
            train=False,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )

    else:
        raise NotImplementedError

    if args.data_pretrain == "all":
            train_datasubset_pu = train_dataset

    if "2class" not in  args.data_pretrain:
            test_datasubset = test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_datasubset_pu, #train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasubset,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # learning rate changed!
    # criterion = torch.nn.CrossEntropyLoss()

    if args.dataset == 'CIFAR10':  
        prior = ((1-args.PU_ratio)*3/33)/(1-args.PU_ratio*3/33) if "imbalanced" in args.data_pretrain else ((1-args.PU_ratio)*2/5)/(1-args.PU_ratio*2/5)
    elif args.dataset == 'CIFAR100':
        prior = ((1-args.PU_ratio)*1/10)/(1-args.PU_ratio*1/10)
    elif args.dataset == 'GLAUCOMA':
        prior = ((1-args.PU_ratio)*817/2037)/(1-args.PU_ratio*817/2037) 

    criterion = OversampledPULoss(prior=prior, prior_prime=0.5, nnPU=True) 

    writer = None
    if args.nr == 0:
        writer = SummaryWriter('runs_final/' + args.config)

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
            writer.add_scalar("TestScore/accuracy", accuracy_epoch, epoch)
            writer.add_scalar("TestScore/F1", f1_epoch, epoch)
            writer.add_scalar("TestScore/auc", auc_epoch, epoch)

            print(
                f"[TEST]\t Loss: {round(loss_epoch / len(test_loader), 4)}\t Accuracy: {round(accuracy_epoch, 4)}\t F1: {round(f1_epoch, 4)}\t AUC: {round(auc_epoch, 4)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)




