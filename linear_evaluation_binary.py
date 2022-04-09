import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook

from sklearn.metrics import f1_score, roc_auc_score


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


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

    return loss_epoch, accuracy_epoch, f1_epoch


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
            loss = criterion(output, y)
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
    accuracy_epoch = (predicted == y).sum()/ len(y)
    f1_epoch = f1_score(labels, predicted)
    auc_epoch = roc_auc_score(labels, output)

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
            transform=TransformsSimCLR(size=args.image_size).test_transform,
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

            # idxs_test = [i for i in range(len(test_dataset.targets)) if test_dataset.targets[i] in [args.class_pos, args.class_neg]]

            idxs_test = [i for i in range(len(test_dataset.targets)) if test_dataset.targets[i] == args.class_pos]
            idxs_test = idxs_test[:250]
            idxs_test_neg = [i for i in range(len(test_dataset.targets)) if test_dataset.targets[i] == args.class_neg]
            idxs_test.extend(idxs_test_neg)
            idxs_test.sort()
            

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
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasubset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    ## Logistic Regression
    # if args.dataset in ["CIFAR10", "STL10"]:
    #     n_classes = 10  # CIFAR-10 / STL-10
    # elif args.dataset == "CIFAR100":
    #     n_classes = 100
    
    # try:
    #     if args.data_classif in ["PU", "binary"]:
    #         n_classes = 2
    # except: pass

    n_classes = 1
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    if args.data_classif == 'PU':
        if args.dataset == 'CIFAR10':  
            prior = ((1-args.PU_ratio)*3/33)/(1-args.PU_ratio*3/33) if "imbalanced" in args.data_pretrain else ((1-args.PU_ratio)*2/5)/(1-args.PU_ratio*2/5)
        elif args.dataset == 'CIFAR100':
            prior = ((1-args.PU_ratio)*1/10)/(1-args.PU_ratio*1/10)
        elif args.dataset == 'GLAUCOMA':
            prior = ((1-args.PU_ratio)*817/2037)/(1-args.PU_ratio*817/2037) 

        criterion = OversampledPULoss(prior=prior, prior_prime=0.5, nnPU=True) 
    elif args.data_classif == 'binary':
        if args.dataset == 'GLAUCOMA':
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1220/817))

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        simclr_model, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )

    print(f"File: {os.path.basename(__file__)}\nConfig: {args.config}\nStart Training...")
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch, f1_epoch = train(
            args, arr_train_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}"
            # \t Accuracy: {accuracy_epoch / len(arr_train_loader)}\t F1: {f1_epoch / len(arr_train_loader)}
        )
        # final testing
        loss_epoch, accuracy_epoch, f1_epoch, auc_epoch  = test(
            args, arr_test_loader, model, criterion, optimizer
        )
        print(
            f"[TEST]:\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}\t F1: {f1_epoch / len(arr_test_loader)}\t AUC: {auc_epoch / len(arr_test_loader)}"
        )
    # final testing
    loss_epoch, accuracy_epoch, f1_epoch, auc_epoch  = test(
        args, arr_test_loader, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}\t F1: {f1_epoch / len(arr_test_loader)}\t AUC: {auc_epoch / len(arr_test_loader)}"
    )
