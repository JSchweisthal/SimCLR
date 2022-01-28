import torch
import torch.nn as nn

class MedianTripletHead(nn.Module):
    def __init__(self,# predictor, size_average=True
                 gamma=2):
        super(MedianTripletHead, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=2.) # actually, biggest margin possible should be 1+2*gamma
        self.gamma = gamma

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = input # self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = -1. * torch.matmul(pred_norm, target_norm.t()) # -2. *
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].median().unsqueeze(0))
            #down_k = torch.topk(dist[i][mask[i]==0], 5, dim=-1, largest=False)
            #down_k = down_k[0][-1].unsqueeze(0)
            #dist_an.append(down_k)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return loss_triplet # dict(loss=loss_triplet)

class SmoothTripletHead(nn.Module):
    def __init__(self,# predictor, size_average=True,
                 k, gamma=1):
        super(SmoothTripletHead, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin= 2.) # actually, biggest margin possible should be 1+gamma # changed
        self.gamma = gamma
        self.k = k

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = input # self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = -1. * torch.matmul(pred_norm, target_norm.t()) # -2.*
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0).repeat(self.k))
            down_k = torch.topk(dist[i][mask[i]==0], self.k, dim=-1, largest=False)
            down_k = down_k[0] # [-1].unsqueeze(0)
            dist_an.append(down_k)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return loss_triplet # dict(loss=loss_triplet)


class TripletNNPULoss(nn.Module):
    def __init__(self,# predictor, size_average=True,
                 prior, k, C=0, gamma=1):
        super(TripletNNPULoss, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        # self.ranking_loss = nn.MarginRankingLoss(margin=2)
        self.gamma = gamma
        self.k = k
        self.C = C
        self.prior = prior

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = input # self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = (-1. * torch.matmul(pred_norm, target_norm.t()) +1)/2 #-2.*
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())

        #dist_ap = []
        losses = []
        for i in range(n):

            dist_ap = (dist[i][mask[i]].max()) #.unsqueeze(0).repeat(k))
           # dist_ap = torch.cat(dist_ap)

            down_k, idx_down_k = torch.topk(dist[i][mask[i]==0], self.k, dim=-1, largest=False)
            # down_k = down_k[0] # [-1].unsqueeze(0)
            up_k, idx_up_k = torch.topk(dist[i][mask[i]==0], self.k, dim=-1, largest=True)
            # up_k = up_k[0] # [-1].unsqueeze(0)

            dist_pos_unl = torch.cat((down_k, up_k))

            # maybe redefine selection of unlabeled samples and esp. calculation of center
            feat_unl = torch.cat((target_norm[mask[i]==0][idx_down_k], target_norm[mask[i]==0][idx_up_k])) #, 
            center_unl = torch.mean(target_norm, dim=0)
            dist_unl = (-1. * torch.matmul(feat_unl, center_unl)  + 1)/2

        # y = torch.ones_like(dist_an)
        # loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)

            prior = self.prior # can be changed
            prior_prime = 0.5 # can be changed
            n_positive = 1
            n_unlabeled = 2*self.k

            positive_risk = prior_prime/n_positive * dist_ap
            negative_risk =  (1-prior_prime)/(n_unlabeled*(1-prior)) * torch.sum(dist_unl) - ((1-prior_prime)*prior/(n_positive*(1-prior))) *torch.sum(dist_pos_unl)

            if negative_risk < self.C: #< -self.beta and self.nnPU:
                loss_n = -self.gamma * negative_risk 
            else:
                loss_n = positive_risk + negative_risk

            losses.append(loss_n.unsqueeze(dim=0))
        loss = torch.cat(losses).mean()

        return loss # dict(loss=loss_triplet)


class HeadNNPU(nn.Module):
    def __init__(self, prior, prior_prime=0.5,
                 loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=True, latent_size=2048):
        super(HeadNNPU, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        # self.ranking_loss = nn.MarginRankingLoss(margin=2)
        self.gamma = gamma
        self.prior = prior
        self.prior_prime = 0.5
        self.latent_size = latent_size
        self.predictor = nn.Linear(latent_size, 1).cuda()
        self.prior = prior
        self.prior_prime = prior_prime
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss #lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)

    def onnpu_loss(self, inp, target, prior=None, prior_prime=None):
    # assert(inp.shape == target.shape)
        N = len(target)
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
            return (-self.gamma * negative_risk) / N
        else:
            return (positive_risk + negative_risk) / N

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, z_i, z_j):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = z_i.size(0)
        pred = self.predictor(torch.cat((z_i, z_j)))
        # pred_norm = nn.functional.normalize(pred, dim=1)
        # target_norm = nn.functional.normalize(target, dim=1)
        n = pred.size(0)
        # dist = (-1. * torch.matmul(pred_norm, target_norm.t()) +1)/2 #-2.*
        # idx = torch.arange(n)
        # mask = idx.expand(n, n).eq(idx.expand(n, n).t())

        targets = -1 * torch.ones((n, n), dtype=bool)
        targets = targets.fill_diagonal_(1)
        for i in range(batch_size): # * world_size
            targets[i, batch_size + i] = 1 # * world_size 
            targets[batch_size + i, i] = 1 # * world_size

        losses = []
        for i in range(n):
            loss_n = self.onnpu_loss(pred, targets)
            losses.append(loss_n.unsqueeze(dim=0))
        loss = torch.cat(losses).mean()

        return loss # dict(loss=loss_triplet)