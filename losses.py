import numpy
import torch
import torch.nn as nn

import pdb

class crossentropy_loss(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss, self).__init__()
        self.name = "crossentropy_loss"
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.zeros(positive.shape[0],
                            dtype=torch.long).cuda()  # positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        # truth[:, 0] = 1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        losses = self.loss(scores, truth)
        return losses


class crossentropy_loss_AllNeg(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss_AllNeg, self).__init__()
        self.name = "crossentropy_loss_bothNegSep"
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, truth, scores):
        # '''
        # -----------#
        truth = torch.autograd.Variable(truth, requires_grad=False)
        truth = truth.view(truth.shape[0])
        losses = self.loss(scores, truth)
        # -----------#
        # '''
        return losses


class softmax_loss_wtpos(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_wtpos, self).__init__()

    def forward(self, positive, negative_1, negative_2):
        negative = torch.cat([positive, negative_1], dim=-1)  # , negative_2], dim=-1)
        max_den = negative.max(dim=1, keepdim=True)[0].detach()
        den = (negative - max_den).exp().sum(dim=-1, keepdim=True)
        losses = (positive - max_den) - den.log()

        return - (losses.mean())


class softmax_loss_AllNeg(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_AllNeg, self).__init__()

    def forward(self, positive, scores):
        # negative = torch.cat([positive, negative_1],dim=-1)#, negative_2], dim=-1)
        max_den = scores.max(dim=1, keepdim=True)[0].detach()
        den = (scores - max_den).exp().sum(dim=-1, keepdim=True)
        print("positive", positive.shape, "scores", scores.shape, "max_den", max_den.shape)
        truth = positive.view(positive.shape[0])
        losses = (scores[:, truth] - max_den) - den.log()

        return - (losses.mean())


class test(torch.nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, positive, negative_1, negative_2):
        # '''
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        ##max_den_e2 = negative_2.max(dim=1, keepdim=True)[0].detach()
        # print("max_den_e1",max_den_e1.shape)
        # print("(negative_1-max_den_e1)",(negative_1-max_den_e1).shape)
        den_e1 = (negative_1 - max_den_e1).exp().sum(dim=-1, keepdim=True)
        ##den_e2 = (negative_2-max_den_e2).exp().sum(dim=-1, keepdim=True)
        # print("den_e1",den_e1.shape)
        ##losses = ((2*positive-max_den_e1-max_den_e2) - den_e1.log() - den_e2.log())
        losses = ((positive - max_den_e1) - den_e1.log())
        # print("positive-max_den_e1", (positive-max_den_e1).shape)
        # print("(positive-max_den_e1)-den_e1.log()",((positive-max_den_e1)-den_e1.log()).shape)
        # '''
        den_e1_noOverflow = (negative_1).exp().sum(dim=-1, keepdim=True)
        losses_noOverflow = ((positive) - den_e1_noOverflow.log())

        ##
        # part 1c
        scores_denPos = torch.cat([positive, negative_1], dim=-1)
        max_den_e1_denPos = scores_denPos.max(dim=1, keepdim=True)[0].detach()
        den_e1_denPos = (scores_denPos - max_den_e1_denPos).exp().sum(dim=-1, keepdim=True)
        losses_denPos = ((positive - max_den_e1_denPos) - den_e1_denPos.log())
        #

        ##part2
        ##scores = torch.cat([positive, negative_1, negative_2], dim=-1)
        scores = torch.cat([positive, negative_1], dim=-1)
        print("scores for pre-built functions", scores.shape)
        truth = torch.zeros(positive.shape[0],
                            dtype=torch.long).cuda()  # positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        # truth[:, 0] = 1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        losses_ce = self.loss(scores, truth)

        # part3
        loss1 = nn.LogSoftmax(dim=1)
        loss2 = nn.NLLLoss()
        scores3 = loss1(scores)
        losses_ce_2 = loss2(scores, truth)

        ##
        print("!!!")
        print("nn.LogSoftmax", scores3[:, 0], scores3[:, 0].shape)
        print("manual log softMax", losses, losses.shape)
        print("manual log softMax noOverflow", losses_noOverflow, losses_noOverflow.shape)
        print("manual log softMax with denPos", losses_denPos, losses_denPos.shape)
        print("!!!")
        ##
        print("CE loss:", losses_ce)
        print("SM loss:", -losses.mean())
        print("SM noOverflow loss:", -losses_noOverflow.mean())
        print("SM denPos loss:", -losses_denPos.mean())
        print("CE 2 loss:", losses_ce_2)

        return -losses.mean()


class logistic_loss(torch.nn.Module):
    def __init__(self):
        super(logistic_loss, self).__init__()

    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.ones(1, positive.shape[1] + negative_1.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)
        # print("Logistic loss forward:",scores*truth)
        x = torch.log(1 + torch.exp(-scores * truth))
        total = x.sum()
        return total / ((positive.shape[1] + negative_1.shape[1]) * positive.shape[0])


class hinge_loss(torch.nn.Module):
    def __init__(self):
        super(hinge_loss, self).__init__()

    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.ones(1, positive.shape[1] + negative_1.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)

        return nn.HingeEmbeddingLoss(margin=5)(scores, truth)


class margin_pairwise_loss(torch.nn.Module):
    def __init__(self, margin=10.0):
        super(margin_pairwise_loss, self).__init__()
        self.margin = margin

    def forward(self, positive, negative):
        # print("positive:{}, negative:{}".format(positive.size(), negative.size()))
        # print("diff: {}".format(diff.size()))
        # pdb.set_trace()

        diff = positive - negative + self.margin
        #diff = torch.max(diff, torch.tensor([0.0]))
        diff = torch.max(diff, torch.tensor([0.0]).cuda())
        # print("max_diff:{}".format(max_diff.size()))
        loss = diff.sum()
        # xx=input()

        return loss


