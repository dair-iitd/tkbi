import sys

sys.path.append('../')

import numpy
import torch

from collections import defaultdict

from pairwise.helper import *
from pairwise.helper import func_load_to_gpu

import pickle
import pdb

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


class ProbDensityScorer(torch.nn.Module):
    """
    Model for scoring relation pair differences via
    probability density functions
    """

    def __init__(self, train_kb, relation_count, t1_map, t2_map, min_support=20, mode='subject', distribution='gaussian',
                 trainable=False, load_to_gpu=True, offset_init=-0.2):#-0.8):#-0.2):
        """
        :param train_kb: train kb
        :param t1_map: mapping time interval id to integer time (t1)
        :param t2_map: mapping time interval id to integer time (t2)

        :param min_support: minimum support to consider for relation pair,
               if lesser relation pair score is always 0
        :param mode: 'subject' or 'object'
        :param distribution: 'gaussian'/'laplacian'
        :param trainable: True if parameters of probability distribution are to be trained
        :param load_to_gpu: True if gpu is to be used
        """
        super(ProbDensityScorer, self).__init__()
        self.load_to_gpu = load_to_gpu

        if distribution in ['gaussian', 'laplacian']:
            self.distribution = distribution
        else:
            raise Exception("Unknown distribution")

        self.relation_count = relation_count

        # define relation pair mask (to make certain relation pairs density 0)
        self.use_mask = True

        if self.use_mask:
            print("***Using relation pair mask")
        else:
            print("***Not using relation pair mask")

        self.mask_r_r = func_load_to_gpu(torch.ones(relation_count, relation_count ), self.load_to_gpu)

        # mask out the dummy relation
        self.mask_r_r[relation_count-1, :] = 0
        self.mask_r_r[:, relation_count-1] = 0
        # --------------------------------------------------------------------- #


        mean_r_r, var_r_r = mean_variance(train_kb.facts, t1_map, t2_map, relation_count, min_support=min_support,
                                          mode=mode, mask = self.mask_r_r)

        # init rxr tensors for mean & variance
        self.mean_r_r = func_load_to_gpu(torch.zeros(relation_count, relation_count ), self.load_to_gpu)
        self.var_r_r = func_load_to_gpu(torch.zeros(relation_count, relation_count), self.load_to_gpu)

        self.mean_r_r[:relation_count, :relation_count] = mean_r_r
        self.var_r_r[:relation_count, :relation_count] = var_r_r
        # --------------------- #

        # set dummy relation's mean high and var close to 0, so that
        # it is essentially ignored during computing scores
        inf = 1000
        min_var = 1.0 #0.01
        self.mean_r_r[relation_count-1, :] = -inf
        self.mean_r_r[:, relation_count-1] = -inf

        self.var_r_r[relation_count-1, :] = min_var
        self.var_r_r[:, relation_count-1] = min_var
        # --------------------- #

        # --define offset, to be added to density-- #
        self.offset_init = offset_init
        self.offset_r_r = func_load_to_gpu(torch.ones(relation_count, relation_count)*self.offset_init, self.load_to_gpu)
        # -----------------------------  #

        # --make offsets model parameters to make them trainable
        # (requires_grad can be passed as an argument) --#
        self.offset_r_r = torch.nn.Parameter(self.offset_r_r, requires_grad=trainable)
        if trainable:
            print("Offset weights trainable")
        else:
            print("Offset weights frozen")
        # -------------------------------------------------------  #

        # --make mean/var model parameters, not trainable for now-- #
        mean_var_trainable = False#True

        if mean_var_trainable:
            print("Mean/Variances trainable")
            self.mean_r_r = torch.nn.Parameter(self.mean_r_r, requires_grad=True)
            self.var_r_r = torch.nn.Parameter(self.var_r_r, requires_grad=True)
        else:
            print("Mean/Variances frozen")
            self.mean_r_r = torch.nn.Parameter(self.mean_r_r, requires_grad=False)
            self.var_r_r = torch.nn.Parameter(self.var_r_r, requires_grad=False)
        # -------------------------------------------------------  #

        # self.use_offset=False
        self.use_offset=True
        print("***self.use_offset:", self.use_offset)



    def forward(self, r_query, r_link, time_diff):
        # check the ordering (time_diff should be r_link - r_query)
        time_diff = time_diff.squeeze()
        mean = self.mean_r_r[r_link, r_query]
        var = self.var_r_r[r_link, r_query]
        offset = self.offset_r_r[r_link, r_query]


        # --compute prob density-- #
        if self.distribution == 'gaussian':
            x = -(time_diff - mean) ** 2
        elif self.distribution == 'laplacian':
            x = -torch.abs(time_diff - mean)

        x = x / (2 * var)
        prob = torch.exp(x)
        # -------------------- #

        if self.use_mask:
            prob = prob * self.mask_r_r[r_link, r_query]

        if self.use_offset:
            prob = prob + offset

        return prob 

    def regularizer(self):
        return (self.offset_r_r**2).sum()


class RecurringFactScorer(torch.nn.Module):
    """
    Model for scoring relation pair differences via
    probability density functions
    """

    def __init__(self, train_kb, relation_count, t1_map, min_support=20, mode='subject', distribution='gaussian',
                 trainable=False, load_to_gpu=True, offset_init=-0.2, gadget_type = 'recurring-fact'):
        """
        :param train_kb: train kb
        :param t1_map: mapping time interval id to integer time (t1)
        :param t2_map: mapping time interval id to integer time (t2)

        :param min_support: minimum support to consider for relation pair,
               if lesser relation pair score is always 0
        :param mode: 'subject' or 'object'
        :param distribution: 'gaussian'/'laplacian'
        :param trainable: True if parameters of probability distribution are to be trained
        :param load_to_gpu: True if gpu is to be used
        """
        super(RecurringFactScorer, self).__init__()
        self.load_to_gpu = load_to_gpu

        if distribution in ['gaussian', 'laplacian']:
            self.distribution = distribution
        else:
            raise Exception("Unknown distribution")

        self.relation_count = relation_count

        if gadget_type == 'recurring-fact':
            mean_r, var_r = recurring_mean_variance(train_kb.facts, t1_map, relation_count, min_support=min_support,
                                            mode = mode)
        elif gadget_type == 'recurring-relation':
            mean_r, var_r = recurring_relation_mean_variance(train_kb.facts, t1_map, relation_count, min_support=min_support,
                                              mode = mode)


        # init rxr tensors for mean & variance
        self.mean_r = func_load_to_gpu(torch.zeros(relation_count), self.load_to_gpu)
        self.var_r = func_load_to_gpu(torch.zeros(relation_count), self.load_to_gpu)

        self.mean_r[:relation_count] = mean_r
        self.var_r[:relation_count] = var_r

        # --------------------- #
        # set dummy relation's mean high and var close to 0, so that
        # it is essentially ignored during computing scores
        inf = 1000
        self.mean_r[relation_count-1] = -inf

        self.var_r[relation_count-1] = 0.01
        # --------------------- #

        # --define offset, to be added to density-- #
        self.offset_init = offset_init
        self.offset_r = func_load_to_gpu(torch.ones(relation_count)*self.offset_init, self.load_to_gpu)
        # -----------------------------  #

        # --relation-wise weights --#
        self.W_r = func_load_to_gpu(torch.ones(relation_count), self.load_to_gpu)
        # -------------------------- #

        # --make relations model parameters to make them trainable --#
        self.W_r = torch.nn.Parameter(self.W_r, requires_grad=trainable)
        # self.W_r = torch.nn.Parameter(self.W_r, requires_grad=False)

        # -------------------------- #



        # --make offsets model parameters to make them trainable
        # (requires_grad can be passed as an argument) --#
        self.offset_r = torch.nn.Parameter(self.offset_r, requires_grad=trainable)
        # -------------------------------------------------------  #

        # --make mean/var model parameters, not trainable for now
        self.mean_r = torch.nn.Parameter(self.mean_r, requires_grad=False)
        self.var_r = torch.nn.Parameter(self.var_r, requires_grad=False)
        # -------------------------------------------------------  #


    def forward(self, r_query, time_diff):
        # check the ordering (time_diff should be r_link - r_query)
        time_diff = time_diff.squeeze()
        mean = self.mean_r[r_query]
        var = self.var_r[r_query]
        offset = self.offset_r[r_query]
        weights = self.W_r[r_query]

        # --compute prob density-- #
        if self.distribution == 'gaussian':
            x = -(time_diff - mean) ** 2
        elif self.distribution == 'laplacian':
            x = -torch.abs(time_diff - mean)

        x = x / (2 * var)
        prob = torch.exp(x)
        prob *= weights
        # -------------------- #

        prob = prob + offset

        return prob

    def regularizer(self):
        return (self.offset_r**2).sum() + (self.W_r**2).sum()
