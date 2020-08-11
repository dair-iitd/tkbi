import torch
import utils
import os
from functools import reduce

import numpy as np

import torch.nn as nn

import pickle

from LSTMLinear import LSTMModel

from pairwise.gadgets import Recurrent, Pairs

import pdb

from models_helper import *


time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}

class TimePlex_base(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, normalize_time=True, init_embed=None, time_smoothing_params=None, flag_add_reverse=0,
                 has_cuda=True, time_reg_wt = 0.0, emb_reg_wt=1.0,  srt_wt=1.0, ort_wt=1.0, sot_wt=0.0):

        super(TimePlex_base, self).__init__()
        
        # self.flag_add_reverse = flag_add_reverse
        # if self.flag_add_reverse==1:
        #     relation_count*=2    

        if init_embed is None:
            init_embed = {}
            for embed_type in ["E_im", "E_re", "R_im", "R_re", "T_im", "T_re"]:
                init_embed[embed_type] = None
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.timeInterval_count = timeInterval_count

        self.has_cuda = has_cuda

        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_im"] is None else \
            init_embed["E_im"]
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_re"] is None else \
            init_embed["E_re"]

        self.R_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_im"] is None else \
            init_embed["R_im"]
        self.R_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_re"] is None else \
            init_embed["R_re"]

        # E embeddingsfor (s,r,t) and (o,r,t) component
        self.E2_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.E2_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)

        # R embeddings for (s,r,t) component
        self.Rs_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)
        self.Rs_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)

        # R embeddings for (o,r,t) component
        self.Ro_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)
        self.Ro_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim)

        # time embeddings for (s,r,t)
        self.Ts_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.Ts_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        # time embeddings for (o,r,t)
        self.To_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.To_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        ##
        self.pad_max = torch.tensor([timeInterval_count + 1])
        self.pad_min = torch.tensor([0])
        if self.has_cuda:
            self.pad_max = self.pad_max.cuda()
            self.pad_min = self.pad_min.cuda()

        # '''
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)

        torch.nn.init.normal_(self.E2_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E2_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Rs_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Rs_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ro_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ro_im.weight.data, 0, 0.05)

        # init time embeddings
        torch.nn.init.normal_(self.Ts_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.Ts_im.weight.data, 0, 0.05)

        torch.nn.init.normal_(self.To_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.To_im.weight.data, 0, 0.05)
        # '''

        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value: in time_complex_fast: ", reg)

        self.normalize_time = normalize_time

        self.batch_norm = batch_norm

        print("batch_norm not being used")

        # --srt, ort weights --#
        self.srt_wt = srt_wt 
        self.ort_wt = ort_wt 
        self.sot_wt = sot_wt

        self.time_reg_wt = time_reg_wt
        self.emb_reg_wt = emb_reg_wt

    def forward(self, s, r, o, t, flag_debug=0):
        if t is not None:
            # if not t.shape[-1]==1:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        # embeddings for s,r,t component
        rs_im = self.Rs_im(r) if r is not None else self.Rs_im.weight.unsqueeze(0)
        rs_re = self.Rs_re(r) if r is not None else self.Rs_re.weight.unsqueeze(0)

        # embeddings for o,r,t component
        ro_im = self.Ro_im(r) if r is not None else self.Ro_im.weight.unsqueeze(0)
        ro_re = self.Ro_re(r) if r is not None else self.Ro_re.weight.unsqueeze(0)

        '''
		##added extra 2 embeddings (padding) for semless time smoothing 
		Need to remove those extra embedding while calculating scores for all posibble time points
		##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
		as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
		'''
        t_re = self.Ts_re(t) if t is not None else self.Ts_re.weight.unsqueeze(0)[:, :-2, :]
        t_im = self.Ts_im(t) if t is not None else self.Ts_im.weight.unsqueeze(0)[:, :-2, :]

        t2_re = self.To_re(t) if t is not None else self.To_re.weight.unsqueeze(0)[:, :-2, :]
        t2_im = self.To_im(t) if t is not None else self.To_im.weight.unsqueeze(0)[:, :-2, :]


        # if flag_debug:
        #     print("Time embedd data")
        #     print("t_re", t_re.shape, torch.mean(t_re), torch.std(t_re))
        #     print("t_im", t_im.shape, torch.mean(t_im), torch.std(t_im))

        #########

        #########
        # '''

        if t is None:
            ##start time scores
            srt = complex_3way_simple(s_re, s_im, rs_re, rs_im, t_re, t_im)
            # ort = complex_3way_simple(o_re, o_im, ro_re, ro_im, t_re, t_im)
            ort = complex_3way_simple(t_re, t_im, ro_re, ro_im, o_re, o_im)

            sot = complex_3way_simple(s_re, s_im,  t_re, t_im, o_re, o_im)

            score = self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot

            # --for inverse facts--#
            r = r + self.relation_count / 2
            
            rs_re = self.Rs_re(r)
            rs_im = self.Rs_im(r)
            ro_re = self.Ro_re(r)
            ro_im = self.Ro_im(r)

            srt = complex_3way_simple(o_re, o_im, rs_re, rs_im, t_re, t_im)
            ort = complex_3way_simple(t_re, t_im, ro_re, ro_im, s_re, s_im)
            sot = complex_3way_simple(o_re, o_im,  t_re, t_im, s_re, s_im)

            score_inv = self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
            # ------------------- #
            
            # result = score
            result = score + score_inv

            return result


        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            sro = complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im)

            srt = complex_3way_simple(s_re, s_im, rs_re, rs_im, t_re, t_im)

            # ort = complex_3way_simple(o_re, o_im, ro_re, ro_im, t_re, t_im)
            ort = complex_3way_simple(t_re, t_im, ro_re, ro_im, o_re, o_im)

            # sot = complex_3way_simple(s_re, s_im,  t2_re, t2_im, o_re, o_im)
            sot = complex_3way_simple(s_re, s_im,  t_re, t_im, o_re, o_im)

        else:
            sro = complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, self.embedding_dim)
            
            srt = complex_3way_fullsoftmax(s, r, t, s_re, s_im, rs_re, rs_im, t_re, t_im, self.embedding_dim)
            
            # ort = complex_3way_fullsoftmax(o, r, t, o_re, o_im, ro_re, ro_im, t_re, t_im, self.embedding_dim)
            ort = complex_3way_fullsoftmax(t, r, o, t_re, t_im, ro_re, ro_im, o_re, o_im, self.embedding_dim)

            # sot = complex_3way_fullsoftmax(s, t, o, s_re, s_im, t2_re, t2_im, o_re, o_im,  self.embedding_dim)
            sot = complex_3way_fullsoftmax(s, t, o, s_re, s_im, t_re, t_im, o_re, o_im,  self.embedding_dim)


        result = sro + self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
        # result = srt

        return result

    def regularizer(self, s, r, o, t, reg_val=0):
        if t is not None:
            # if not t.shape[-1]==1:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

            # if (t.shape[-1] == len(time_index)):  # pick which dimension to index
            #     t = t[:, :, 0]
            # else:
            #     t = t[:, 0, :]

        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        ts_re = self.Ts_re(t)
        ts_im = self.Ts_im(t)
        to_re = self.To_re(t)
        to_im = self.To_im(t)

        ####
        s2_im = self.E2_im(s)
        s2_re = self.E2_re(s)
        o2_im = self.E2_im(o)
        o2_re = self.E2_re(o)

        rs_re = self.Rs_re(r)
        rs_im = self.Rs_im(r)
        ro_re = self.Ro_re(r)
        ro_im = self.Ro_im(r)

        ####

        # te_re = self.Te_re(t)
        # te_im = self.Te_im(t)
        if reg_val:
            self.reg = reg_val
        # print("CX reg", reg_val)

        #--time regularization--#
        time_reg = 0.0
        if self.time_reg_wt!=0:
            ts_re_all = (self.Ts_re.weight.unsqueeze(0))#[:, :-2, :])
            ts_im_all = (self.Ts_im.weight.unsqueeze(0))#[:, :-2, :])
            to_re_all = (self.To_re.weight.unsqueeze(0))#[:, :-2, :])
            to_im_all = (self.To_im.weight.unsqueeze(0))#[:, :-2, :])
            
            time_reg = time_regularizer(ts_re_all, ts_im_all) + time_regularizer(to_re_all, to_im_all) 
            time_reg *= self.time_reg_wt
        
        # ------------------#

        if self.reg == 2:
            # return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2 + tr_re**2 + tr_im**2).sum()
            # return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum() + (tr_re**2 + tr_im**2).sum()
            rs_sum = (rs_re ** 2 + rs_im ** 2).sum()
            ro_sum = (ro_re ** 2 + ro_im ** 2).sum()
            o2_sum = (o2_re ** 2 + o2_im ** 2).sum()
            s2_sum = (s2_re ** 2 + s2_im ** 2).sum()

            ts_sum = (ts_re ** 2 + ts_im ** 2).sum()
            to_sum = (to_re ** 2 + to_im ** 2).sum()


            ret = (s_re ** 2 + o_re ** 2 + r_re ** 2 + s_im ** 2 + r_im ** 2 + o_im ** 2).sum() + ts_sum + to_sum + rs_sum + ro_sum
            ret = self.emb_reg_wt * (ret/ s.shape[0])


        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), 
                      torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2),
                      torch.sqrt(rs_re ** 2 + rs_im ** 2),
                      torch.sqrt(ro_re ** 2 + ro_im ** 2), 
                      torch.sqrt(ts_re ** 2 + ts_im ** 2),
                      torch.sqrt(to_re ** 2 + to_im ** 2)]
            factor_wt = [1, 1, 1, 1, 1, 1, 1]
            reg = 0
            for ele,wt in zip(factor,factor_wt):
                reg += wt* torch.sum(torch.abs(ele) ** 3)
            ret =  self.emb_reg_wt * (reg / s.shape[0])
        else:
            print("Unknown reg for complex model")
            assert (False)

        return ret + time_reg


    def normalize_complex(self, T_re, T_im):
        with torch.no_grad():
            re = T_re.weight
            im = T_im.weight
            norm = re ** 2 + im ** 2
            T_re.weight.div_(norm)
            T_im.weight.div_(norm)

        return

    def post_epoch(self):
        if (self.normalize_time):
            with torch.no_grad():
                # normalize Tr
                # self.normalize_complex(self.Tr_re, self.Tr_im)
                # norm=torch.sqrt(self.Tr_re.weight**2 + self.Tr_im.weight**2)
                # self.Tr_re.weight.div_(norm)
                # self.Tr_im.weight.div_(norm)

                # self.Tr_re.weight.div_(torch.norm(self.Tr_re.weight, dim=-1, keepdim=True))
                # self.Tr_im.weight.div_(torch.norm(self.Tr_im.weight, dim=-1, keepdim=True))

                self.Ts_re.weight.div_(torch.norm(self.Ts_re.weight, dim=-1, keepdim=True))
                self.Ts_im.weight.div_(torch.norm(self.Ts_im.weight, dim=-1, keepdim=True))
                self.To_re.weight.div_(torch.norm(self.To_re.weight, dim=-1, keepdim=True))
                self.To_im.weight.div_(torch.norm(self.To_im.weight, dim=-1, keepdim=True))

        # normalize Te
        # self.normalize_complex(self.Te_re, self.Te_im)
        # self.Te_re.weight.div_(torch.norm(self.Te_re.weight, dim=-1, keepdim=True))
        # self.Te_im.weight.div_(torch.norm(self.Te_im.weight, dim=-1, keepdim=True))

        if (self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""

class TComplex_lx(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, normalize_time=False, init_embed=None, time_smoothing_params=None, emb_init=1e-2, time_reg_wt=0.0,
                 emb_reg_wt = 1.0, flag_add_reverse = True, has_cuda=True):
        super(TComplex_lx, self).__init__()
        if init_embed is None:
            init_embed = {}
            for embed_type in ["E_im", "E_re", "R_im", "R_re", "T_im", "T_re"]:
                init_embed[embed_type] = None
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.timeInterval_count = timeInterval_count

        self.has_cuda = has_cuda

        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_im"] is None else \
            init_embed["E_im"]
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_re"] is None else \
            init_embed["E_re"]

        self.R_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_im"] is None else \
            init_embed["R_im"]
        self.R_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_re"] is None else \
            init_embed["R_re"]

        self.T_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.T_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        self.pad_max = torch.tensor([timeInterval_count + 1])
        self.pad_min = torch.tensor([0])
        if self.has_cuda:
            self.pad_max = self.pad_max.cuda()
            self.pad_min = self.pad_min.cuda()

        # '''
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)

        # init time embeddings
        torch.nn.init.normal_(self.T_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.T_im.weight.data, 0, 0.05)
        # '''

        '''
        torch.nn.init.constant_(self.E_re.weight.data, emb_init)
        torch.nn.init.constant_(self.E_im.weight.data, emb_init)
        torch.nn.init.constant_(self.R_re.weight.data, emb_init)
        torch.nn.init.constant_(self.R_im.weight.data, emb_init)

        # init time embeddings
        torch.nn.init.constant_(self.T_re.weight.data, emb_init)
        torch.nn.init.constant_(self.T_im.weight.data, emb_init)
        # '''

        # self.E_re.weight.data *= emb_init
        # self.E_im.weight.data *= emb_init
        # self.R_re.weight.data *= emb_init
        # self.R_im.weight.data *= emb_init
        # self.T_re.weight.data *= emb_init
        # self.T_im.weight.data *= emb_init

        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value: in time_complex_fast: ", reg)

        self.normalize_time = normalize_time

        self.batch_norm = batch_norm

        print("batch_norm not being used")

        self.time_reg_wt = time_reg_wt
        self.emb_reg_wt = emb_reg_wt


    def forward(self, s, r, o, t, flag_debug=0):
        if t is not None:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)


        '''
		##added extra 2 embeddings (padding) for semless time smoothing 
		Need to remove those extra embedding while calculating scores for all posibble time points
		##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
		as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
		'''
        t_re = self.T_re(t) if t is not None else self.T_re.weight.unsqueeze(0)[:, :-2, :]
        t_im = self.T_im(t) if t is not None else self.T_im.weight.unsqueeze(0)[:, :-2, :]

        if flag_debug:
            print("Time embedd data")
            print("t_re", t_re.shape, torch.mean(t_re), torch.std(t_re))
            print("t_im", t_im.shape, torch.mean(t_im), torch.std(t_im))

        #########

        #########
        # '''
        r_re_t, r_im_t = complex_hadamard(r_re, r_im, t_re, t_im)

        if t is None:
            ##start time scores
            srto = complex_3way_simple(s_re, s_im, r_re_t, r_im_t, o_re, o_im)

            result = srto
            return result


        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            srto = complex_3way_simple(s_re, s_im, r_re_t, r_im_t, o_re, o_im)

        else:
            srto = complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re_t, r_im_t, o_re, o_im, self.embedding_dim)

        result = srto

        return result

    def regularizer(self, s, r, o, t, reg_val=0):
        if t is not None:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

            # # if not t.shape[-1]==1:
            # if (t.shape[-1] == len(time_index)):  # pick which dimension to index
            #     t = t[:, :, 0]
            # else:
            #     t = t[:, 0, :]

        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        t_re = self.T_re(t)
        t_im = self.T_im(t)

        r_re_t, r_im_t = complex_hadamard(r_re, r_im, t_re, t_im)

        if reg_val:
            self.reg = reg_val

        #--time regularization--#
        t_re_all = (self.T_re.weight.unsqueeze(0))#[:, :-2, :])
        t_im_all = (self.T_im.weight.unsqueeze(0))#[:, :-2, :])
        
        time_reg = self.time_reg_wt * time_regularizer(t_re_all, t_im_all)  
        # ------------------#


        if self.reg == 2:
            ret = (s_re**2 + o_re**2 + r_re**2 + s_im**2 + r_im**2 + o_im**2 + t_re**2 + t_im**2).sum()
            ret = self.emb_reg_wt * ret

        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), 
                      torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2),
                    #   torch.sqrt(t_re ** 2 + t_im ** 2)]
                      torch.sqrt(r_re_t ** 2 + r_im_t ** 2)]

            factor_wt = [1, 1, 1, 1]
            # factor_wt = [1, 1, 1]

            reg = 0
            for ele,wt in zip(factor,factor_wt):
                reg += wt* torch.sum(torch.abs(ele) ** 3)
            # pdb.set_trace()
            ret =  self.emb_reg_wt * reg / factor[0].shape[0]

        else:
            print("Unknown reg for complex model")
            assert (False)

        return ret + time_reg


    def post_epoch(self):
        if (self.normalize_time):
            print("\nNormalizing time")
            with torch.no_grad():
                self.T_re.weight.div_(torch.norm(self.T_re.weight, dim=-1, keepdim=True))
                self.T_im.weight.div_(torch.norm(self.T_im.weight, dim=-1, keepdim=True))

        if (self.unit_reg):
            print("\nApplying unit regularization")
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""

class TNTComplex_lx(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, normalize_time=False, init_embed=None, time_smoothing_params=None, emb_init=1e-2, time_reg_wt=0.0,
                 emb_reg_wt = 1.0, flag_add_reverse = True, has_cuda=True):
        super(TNTComplex_lx, self).__init__()
        if init_embed is None:
            init_embed = {}
            for embed_type in ["E_im", "E_re", "R_im", "R_re", "T_im", "T_re", "R_no_time_re", "R_no_time_im"]:
                init_embed[embed_type] = None
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.timeInterval_count = timeInterval_count

        self.has_cuda = has_cuda

        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_im"] is None else \
            init_embed["E_im"]
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim) if init_embed["E_re"] is None else \
            init_embed["E_re"]

        self.R_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_im"] is None else \
            init_embed["R_im"]
        self.R_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_re"] is None else \
            init_embed["R_re"]

        self.R_no_time_im = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_no_time_im"] is None else \
            init_embed["R_no_time_im"]
        self.R_no_time_re = torch.nn.Embedding(2 * self.relation_count, self.embedding_dim) if init_embed["R_no_time_re"] is None else \
            init_embed["R_no_time_re"]


        self.T_im = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_im"] is None else init_embed["T_im"] #padding for smoothing: 1 for start and 1 for end
        self.T_re = torch.nn.Embedding(self.timeInterval_count + 2,
                                        self.embedding_dim)  # if init_embed["T_re"] is None else init_embed["T_re"]#padding for smoothing: 1 for start and 1 for end

        self.pad_max = torch.tensor([timeInterval_count + 1])
        self.pad_min = torch.tensor([0])
        if self.has_cuda:
            self.pad_max = self.pad_max.cuda()
            self.pad_min = self.pad_min.cuda()

        # '''
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_no_time_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_no_time_im.weight.data, 0, 0.05)


        # init time embeddings
        torch.nn.init.normal_(self.T_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.T_im.weight.data, 0, 0.05)
        # '''

        '''
        torch.nn.init.constant_(self.E_re.weight.data, emb_init)
        torch.nn.init.constant_(self.E_im.weight.data, emb_init)
        torch.nn.init.constant_(self.R_re.weight.data, emb_init)
        torch.nn.init.constant_(self.R_im.weight.data, emb_init)

        # init time embeddings
        torch.nn.init.constant_(self.T_re.weight.data, emb_init)
        torch.nn.init.constant_(self.T_im.weight.data, emb_init)
        # '''

        # self.E_re.weight.data *= emb_init
        # self.E_im.weight.data *= emb_init
        # self.R_re.weight.data *= emb_init
        # self.R_im.weight.data *= emb_init
        # self.T_re.weight.data *= emb_init
        # self.T_im.weight.data *= emb_init

        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value: in time_complex_fast: ", reg)

        self.normalize_time = normalize_time

        self.batch_norm = batch_norm

        print("batch_norm not being used")

        self.time_reg_wt = time_reg_wt
        self.emb_reg_wt = emb_reg_wt


    def forward(self, s, r, o, t, flag_debug=0):
        if t is not None:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        r_no_time_im = self.R_no_time_im(r) if r is not None else self.R_no_time_im.weight.unsqueeze(0)
        r_no_time_re = self.R_no_time_re(r) if r is not None else self.R_no_time_re.weight.unsqueeze(0)


        '''
		##added extra 2 embeddings (padding) for semless time smoothing 
		Need to remove those extra embedding while calculating scores for all posibble time points
		##Currenty there is a minor bug in code -- time smoothing may not work properly until you add 1 to all i/p time points
		as seen tim tim_complex_smooth model --Resolved --underflow padding is pad_max and overflow padding is pad_max+1
		'''
        t_re = self.T_re(t) if t is not None else self.T_re.weight.unsqueeze(0)[:, :-2, :]
        t_im = self.T_im(t) if t is not None else self.T_im.weight.unsqueeze(0)[:, :-2, :]

        if flag_debug:
            print("Time embedd data")
            print("t_re", t_re.shape, torch.mean(t_re), torch.std(t_re))
            print("t_im", t_im.shape, torch.mean(t_im), torch.std(t_im))

        #########

        #########
        # '''
        r_re_t, r_im_t = complex_hadamard(r_re, r_im, t_re, t_im)

        r_re_t = r_re_t + r_no_time_re
        r_im_t = r_im_t + r_no_time_im
        

        if t is None:
            ##start time scores
            srto = complex_3way_simple(s_re, s_im, r_re_t, r_im_t, o_re, o_im)

            # for inverse facts
            r = r + self.relation_count / 2
            
            r_re = self.R_re(r)
            r_im = self.R_im(r)
            r_no_time_re = self.R_no_time_re(r)
            r_no_time_im = self.R_no_time_im(r)
            r_re_t, r_im_t = complex_hadamard(r_re, r_im, t_re, t_im)

            r_re_t = r_re_t + r_no_time_re
            r_im_t = r_im_t + r_no_time_im

            srto_inv = complex_3way_simple(o_re, o_im, r_re_t, r_im_t, s_re, s_im)


            # result = srto 
            result = srto + srto_inv
            return result


        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            srto = complex_3way_simple(s_re, s_im, r_re_t, r_im_t, o_re, o_im)

        else:
            srto = complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re_t, r_im_t, o_re, o_im, self.embedding_dim)

        result = srto

        return result

    def regularizer(self, s, r, o, t, reg_val=0):
        if t is not None:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]

            # # if not t.shape[-1]==1:
            # if (t.shape[-1] == len(time_index)):  # pick which dimension to index
            #     t = t[:, :, 0]
            # else:
            #     t = t[:, 0, :]

        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        t_re = self.T_re(t)
        t_im = self.T_im(t)

        r_no_time_re = self.R_no_time_re(r)
        r_no_time_im = self.R_no_time_im(r)

        r_re_t, r_im_t = complex_hadamard(r_re, r_im, t_re, t_im)

        if reg_val:
            self.reg = reg_val

        #--time regularization--#
        t_re_all = (self.T_re.weight.unsqueeze(0))#[:, :-2, :])
        t_im_all = (self.T_im.weight.unsqueeze(0))#[:, :-2, :])
        
        time_reg = self.time_reg_wt * time_regularizer(t_re_all, t_im_all)  
        # ------------------#


        if self.reg == 2:
            ret = (s_re**2 + o_re**2 + r_re**2 + s_im**2 + r_im**2 + o_im**2 + t_re**2 + t_im**2 + r_no_time_re**2 + r_no_time_im**2).sum()
            ret = self.emb_reg_wt * (ret / s_re.shape[0]) 

        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), 
                      torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2),
                    #   torch.sqrt(t_re ** 2 + t_im ** 2)]
                      torch.sqrt(r_re_t ** 2 + r_im_t ** 2),
                      torch.sqrt(r_no_time_re ** 2 + r_no_time_im ** 2)]

            factor_wt = [1, 1, 1, 1, 1]
            # factor_wt = [1, 1, 1]

            reg = 0
            for ele,wt in zip(factor,factor_wt):
                reg += wt* torch.sum(torch.abs(ele) ** 3)
            # pdb.set_trace()
            ret =  self.emb_reg_wt * reg / factor[0].shape[0]

        else:
            print("Unknown reg for complex model")
            assert (False)

        return ret + time_reg


    def post_epoch(self):
        if (self.normalize_time):
            print("\nNormalizing time")
            with torch.no_grad():
                self.T_re.weight.div_(torch.norm(self.T_re.weight, dim=-1, keepdim=True))
                self.T_im.weight.div_(torch.norm(self.T_im.weight, dim=-1, keepdim=True))

        if (self.unit_reg):
            print("\nApplying unit regularization")
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""

class TimePlex(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, batch_norm=False, reg=2,
                 train_kb=None,
                 has_cuda=True, freeze_weights=True,
                 model_path="", recurrent_args={}, pairs_args={}, pairs_wt=0.0, recurrent_wt=0.0, eval_batch_size=10, use_obj_scores=True, 
                 srt_wt=1.0, ort_wt=1.0, sot_wt=0.0, base_model_inverse=False):
        super(TimePlex, self).__init__()

        # print("Hard-coding U2_gadget_wt!!")
        # U2_gadget_wt = 5.0
        # print("Hard-coding pairwise_gadget_wt!!")
        # pairwise_gadget_wt = 1.0



        self.entity_count = entity_count
        self.relation_count = relation_count

        print("Recurrent args:",recurrent_args)
        print("Pairs args:",pairs_args)



        # if not self.base_model_inverse:
        #     self.base_model = TimePlex_base(entity_count, relation_count, timeInterval_count,
        #                                                     embedding_dim, reg=reg, srt_wt=srt_wt, ort_wt=ort_wt, sot_wt=sot_wt)
        # else:
        #     self.base_model = TimePlex_base(entity_count, 2*relation_count, timeInterval_count,
        #                                                     embedding_dim, reg=reg, srt_wt=srt_wt, ort_wt=ort_wt, sot_wt=sot_wt)




        # --Load pretrained TimePlex(base) embeddings--#
        if model_path != "":
            print("Loading embeddings from model saved at {}".format(model_path))
            state = torch.load(model_path)
            self.base_model = TimePlex_base(**state['model_arguments'])
            self.base_model.load_state_dict(state['model_weights'])
            base_model_inverse = state['model_arguments'].get('flag_add_reverse',False)
            print("Initialized base model (TimePlex)")

        else:
            raise Exception("Please provide path to Timeplex(base) embeddings")
        # ----------#

        self.embedding_dim = embedding_dim

        self.base_model_inverse = base_model_inverse
        print("***Base model inverse:{}".format(self.base_model_inverse))


        # --Freezing base model--#
        # '''
        if freeze_weights:
            print("Freezing base model weights")
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            print("Not freezing base model weights")

        self.freeze_weights = freeze_weights
        # '''
        # ----------------------#

        self.minimum_value = -(embedding_dim * embedding_dim)

        self.pairs_wt = pairs_wt
        self.recurrent_wt = recurrent_wt

        if pairs_wt!=0.0:
            self.pairs = Pairs(train_kb, entity_count, relation_count, load_to_gpu=has_cuda,
                                                eval_batch_size=eval_batch_size,
                                                use_obj_scores=use_obj_scores, **pairs_args)
            print("Initialized Pairs")
        else:
            print("Not  Initializing Pairs")



        if recurrent_wt!=0.0:
            self.recurrent = Recurrent(train_kb, entity_count, relation_count, load_to_gpu=has_cuda,
                                                eval_batch_size=eval_batch_size,
                                                use_obj_scores=use_obj_scores, **recurrent_args)
            print("Initialized Recurrent")

        else:
            print("Not Initializing Recurrent")
                    
        # pdb.set_trace()


    def forward(self, s, r, o, t, flag_debug=False):

        # if not self.base_model_inverse:
        if not self.base_model_inverse or t is None:
            base_score = self.base_model(s, r, o, t)
        else:
            rel_cnt = self.relation_count
            try:
                if s is None:
                    base_score = self.base_model(o, r + rel_cnt, s, t)
                elif o is None:
                    base_score = self.base_model(s, r, o, t)
                else:
                    base_score = self.base_model(s, r, o, t) + self.base_model(o, r + rel_cnt, s, t)
            except:
                pdb.set_trace()

        pairs_score = self.pairs(s, r, o, t) if self.pairs_wt else 0.0
        recurrent_score = self.recurrent(s, r, o, t) if self.recurrent_wt else 0.0
                
        return base_score + self.pairs_wt * pairs_score + self.recurrent_wt * recurrent_score


    def post_epoch(self):
        base_post_epoch = self.base_model.post_epoch()
        return base_post_epoch

    def regularizer(self, s, r, o, t=None):
        pairs_reg = self.pairs.regularizer(s, r, o, t) if self.pairs_wt else 0.0
        recurrent_reg = self.recurrent.regularizer(s, r, o, t) if self.recurrent_wt else 0.0
        
        # pdb.set_trace()

        if self.freeze_weights:
            return pairs_reg + recurrent_reg
        else:
            return pairs_reg + recurrent_reg + self.base_model.regularizer(s, r, o, t)



class time_transE(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=0,
                 batch_norm=False, unit_reg=False, normalize_time=True, has_cuda=True, flag_add_reverse=0):

        super(time_transE, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.timeInterval_count = timeInterval_count
        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.T = torch.nn.Embedding(self.timeInterval_count, self.embedding_dim)
        '''
		torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
		torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
		torch.nn.init.normal_(self.T_re.weight.data, 0, 0.05)
		'''

        self.minimum_value = self.embedding_dim * self.embedding_dim  # opposite for transE
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)

        self.normalize_time = normalize_time

        '''
        # init- for testing
        # emb_dir='./debug/wiki12k_hyte_emb/'
        emb_dir = './debug/yago11k_hyte_emb/'
        print("Initializing with trained weights loaded from {}".format(emb_dir))

        # emb_dir='./debug/'
        ent_init = np.load(os.path.join(emb_dir, "ent_embedding.npy"))
        rel_init = np.load(os.path.join(emb_dir, "rel_embedding.npy"))
        time_init = np.load(os.path.join(emb_dir, "time_embedding.npy"))

        print("ent_init", self.entity_count, ent_init.shape)
        print("rel_init", self.relation_count, rel_init.shape)
        print("time_init", self.timeInterval_count, time_init.shape)
        oov_embed = torch.randn(1, self.embedding_dim, dtype=torch.double)
        ent_init_new = torch.cat((torch.tensor(ent_init, dtype=torch.double), oov_embed))
        self.E.weight.data.copy_(ent_init_new)
        self.R.weight.data.copy_(torch.from_numpy(rel_init))
        self.T.weight.data.copy_(torch.from_numpy(time_init))

        '''
        #'''
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)
        torch.nn.init.xavier_uniform_(self.T.weight.data)
        #'''

    def time_projection(self, data, t):
        inner_prod = ((data * t).sum(dim=-1)).unsqueeze(
            -1)  # *t#tf.tile(tf.expand_dims(tf.reduce_sum(data*t,axis=1),axis=1),[1,self.p.inp_dim])
        prod = (t * inner_prod)
        data = data - prod
        return data

    def forward(self, s, r, o, t, flag_debug=0):

        if t is not None:
            if (t.shape[-1] == len(time_index)):
                t = t[:, :, 0]
            else:
                t = t[:, 0, :]

        s_e = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_e = self.R(r) if r is not None else self.R.weight.unsqueeze(0)
        o_e = self.E(o) if o is not None else self.E.weight.unsqueeze(0)
        t_e = self.T(t) if t is not None else self.T.weight.unsqueeze(0)

        # Time projection
        s_t = self.time_projection(s_e, t_e)
        r_t = self.time_projection(r_e, t_e)

        o_t = self.time_projection(o_e, t_e)

        result = torch.abs(s_t + r_t - o_t).sum(dim=-1)

        '''
		print("s_e", s_e.shape, s_e[0,0,:10])
		print("r_e", r_e.shape, r_e[0,0,:10])
		print("o_e", o_e.shape, o_e[0,0,:10])
		print("t_e", t_e.shape, t_e[0,0,:10])

		print("s_t", s_t.shape, s_t[0,0,:10])
		print("r_t", r_t.shape, r_t[0,0,:10])
		print("o_t", o_t.shape, o_t[0,0,:10])

		print("t", t.shape, t)

		print("result", result.shape, result)
		tmp = s_t + r_t - o_t
		print("tmp", tmp.shape,tmp)
		print("torch.abs(tmp)", torch.abs(tmp).shape,torch.abs(tmp))

		print("\n")
		'''
        return result

    def regularizer(self, s, r, o, t, reg_val=0):
        s_e = self.E(s)
        r_e = self.R(r)
        o_e = self.E(o)
        #t_e = self.T(t)
        if reg_val:
            self.reg = reg_val
        # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_e ** 2).sum() + (o_e ** 2).sum() + (r_e ** 2).sum() #+ (t_e**2).sum()
        elif self.reg == 0:
            return None
        else:
            print("Unknown reg for TransE model")
            assert (False)

    def post_epoch(self):
        if (self.normalize_time):
            with torch.no_grad():
                self.T.weight.div_(torch.norm(self.T.weight, dim=-1, keepdim=True))

        return ""

class transE(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, reg=0,
                 batch_norm=False, has_cuda=True, flag_add_reverse=0, **kwargs):

        super(transE, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = torch.nn.Embedding(self.relation_count, self.embedding_dim)

        self.minimum_value = self.embedding_dim * self.embedding_dim  # opposite for transE

        self.reg = reg
        print("Regularization value:", reg)

        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, s, r, o, t, flag_debug=0):
        s_e = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_e = self.R(r) if r is not None else self.R.weight.unsqueeze(0)
        o_e = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        result = torch.abs(s_e + r_e - o_e).sum(dim=-1)

        # return result
        return -result # negated for CE-loss


    def regularizer(self, s, r, o, t, reg_val=0):
        s_e = self.E(s)
        r_e = self.R(r)
        o_e = self.E(o)

        if reg_val:
            self.reg = reg_val

        if self.reg == 2:
            return (s_e ** 2).sum() + (o_e ** 2).sum() + (r_e ** 2).sum() 
        elif self.reg == 0:
            return None
        else:
            print("Unknown reg for TransE model")
            assert (False)

    # def post_epoch(self):
    #     return ""


def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.001)



class distmult(torch.nn.Module):
    """
	DistMult Model from Trullion et al 2014.\n
	Scoring function (s, r, o) = <s, r, o> # dot product
	"""

    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False,
                 reg=2, batch_norm=False):
        """
		The initializing function. These parameters are expected to be supplied from the command line when running the\n
		program from main.\n
		:param entity_count: The number of entities in the knowledge base/model
		:param relation_count: Number of relations in the knowledge base/model
		:param embedding_dim: The size of the embeddings of entities and relations
		:param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
		:param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
		:param display_norms: Whether to display the max and min entity and relation embedding norms with each update
		:param reg: The type of regularization (example-l1,l2)
		"""
        super(distmult, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v
        self.batch_norm = batch_norm
        if batch_norm:
            E_bn = nn.BatchNorm1d(self.embedding_dim)
            R_bn = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, s, r, o, flag_debug=0):
        """
		This is the scoring function \n
		:param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
		:param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
		:param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
		:return: The computation graph corresponding to the forward pass of the scoring function
		"""
        s_e = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_e = self.R(r)
        o_e = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        if self.batch_norm:
            s_e = self.E_bn(s_e)
            o_e = self.E_bn(o_e)
            r_e = self.R_bn(r_e)

        if self.clamp_v:
            s_e.data.clamp_(-self.clamp_v, self.clamp_v)
            r_e.data.clamp_(-self.clamp_v, self.clamp_v)
            o_e.data.clamp_(-self.clamp_v, self.clamp_v)
        # '''
        if o is None or o.shape[1] > 1:
            # # '''
            # tmp1 = s_e * r_e
            # result = (tmp1 * o_e).sum(dim=-1)
            # # '''
            # '''
            tmp1 = s_e * r_e
            if o is not None:
                result = (tmp1 * o_e).sum(dim=-1)
            else:
                # All negative samples
                o_e = o_e.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ o_e
            result = result.squeeze(1)
        # '''

        else:
            # '''
            tmp1 = o_e * r_e
            if s is not None:
                result = (tmp1 * s_e).sum(dim=-1)
            else:
                # All negative samples
                s_e = s_e.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ s_e
            result = result.squeeze(1)
        # '''
        return result

    def regularizer(self, s, r, o):
        """
		This is the regularization term \n
		:param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
		:param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
		:param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
		:return: The computation graph corresponding to the forward pass of the regularization term
		"""
        s = self.E(s)
        r = self.R(r)
        o = self.E(o)
        if self.reg == 2:
            return (s * s + o * o + r * r).sum()
        elif self.reg == 1:
            return (s.abs() + r.abs() + o.abs()).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s ** 2), torch.sqrt(o ** 2), torch.sqrt(r ** 2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg / s.shape[0]
        else:
            print("Unknown reg for distmult model")
            assert (False)

    def post_epoch(self):
        """
		Post epoch/batch processing stuff.
		:return: Any message that needs to be displayed after each batch
		"""
        if (not self.unit_reg and not self.display_norms):
            return ""
        e_norms = self.E.weight.data.norm(2, dim=-1)
        r_norms = self.R.weight.data.norm(2, dim=-1)
        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r, min_r = torch.max(r_norms), torch.min(r_norms)
        if self.unit_reg:
            self.E.weight.data.div_(e_norms.unsqueeze(1))
        if self.display_norms:
            return "E[%4f, %4f] R[%4f, %4f]" % (max_e, min_e, max_r, min_r)
        else:
            return ""




class complex(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, has_cuda=True, flag_add_reverse=0):

        super(complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)
        print("embedding_dim value:", embedding_dim)
        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        if batch_norm:
            self.E_im_bn = torch.nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = torch.nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = torch.nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = torch.nn.BatchNorm1d(self.embedding_dim)

    def forward(self, s, r, o, t, flag_debug=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)

        # r_im = self.R_im(r)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)

        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)

        # r_re = self.R_re(r)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)

        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)


        result = None
        # '''
        #
        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            result = complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im)

        else:
            result = complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, self.embedding_dim)


        return result
        # '''


    def regularizer(self, s, r, o, t, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        # if reg_val:
        # self.reg = reg_val
        # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re ** 2 + o_re ** 2 + r_re ** 2 + s_im ** 2 + r_im ** 2 + o_im ** 2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg / s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert (False)

    def post_epoch(self):
        if (self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class TA_complex(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, tem_total=41, has_cuda=True):

        super(TA_complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)

        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        # --TA specific--#
        self.lstm_re = LSTMModel(self.embedding_dim, n_layer=1)
        self.lstm_im = LSTMModel(self.embedding_dim, n_layer=1)
        self.tem_total = tem_total
        self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_dim)
        torch.nn.init.normal_(self.tem_embeddings.weight.data, 0, 0.05)

    # ------------#

    def forward(self, s, r, o, t, flag_debug=0):
        t = t.squeeze()

        # print("t :{}".format(t.size()))
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)

        # r_im = self.R_im(r)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)

        # r_re = self.R_re(r)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)

        batch_size = list(s.shape)[0] if s is not None else list(o.shape)[0]
        num_rel = list(r_re.shape)[1]
        relation_dim = list(r_re.shape)[2]

        if (r is None):
            r_re = r_re.expand(batch_size, num_rel, relation_dim)
            r_re.unsqueeze_(2)

            r_im = r_im.expand(batch_size, num_rel, relation_dim)
            r_im.unsqueeze_(2)

            t.unsqueeze_(1)
            num_token = list(t.shape)[2]
            t = t.expand(batch_size, num_rel, num_token)

            r_im = self.get_rseq_all(r_im, t, self.lstm_im, batch_size)
            r_re = self.get_rseq_all(r_re, t, self.lstm_re, batch_size)


        else:
            r_im = self.get_rseq(r_im, t, self.lstm_im, batch_size)
            r_re = self.get_rseq(r_re, t, self.lstm_re, batch_size)

        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)

        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)

        # '''
        #
        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            sro = complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im)
            return sro

        if o is None or o.shape[1] > 1:
            tmp1 = (s_im * r_re + s_re * r_im);
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = (s_re * r_re - s_im * r_im);
            tmp2 = tmp2.view(-1, self.embedding_dim)

            if o is not None:  # o.shape[1] > 1: #this doesn't work (only all ent as neg samples works)

                # print("tmp1: {}, tmp2: {}".format(tmp1.size(), tmp2.size()))
                # print("o_re size:",o_re.size())
                # print("o_im size:",o_im.size())

                result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1)

            else:  # all ent as neg samples
                o_re = o_re.view(-1, self.embedding_dim).transpose(0, 1)
                o_im = o_im.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ o_im + tmp2 @ o_re
        elif s is None or s.shape[1] > 1:
            tmp1 = o_im * r_re - o_re * r_im;
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = o_im * r_im + o_re * r_re;
            tmp2 = tmp2.view(-1, self.embedding_dim)

            if s is not None:  # s.shape[1] > 1:
                result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1)
            else:
                s_im = s_im.view(-1, self.embedding_dim).transpose(0, 1)
                s_re = s_re.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ s_im + tmp2 @ s_re
        elif r is None:
            # print("Evaluating scores for all relations")
            tmp1 = o_im * s_re - o_re * s_im;
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = o_im * s_im + o_re * s_re;
            tmp2 = tmp2.view(-1, self.embedding_dim)

            # r_im = r_im.view(-1,self.embedding_dim).transpose(0,1)
            # r_re = r_re.view(-1,self.embedding_dim).transpose(0,1)

            tmp1.unsqueeze_(1)
            tmp2.unsqueeze_(1)

            # print("tmp1:{} , tmp2:{}, r_re:{}, r_im:{}".format(tmp1.size(), tmp2.size(), r_re.size(), r_im.size() ))

            result = (tmp1 * r_im + tmp2 * r_re).sum(dim=-1)
        # print("result: {}".format(result.size()))
        # print("Done")
        # xx=input()
        # pass
        return result

    # '''

    def get_rseq(self, r_e, tem, lstm, bs):
        # r_e = r_e.unsqueeze(0).transpose(0, 1)

        tem_len = tem.shape[-1]
        # print("bs: {}, tem_len:{}".format(bs,tem_len))
        # print("r_e:{}, tem:{}".format(r_e.size(), tem.size()))

        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, tem_len, self.embedding_dim)
        # print("r_e:{}, token_e:{}".format(r_e.size(), token_e.size()))

        seq_e = torch.cat((r_e, token_e), 1)
        # print("seq_e:{}".format(seq_e.size())
        hidden_tem = lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        rseq_e.unsqueeze_(1)
        # print("rseq_e: {}\n".format(rseq_e.size()))

        return rseq_e

    def get_rseq_all(self, r_e, tem, lstm, bs):
        # r_e = r_e.unsqueeze(0).transpose(0, 1)

        tem_len = tem.shape[-1]
        num_rel = r_e.shape[1]

        # print("bs: {}, tem_len:{}".format(bs,tem_len))
        # print("r_e:{}, tem:{}".format(r_e.size(), tem.size()))

        tem = tem.contiguous()
        tem = tem.view(bs * num_rel * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, num_rel, tem_len, self.embedding_dim)
        # print("r_e:{}, token_e:{}".format(r_e.size(), token_e.size()))

        seq_e = torch.cat((r_e, token_e), 2)
        # print("seq_e:{}".format(seq_e.size()))

        seq_e = seq_e.view(bs * num_rel, tem_len + 1, self.embedding_dim)
        # print("seq_e after resizing:{}".format(seq_e.size()))

        hidden_tem = lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]

        # print("hidden_tem:{}".format(hidden_tem.size()))

        hidden_tem = hidden_tem.view(bs, num_rel, self.embedding_dim)
        # print("hidden_tem after resizing:{}".format(hidden_tem.size()))

        # hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        # rseq_e.unsqueeze_(2)
        # print("rseq_e: {}\n".format(rseq_e.size()))

        return rseq_e

    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        # if reg_val:
        # 	self.reg = reg_val
        # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re ** 2 + o_re ** 2 + r_re ** 2 + s_im ** 2 + r_im ** 2 + o_im ** 2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg / s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert (False)

    def post_epoch(self):
        if (self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class TA_distmult(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, tem_total=41, has_cuda=True):

        super(TA_distmult, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)

        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        # --TA specific--#
        self.lstm_re = LSTMModel(self.embedding_dim, n_layer=1)
        self.lstm_im = LSTMModel(self.embedding_dim, n_layer=1)
        self.tem_total = tem_total
        self.tem_embeddings = nn.Embedding(self.tem_total, self.embedding_dim)
        torch.nn.init.normal_(self.tem_embeddings.weight.data, 0, 0.05)

    # ------------#

    def forward(self, s, r, o, t, flag_debug=0):
        t = t.squeeze()

        # print("t :{}".format(t.size()))
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)

        # r_im = self.R_im(r)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)

        # r_re = self.R_re(r)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)

        batch_size = list(s.shape)[0] if s is not None else list(o.shape)[0]
        num_rel = list(r_re.shape)[1]
        relation_dim = list(r_re.shape)[2]

        if (r is None):
            r_re = r_re.expand(batch_size, num_rel, relation_dim)
            r_re.unsqueeze_(2)

            r_im = r_im.expand(batch_size, num_rel, relation_dim)
            r_im.unsqueeze_(2)

            t.unsqueeze_(1)
            num_token = list(t.shape)[2]
            t = t.expand(batch_size, num_rel, num_token)

            r_im = self.get_rseq_all(r_im, t, self.lstm_im, batch_size)
            r_re = self.get_rseq_all(r_re, t, self.lstm_re, batch_size)


        else:
            r_im = self.get_rseq(r_im, t, self.lstm_im, batch_size)
            r_re = self.get_rseq(r_re, t, self.lstm_re, batch_size)

        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)

        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)

        # '''
        #

        if s is not None and o is not None and s.shape == o.shape:  # positive samples
            sro = (s_re * r_re * o_re).sum(dim=-1)
            return sro

        if o is None or o.shape[1] > 1:
            tmp1 = (s_re * r_re);

            if o is not None:  # o.shape[1] > 1: #this doesn't work (only all ent as neg samples works)

                # print("tmp1: {}, tmp2: {}".format(tmp1.size(), tmp2.size()))
                # print("o_re size:",o_re.size())
                # print("o_im size:",o_im.size())

                result = (tmp1 * o_re).sum(dim=-1)

            else:  # all ent as neg samples
                tmp1 = tmp1.view(-1, self.embedding_dim)
                o_re = o_re.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ o_re

        elif s is None or s.shape[1] > 1:
            tmp1 = (o_re * r_re);

            if s is not None:  # s.shape[1] > 1:
                result = (tmp1 * s_re).sum(dim=-1)
            else:
                tmp1 = tmp1.view(-1, self.embedding_dim)
                s_re = s_re.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ s_re

        return result

    # '''

    def get_rseq(self, r_e, tem, lstm, bs):
        # r_e = r_e.unsqueeze(0).transpose(0, 1)

        tem_len = tem.shape[-1]
        # print("bs: {}, tem_len:{}".format(bs,tem_len))
        # print("r_e:{}, tem:{}".format(r_e.size(), tem.size()))

        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, tem_len, self.embedding_dim)
        # print("r_e:{}, token_e:{}".format(r_e.size(), token_e.size()))

        seq_e = torch.cat((r_e, token_e), 1)
        # print("seq_e:{}".format(seq_e.size()))

        hidden_tem = lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        rseq_e.unsqueeze_(1)
        # print("rseq_e: {}\n".format(rseq_e.size()))

        return rseq_e

    def get_rseq_all(self, r_e, tem, lstm, bs):
        # r_e = r_e.unsqueeze(0).transpose(0, 1)

        tem_len = tem.shape[-1]
        num_rel = r_e.shape[1]

        # print("bs: {}, tem_len:{}".format(bs,tem_len))
        # print("r_e:{}, tem:{}".format(r_e.size(), tem.size()))

        tem = tem.contiguous()
        tem = tem.view(bs * num_rel * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, num_rel, tem_len, self.embedding_dim)
        # print("r_e:{}, token_e:{}".format(r_e.size(), token_e.size()))

        seq_e = torch.cat((r_e, token_e), 2)
        # print("seq_e:{}".format(seq_e.size()))

        seq_e = seq_e.view(bs * num_rel, tem_len + 1, self.embedding_dim)
        # print("seq_e after resizing:{}".format(seq_e.size()))

        hidden_tem = lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]

        # print("hidden_tem:{}".format(hidden_tem.size()))

        hidden_tem = hidden_tem.view(bs, num_rel, self.embedding_dim)
        # print("hidden_tem after resizing:{}".format(hidden_tem.size()))

        # hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        # rseq_e.unsqueeze_(2)
        # print("rseq_e: {}\n".format(rseq_e.size()))

        return rseq_e

    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        # if reg_val:
        # 	self.reg = reg_val
        # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re ** 2 + o_re ** 2 + r_re ** 2 + s_im ** 2 + r_im ** 2 + o_im ** 2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re ** 2 + s_im ** 2), torch.sqrt(o_re ** 2 + o_im ** 2),
                      torch.sqrt(r_re ** 2 + r_im ** 2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg / s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert (False)

    def post_epoch(self):
        if (self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class DE_SimplE(torch.nn.Module):
    def __init__(self, entity_count, relation_count, timeInterval_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, normalize_time=True, init_embed=None, time_smoothing_params=None, flag_add_reverse=0,
                 has_cuda=True, time_reg_wt = 0.0, emb_reg_wt=0.0,  flag_avg_scores=0, dropout=0.4):
        super(DE_SimplE, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.dropout_layer = torch.nn.Dropout(p=0.4)


        entity_embedding_dim = int(self.embedding_dim/2)
        # entity_embedding_dim = self.embedding_dim


        #self.display_norms = display_norms
        self.E_s = torch.nn.Embedding(self.entity_count, entity_embedding_dim)
        self.E_o = torch.nn.Embedding(self.entity_count, entity_embedding_dim)

        self.R = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_inv = torch.nn.Embedding(self.relation_count, self.embedding_dim)

        ###### time #######
        
        time_embedding_dim = entity_embedding_dim

        # freq embeddings for the entities
        self.freq_s = torch.nn.Embedding(self.entity_count, time_embedding_dim).cuda()
        self.freq_o = torch.nn.Embedding(self.entity_count, time_embedding_dim).cuda()

        # phi embeddings for the entities
        self.phi_s = torch.nn.Embedding(self.entity_count, time_embedding_dim).cuda()
        self.phi_o = torch.nn.Embedding(self.entity_count, time_embedding_dim).cuda()

        # frequency embeddings for the entities
        self.amp_s = torch.nn.Embedding(self.entity_count, time_embedding_dim).cuda()
        self.amp_o = torch.nn.Embedding(self.entity_count, time_embedding_dim).cuda()

        torch.nn.init.xavier_uniform_(self.freq_s.weight)
        torch.nn.init.xavier_uniform_(self.freq_o.weight)
        torch.nn.init.xavier_uniform_(self.phi_s.weight)
        torch.nn.init.xavier_uniform_(self.phi_o.weight)
        torch.nn.init.xavier_uniform_(self.amp_s.weight)
        torch.nn.init.xavier_uniform_(self.amp_o.weight)
        ###### ######

        torch.nn.init.normal_(self.E_s.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_o.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_inv.weight.data, 0, 0.05)
        
        # torch.nn.init.xavier_uniform_(self.E_s.weight)
        # torch.nn.init.xavier_uniform_(self.E_o.weight)
        # torch.nn.init.xavier_uniform_(self.R.weight)
        # torch.nn.init.xavier_uniform_(self.R_inv.weight)

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count

        self.flag_add_reverse = flag_add_reverse
        self.flag_avg_scores  = flag_avg_scores

        self.emb_reg_wt = emb_reg_wt
        self.timeInterval_count = timeInterval_count
        self.dropout = dropout

    def forward(self, s, r, o, t, flag_debug=0):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        if t is not None:
            if (t.shape[-1] == len(time_index)):  # pick which dimension to index
                t = t[:, :, time_index["t_s"]]
            else:
                t = t[:, time_index["t_s"], :]
            t_values = t.type(torch.cuda.FloatTensor).unsqueeze(-1)
        else:
            t_values = torch.arange(0,self.timeInterval_count-1).type(torch.cuda.FloatTensor)
            t_values = t_values.unsqueeze(1).unsqueeze(0)
            batch_size = len(s)
            t_values = t_values.repeat(batch_size,1,1)
            # pdb.set_trace()



        s_e_h = self.E_s(s) if s is not None else self.E_s.weight.unsqueeze(0)
        s_e_t = self.E_s(o) if o is not None else self.E_s.weight.unsqueeze(0)
        r_e = self.R(r)
        o_e_t = self.E_o(o) if o is not None else self.E_o.weight.unsqueeze(0)
        o_e_h = self.E_o(s) if s is not None else self.E_o.weight.unsqueeze(0)
        r_e_inv = self.R_inv(r)

        # '''
        ##
        amp_e_s_h   = self.amp_s(s) if s is not None else self.amp_s.weight.unsqueeze(0)
        freq_e_s_h  = self.freq_s(s) if s is not None else self.freq_s.weight.unsqueeze(0)
        phi_e_s_h   = self.phi_s(s) if s is not None else self.phi_s.weight.unsqueeze(0)
        ti_s_h      = amp_e_s_h * torch.sin(freq_e_s_h * t_values  + phi_e_s_h) 

        amp_e_s_t   = self.amp_s(o) if o is not None else self.amp_s.weight.unsqueeze(0)
        freq_e_s_t  = self.freq_s(o) if o is not None else self.freq_s.weight.unsqueeze(0)
        phi_e_s_t   = self.phi_s(o) if o is not None else self.phi_s.weight.unsqueeze(0)
        if 1:#try:
            ti_s_t      = amp_e_s_t * torch.sin(freq_e_s_t * t_values  + phi_e_s_t)
        #except:
        #    pdb.set_trace()

        amp_e_o_t   = self.amp_o(o) if o is not None else self.amp_o.weight.unsqueeze(0)
        freq_e_o_t  = self.freq_o(o) if o is not None else self.freq_o.weight.unsqueeze(0)
        phi_e_o_t   = self.phi_o(o) if o is not None else self.phi_o.weight.unsqueeze(0)
        ti_o_t      = amp_e_o_t * torch.sin(freq_e_o_t * t_values  + phi_e_o_t)                                                  

        amp_e_o_h   = self.amp_o(s) if s is not None else self.amp_o.weight.unsqueeze(0)
        freq_e_o_h  = self.freq_o(s) if s is not None else self.freq_o.weight.unsqueeze(0)
        phi_e_o_h   = self.phi_o(s) if s is not None else self.phi_o.weight.unsqueeze(0)
        ti_o_h      = amp_e_o_h * torch.sin(freq_e_o_h * t_values  + phi_e_o_h)
        ##
        #'''

        try:
            # pdb.set_trace()
            if s is None:
                s_e_h = s_e_h.expand(ti_s_h.shape[0], -1, -1)
                o_e_h = o_e_h.expand(ti_o_h.shape[0], -1, -1)
            if o is None:
                # s_e_t
                s_e_t = s_e_t.expand(ti_s_t.shape[0], -1, -1)
                o_e_t = o_e_t.expand(ti_o_t.shape[0], -1, -1)
            elif t is None:
                s_e_h = s_e_h.repeat(1,self.timeInterval_count-1,1)
                o_e_h = o_e_h.repeat(1,self.timeInterval_count-1,1)
                s_e_t = s_e_t.repeat(1,self.timeInterval_count-1,1)
                o_e_t = o_e_t.repeat(1,self.timeInterval_count-1,1)

                
            s_e_h_ti = torch.cat((s_e_h, ti_s_h), 2) 
            s_e_t_ti = torch.cat((s_e_t, ti_s_t), 2)   
            o_e_t_ti = torch.cat((o_e_t, ti_o_t), 2)   
            o_e_h_ti = torch.cat((o_e_h, ti_o_h), 2)   

            # s_e_h_ti = s_e_h
            # s_e_t_ti = s_e_t
            # o_e_t_ti = o_e_t
            # o_e_h_ti = o_e_h 


        except:
            pdb.set_trace()


        result = distmult_3way_simple(s_e_h_ti, r_e, o_e_t_ti)
        result_inv = distmult_3way_simple(o_e_h_ti, r_e_inv, s_e_t_ti)

        score =  (result + result_inv)/2
        # pdb.set_trace()
        score = torch.nn.functional.dropout(score, p=self.dropout, training=((s is not None) and (o is not None) and (t is not None)) )
        # score = self.dropout_layer(score)
        return score



        

    def regularizer(self, s, r, o, t):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """

        s1 = self.E_s(s)
        r1 = self.R(r)
        o1 = self.E_o(o)

        o2 = self.E_s(o)
        r2 = self.R_inv(r)
        s2 = self.E_o(s)

        s_amp   = self.amp_s(s) 
        s_freq   = self.amp_s(s) 
        s_phi   = self.phi_s(s) 

        o_amp   = self.amp_o(o) 
        o_freq   = self.amp_o(o) 
        o_phi   = self.phi_o(o) 



        if self.reg==2:
            ret =  (s1**2).sum() + (o1**2).sum() + (r1**2).sum() + (o2**2).sum() + (r2**2).sum() + (s2**2).sum()

            ret += (s_amp**2).sum() + (s_freq**2).sum() + (s_phi**2).sum() 
            ret += (o_amp**2).sum() + (o_freq**2).sum() + (o_phi**2).sum() 

            return self.emb_reg_wt* (ret/ s.shape[0])
        elif self.reg==22:
            ret=(
                          (torch.norm(self.E_s.weight, p=2)**2)    + (torch.norm(self.E_o.weight, p=2)**2) 
                        + (torch.norm(self.R.weight, p=2)**2)      + (torch.norm(self.R_inv.weight,p=2)**2)
                        + (torch.norm(self.freq_s.weight, p=2)**2) + (torch.norm(self.freq_o.weight, p=2)**2) 
                        + (torch.norm(self.phi_s.weight, p=2)**2)  + (torch.norm(self.phi_o.weight, p=2)**2)
                        + (torch.norm(self.amp_s.weight, p=2)**2)  + (torch.norm(self.amp_o.weight, p=2)**2)
                    )
            return self.emb_reg_wt* (ret/ s.shape[0])
            
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s**2),torch.sqrt(o**2),torch.sqrt(r**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        if(self.unit_reg):
            self.E_s.weight.data.div_(self.E_s.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_o.weight.data.div_(self.E_o.weight.data.norm(2, dim=-1, keepdim=True))
            self.R.weight.data.div_(self.R.weight.data.norm(2, dim=-1, keepdim=True))
        return ""
