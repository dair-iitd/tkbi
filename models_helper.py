# helper functions for models

import torch

EPS = 1e-9


def complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, embedding_dim):
    if o is None or o.shape[1] > 1:
        tmp1 = (s_im * r_re + s_re * r_im);  # tmp1 = tmp1.view(-1,self.embedding_dim)
        tmp2 = (s_re * r_re - s_im * r_im);  # tmp2 = tmp2.view(-1,self.embedding_dim)

        if o is not None:  # o.shape[1] > 1:
            result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1)
        else:  # all ent as neg samples
            tmp1 = tmp1.view(-1, embedding_dim)
            tmp2 = tmp2.view(-1, embedding_dim)

            o_re_tmp = o_re.view(-1, embedding_dim).transpose(0, 1)
            o_im_tmp = o_im.view(-1, embedding_dim).transpose(0, 1)
            result = tmp1 @ o_im_tmp + tmp2 @ o_re_tmp
        # result.squeeze_()
    else:
        tmp1 = o_im * r_re - o_re * r_im;  # tmp1 = tmp1.view(-1,self.embedding_dim)
        tmp2 = o_im * r_im + o_re * r_re;  # tmp2 = tmp2.view(-1,self.embedding_dim)

        if s is not None:  # s.shape[1] > 1:
            result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1)
        else:
            tmp1 = tmp1.view(-1, embedding_dim)
            tmp2 = tmp2.view(-1, embedding_dim)

            s_im_tmp = s_im.view(-1, embedding_dim).transpose(0, 1)
            s_re_tmp = s_re.view(-1, embedding_dim).transpose(0, 1)
            result = tmp1 @ s_im_tmp + tmp2 @ s_re_tmp
        # result.squeeze_()
    return result


def distmult_3way_fullsoftmax(s, r, o, s_re, r_re, o_re, embedding_dim):
    if o is None or o.shape[1] > 1:
        tmp1 = (s_re*r_re);  # tmp1 = tmp1.view(-1,self.embedding_dim)

        if o is not None:  # o.shape[1] > 1:
            result = (tmp1*o_re).sum(dim=-1)
        else:  # all ent as neg samples
            tmp1 = tmp1.view(-1, embedding_dim)

            o_re_tmp = o_re.view(-1, embedding_dim).transpose(0, 1)
            result = tmp1 @ o_re_tmp 
        # result.squeeze_()
    else:
        tmp1 = o_re*r_re;  # tmp1 = tmp1.view(-1,self.embedding_dim)

        if s is not None:  # s.shape[1] > 1:
            result = (tmp1 * s_re).sum(dim=-1)
        else:
            tmp1 = tmp1.view(-1, embedding_dim)

            s_re_tmp = s_re.view(-1, embedding_dim).transpose(0, 1)
            result = tmp1 @ s_re_tmp 
        # result.squeeze_()
    return result



def complex_3way_simple(s_re, s_im, r_re, r_im, o_re, o_im):  # <s,r,o_conjugate> when dim(s)==dim(r)==dim(o)
    sro = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
    return sro.sum(dim=-1)

def distmult_3way_simple(s, r, o):  # <s,r,o_conjugate> when dim(s)==dim(r)==dim(o)
    sro = s*r*o
    return sro.sum(dim=-1)




def complex_hadamard(a_re, a_im, b_re, b_im):
    result_re = a_re * b_re - a_im * b_im
    result_im = a_re * b_im + a_im * b_re

    return result_re, result_im

def time_regularizer(t_re, t_im):
    t_re, t_im = t_re.squeeze(), t_im.squeeze()
    t_re_diff = t_re[1:] - t_re[:-1]
    t_im_diff = t_im[1:] - t_im[:-1]

    diff = torch.sqrt(t_re_diff**2 + t_im_diff**2 + EPS)**3
    return torch.sum(diff) / (t_re.shape[0] - 1)
