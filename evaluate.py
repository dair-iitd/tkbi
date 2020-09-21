import utils
import numpy
import torch
import time
import gc
import re
import csv
import pickle

import pdb
from collections import defaultdict as dd

from utils import func_load_to_gpu
from analysis_helper import save_preds


time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2,
              "t_e_orig": 3, "t_str": 4, "t_i": 5}


class Ranker(object):
    """
    A network that ranks entities based on a scoring function. It excludes entities that have already
    been seen in any kb to compute the ranking as in ####### cite paper here ########. It can be constructed
    from any scoring function/model from models.py
    """

    def __init__(self, scoring_function, all_kb, kb_data=None,
                 filter_method='time-interval', flag_additional_filter=True, expand_mode="None", load_to_gpu=True):
        """
        The initializer\n
        :param scoring_function: The model function to used to rank the entities
        :param all_kb: A union of all the knowledge bases (train/test/valid)
        :param kb_data: kb for which this ranker is being used  (used to access timeStr dicts and file paths in some methods)
        :filter_method: which filtering method is being used (time-interval/ start-time/ no-filter/ ignore-time)
        :flag_additional_filter: whether to do partial filtering for entities with overlapping time intervals
        :expand_mode: mode of expanding data during evaluation. Inference method in ranker.forward depends on this.
        """
        super(Ranker, self).__init__()
        self.scoring_function = scoring_function
        self.all_kb = all_kb
        self.type_filter = False
        self.kb_data = kb_data
        self.flag_additional_filter = flag_additional_filter
        self.expand_mode = expand_mode
        self.load_to_gpu = load_to_gpu

        self.knowns_o = {}  # o seen w/t s,r (and t as well)
        self.knowns_s = {}
        self.knowns_r = {}
        self.knowns_t = {}

        self.knowns_t_e_o = {}
        self.knowns_t_e_s = {}

        print("building all known database from joint kb")

        print("Applying filter- {}".format(filter_method))

        self.filter_method = filter_method
        for fact in self.all_kb.facts:
            time = fact[3]  # default

            all_time = fact[3:]

            srt_list = []
            ort_list = []
            sot_list = []
            sro_list = []

            if filter_method == 'time-interval':
                #filtering based on time interval id
                time = all_time[time_index["t_i"]]


            elif filter_method == 'ignore-time':
                time = -1  # all times treated the same


            elif filter_method == 'start-time':
                #look at start time only
                time = all_time[time_index["t_s"]]


            elif filter_method == 'time-str':
                #look at the exact time
                time = all_time[time_index["t_str"]]

    
            elif filter_method == 'enumerate-time':
                #look at all time points between t_s_orig and t_e_orig
                time = all_time[time_index["t_s"]]
                
                for time in range(all_time[time_index["t_s_orig"]], all_time[time_index["t_e_orig"]] + 1):
                    srt_list.append((fact[0], fact[1], time))
                    ort_list.append((fact[2], fact[1], time))
                    sot_list.append((fact[0], fact[2], time))
                    sro_list.append((fact[0], fact[1], fact[2]))


            if filter_method != 'enumerate_time': # for all other methods
                srt_list.append((fact[0], fact[1], time))
                ort_list.append((fact[2], fact[1], time))
                sot_list.append((fact[0], fact[2], time))
                sro_list.append((fact[0], fact[1], fact[2]))

            for srt, ort, sot, sro  in zip(srt_list, ort_list, sot_list, sro_list):

                if srt not in self.knowns_o:
                    self.knowns_o[srt] = set()
                self.knowns_o[srt].add(fact[2])

                
                if ort not in self.knowns_s:
                    self.knowns_s[ort] = set()
                self.knowns_s[ort].add(fact[0])


                if sot not in self.knowns_r:
                    self.knowns_r[sot] = set()
                self.knowns_r[sot].add(fact[1])

                if sro not in self.knowns_t:  # knowns_t not being used!
                    self.knowns_t[sro] = set()
                # knowns_t not being used!
                self.knowns_t[sro].add(fact[3])

        print("converting to lists")
        for k in self.knowns_o:
            self.knowns_o[k] = list(self.knowns_o[k])
        for k in self.knowns_s:
            self.knowns_s[k] = list(self.knowns_s[k])
        for k in self.knowns_r:
            self.knowns_r[k] = list(self.knowns_r[k])
        for k in self.knowns_t:
            self.knowns_t[k] = list(self.knowns_t[k])

        for k in self.knowns_t_e_o:
            for k2 in self.knowns_t_e_o[k]:
                self.knowns_t_e_o[k][k2] = list(
                    self.knowns_t_e_o[k][k2])
        for k in self.knowns_t_e_s:
            for k2 in self.knowns_t_e_s[k]:
                self.knowns_t_e_s[k][k2] = list(
                    self.knowns_t_e_s[k][k2])

        print("done")

    def get_knowns(self, e, r, t, flag_s_o=0, flag_r=0, flag_t=0):
        """
        computes and returns the set of all entites that have been seen as a fact (s, r, _) or (_, r, o)\n
        :param e: The head(s)/tail(o) of the fact
        :param r: The relation of the fact
        :param flag_s_o: whether e is s #s is fixed we try o
        :return: All entites o such that (s, r, o) has been seen in all_kb
                 OR
                 All entites s such that (s, r, o) has been seen in all_kb
        """

        if flag_t == 0:
            if self.filter_method == 'time-interval':
                t = t[:, time_index["t_i"]]  # time-interval id
            elif self.filter_method == 'ignore-time':
                t = [-1] * len(t)  # dummy time
            elif self.filter_method == 'start-time':
                t = t[:, time_index["t_s"]]  # start time/bin
            elif self.filter_method == 'time-str':
                t = t[:, time_index["t_str"]]  # start time/bin
            elif self.filter_method == 'enumerate-time':
                t = t[:, time_index["t_s"]]  # start time/bin
            else:
                t = t[:, time_index["t_s"]]  # default

        if flag_r:
            ks = [self.knowns_r[(a, b, c)]
                  for a, b, c in zip(e, r, t)]  # e1,e2,t
        elif flag_t:
            ks = [self.knowns_t[(a, b, c)]
                  for a, b, c in zip(e, r, t)]  # e1,r,e2
        else:
            if flag_s_o:
                ks = [self.knowns_o[(a, b, c)]
                      for a, b, c in zip(e, r, t)]

            else:
                ks = [self.knowns_s[(a, b, c)]
                      for a, b, c in zip(e, r, t)]

        result = self.convert2numpy_array(ks)
        return result

    def convert2numpy_array(self, list_of_list):
        lens = [len(x) for x in list_of_list]
        max_lens = max(lens)
        ks = [numpy.pad(x, (0, max_lens - len(x)), 'edge')
              for x in list_of_list]
        result = numpy.array(ks)
        return result

    def get_start_end_scores(self, s, r, o, t):
        '''
        get start,end scores for time_complex_combined model where r+c is
        end time relation corresponding to r (where c is relation_count in kb).
        '''
        # ipdb.set_trace()
        t_start, r_start = t.clone(), r.clone()

        t_start[:, :, time_index["t_s"]
        ] = t[:, :, time_index["t_s_orig"]]
        start_scores = self.scoring_function(
            s, r_start, o, t_start).data

        t_end, r_end = t.clone(), r.clone()

        t_end[:, :, time_index["t_s"]] = t[:,
                                         :, time_index["t_e_orig"]]
        relation_count = len(self.all_kb.datamap.relation_map)
        r_end += relation_count

        end_scores = self.scoring_function(s, r_end, o, t_end).data
        scores = start_scores + end_scores

        # score_of_expected = scores.gather(1, o.data)
        return scores

    def get_expanded_scores(self, s, r, o, t):
        '''
        get start,end scores for time_complex_combined model where r+c is
        end time relation corresponding to r (where c is relation_count in kb).
        '''
        # pdb.set_trace()

        if o is not None:
            batch_size = len(o)
        else:
            batch_size = len(s)

        scores_list = []  # evaluate 1 example at a time and append to scores_list

        for i in range(batch_size):
            t_i = t[i, 0, :]
            start, end = t_i[time_index["t_s_orig"]
                         ], t_i[time_index["t_e_orig"]]
            num_times = end - start + 1

            # form t
            t_i = t_i.repeat(num_times, 1)
            t_i[:, 0] = torch.arange(start, end + 1)
            t_i = t_i.unsqueeze(1)

            # form r
            r_i = r[i, :]
            r_i = r_i.repeat(num_times, 1)

            if s is not None:
                s_i = s[i, :]
                s_i = s_i.repeat(num_times, 1)
                scores = self.scoring_function(
                    s_i, r_i, None, t_i).data
            else:
                o_i = o[i, :]
                o_i = o_i.repeat(num_times, 1)
                scores = self.scoring_function(
                    None, r_i, o_i, t_i).data

            scores = scores.sum(dim=0)
            scores_list.append(scores)

        scores = torch.stack(scores_list)

        # start_scores= self.scoring_function(s, r_start, o, t_start).data
        # score_of_expected = scores.gather(1, o.data)
        return scores

    def get_start_end_time_scores(self, s, r, o):
        start_scores = self.scoring_function(s, r, o, None).data

        relation_count = len(self.all_kb.datamap.relation_map)
        r_end = r + relation_count

        end_scores = self.scoring_function(s, r_end, o, None).data

        return start_scores, end_scores

    def forward(self, s, r, o, t, flag_s_o=0,
                flag_r=0, flag_t=0, flag_tp=0, predict_s=False, predict_o=False, load_to_gpu=False):
        """
        Returns the rank of o for the query (s, r, _) as given by the scoring function\n
        :param s: The head of the query
        :param r: The relation of the query
        :param o: The Gold object of the query
        :param knowns: The set of all o that have been seen in all_kb with (s, r, _) as given by ket_knowns above
        :return: rank of o, score of each entity and score of the gold o
        """

        if flag_r:
            scores = self.scoring_function(s, None, o, t).data
            score_of_expected = scores.gather(1, r.data)
        elif flag_t:
            # ipdb.set_trace()
            scores = self.scoring_function(s, r, o, None).data
            # ipdb.set_trace()
            score_of_expected = scores.gather(1, t.data)
        elif flag_tp:
            if t is not None:
                if t.shape[-1] == len(time_index):  # pick dimension to index
                    t_s = t[:, :, time_index["t_s_orig"]]
                    t_e = t[:, :, time_index["t_e_orig"]]
                else:
                    t_s = t[:, time_index["t_s_orig"], :]
                    t_e = t[:, time_index["t_e_orig"], :]
            else:
                t_s = t_e = None

            score_from_model = self.scoring_function(s, r, o, None)
            if len(score_from_model) == 2:  # start time scores, end time scores
                scores_s, scores_e = score_from_model
                scores_s, scores_e = (scores_s.data, scores_e.data)
                scores_s = scores_s.unsqueeze(-1)
                scores_e = scores_e.unsqueeze(1)
                scores = scores_s + scores_e
                scores = scores.view((scores.shape[0], -1))
                scores_s = scores_s.squeeze()
                scores_e = scores_e.squeeze()
                score_of_expected = scores_s.gather(
                    1, t_s.data) + scores_e.gather(1, t_e.data)
            # scores for whether the fact is likely to hold at time t
            # (for each t)
            else:
                scores_t = score_from_model
                # ipdb.set_trace()
                scores, scores_sym = self.get_interval_scores(
                    scores_t, load_to_gpu)
                score_of_expected = self.func_load_to_gpu(torch.zeros(
                    scores.shape[0]), load_to_gpu)  # scores.gather(1, s.data)
                for index in range(scores.shape[0]):
                    score_of_expected[index] = scores[index,
                                                      t_s[index], t_e[index]]
                score_of_expected = score_of_expected.unsqueeze(-1)
                scores = scores.view((scores.shape[0], -1))
                # scores_s, scores_e = self.get_all_time_pair_scores(scores_t)

        else:
            if flag_s_o:
                if self.expand_mode == "start-end-diff-relation":
                    # start and end scores
                    scores = self.get_start_end_scores(
                        s, r, None, t).data
                elif self.expand_mode == "all":
                    # enumerate facts from start to end
                    scores = self.get_expanded_scores(
                        s, r, None, t).data
                else:
                    scores = self.scoring_function(s, r, None, t).data

                score_of_expected = scores.gather(1, o.data)

            else:
                if self.expand_mode == "start-end-diff-relation":
                    # start and end scores
                    scores = self.get_start_end_scores(
                        None, r, o, t).data
                elif self.expand_mode == "all":
                    # enumerate facts from start to end
                    scores = self.get_expanded_scores(
                        None, r, o, t).data
                else:
                    scores = self.scoring_function(None, r, o, t).data

                score_of_expected = scores.gather(1, s.data)

        # scores computed, now we need to filter out the correct entities (apart from gold)
        return scores, score_of_expected


    def filtered_ranks(self, start, end, scores, score_of_expected, predict = 's', load_to_gpu=False):
        """
        Returns filtered ranks for the given scores.

        Params:
        start: start boundary for set of facts whose scores have been provided
        end:   end boundary for set of facts whose scores have been provided  
        predict: 's'/'o'/'r'/'t'

        """
        facts = self.kb_data.facts

        s = facts[start:end, 0]
        r = facts[start:end, 1]
        o = facts[start:end, 2]
        t = facts[start:end, 3:]

        
        ## get knowns for filtering

        if self.filter_method=='enumerate-time':
            indices = self.get_indices_array(t)

            if load_to_gpu:
                indices = indices.cpu()

            # repeat each array appropriate number of times
            t_repeated = t[indices]
            s_repeated = s[indices]
            o_repeated = o[indices]
            r_repeated = r[indices]

            # now, we replace start time of t_ids_repeated with each value from
            # time_index["t_s_orig"] to time_index["t_e_orig"]
            t_counter = 0
            for t_index in range(len(t)):
                t_start = t[t_index][time_index["t_s_orig"]]
                t_end = t[t_index][time_index["t_e_orig"]]
                for t_point in range(t_start, t_end+1):
                    t_repeated[t_counter][time_index["t_s"]] = t_point
                    t_counter += 1

            # now get the knowns_o and knowns_s
            knowns_o = self.get_knowns(s_repeated, r_repeated, t_repeated, flag_s_o=1)
            knowns_s = self.get_knowns(o_repeated, r_repeated, t_repeated, flag_s_o=0)

        else:
            knowns_o = self.get_knowns(s, r, t, flag_s_o=1)
            knowns_s = self.get_knowns(o, r, t, flag_s_o=0)

        if predict=='r':
            knowns_r = self.get_knowns(s, o, t, flag_r=1)
        if predict=='t':
            knowns_t = self.get_knowns(s, r, o, flag_t=1)

        ## create tensors
        if load_to_gpu:
            s = torch.autograd.Variable(torch.from_numpy(
                s).cuda().unsqueeze(1), requires_grad=False)
            r = torch.autograd.Variable(torch.from_numpy(
                r).cuda().unsqueeze(1), requires_grad=False)
            o = torch.autograd.Variable(torch.from_numpy(
                o).cuda().unsqueeze(1), requires_grad=False)
            t = torch.autograd.Variable(torch.from_numpy(
                t).cuda().unsqueeze(1), requires_grad=False)

            knowns_s = torch.from_numpy(knowns_s).cuda()
            knowns_o = torch.from_numpy(knowns_o).cuda()
            if predict=='r':
                knowns_r = torch.from_numpy(knowns_r).cuda()
            if predict=='t':
                knowns_t = torch.from_numpy(knowns_t).cuda()
        else:
            # CPU mode
            s = torch.autograd.Variable(torch.from_numpy(
                s).unsqueeze(1), requires_grad=False)
            r = torch.autograd.Variable(torch.from_numpy(
                r).unsqueeze(1), requires_grad=False)
            o = torch.autograd.Variable(torch.from_numpy(
                o).unsqueeze(1), requires_grad=False)
            t = torch.autograd.Variable(torch.from_numpy(
                t).unsqueeze(1), requires_grad=False)

            knowns_s = torch.from_numpy(knowns_s)
            knowns_o = torch.from_numpy(knowns_o)
            if predict=='r':
                knowns_r = torch.from_numpy(knowns_r)
            if predict=='t':
                knowns_t = torch.from_numpy(knowns_t)


        ## now filter
        
        if predict=='s':
            knowns = knowns_s
        elif predict=='o':
            knowns = knowns_o
        elif predict=='r':
            knowns = knowns_r
        elif predict=='t':
            knowns = knowns_t


        if self.filter_method=='enumerate-time' and predict!='t':
            # get indices array with each index 
            # repeated time interval len number of times

            if self.load_to_gpu:
                indices = self.get_indices_array(t.squeeze().cpu())
            else:
                indices = self.get_indices_array(t.squeeze())

            scores_repeated = scores[indices]
            score_of_expected_repeated = score_of_expected[indices]

            scores_repeated.scatter_(1, knowns, self.scoring_function.minimum_value)
            if 'time_transE' in self.scoring_function.__class__.__name__:
                scores_repeated *= -1  # lower the better
                score_of_expected_repeated *= -1
    
            greater = scores_repeated.ge(score_of_expected_repeated).float()

            all_ranks = greater.sum(dim=1) + 1  

            # aggregate all_ranks by taking a mean of
            # reciprocal ranks (use index_add for this)
            num_samples = len(t)

            aggregate_type = 'mr' # 'mrr' or 'mr'

            if aggregate_type=='mrr':
                all_ranks_inv = 1 / all_ranks

                all_ranks_sum = self.func_load_to_gpu(torch.zeros(num_samples), self.load_to_gpu)
                all_ranks_sum.index_add_(0, indices, all_ranks_inv)
                all_ranks_sum /= (t[:,0, time_index["t_e_orig"]] - t[:,0, time_index["t_s_orig"]] + 1).float() # divide by length of time interval

                rank = 1 / all_ranks_sum
            
            elif aggregate_type=='mr':
                all_ranks_sum = self.func_load_to_gpu(torch.zeros(num_samples), self.load_to_gpu)
                all_ranks_sum.index_add_(0, indices, all_ranks)
                all_ranks_sum /= (t[:,0,time_index["t_e_orig"]] - t[:,0,time_index["t_s_orig"]] + 1).float() # divide by length of time interval

                rank = all_ranks_sum
            else:
                raise Exception

            return rank


        if predict!='t' and self.filter_method != 'no-filter':
            scores.scatter_(
                1, knowns, self.scoring_function.minimum_value)

        if 'time_transE' in self.scoring_function.__class__.__name__:
            # print("******TRANSE SCORE*******")
            scores *= -1  # lower the better
            score_of_expected *= -1

        greater = scores.ge(score_of_expected).float()

        rank = greater.sum(dim=1) + 1  

        return rank

    def func_load_to_gpu(self, data, load_to_gpu):
        if load_to_gpu:
            data = data.cuda()
        return data

    # ---helper functions for the new filtering --#
    def get_indices_array(self, t):
        """
        Returns an indices array, where i is repeated len(t[i])
        times, where len(t[i]) equals the the length of time interval 
        t[i] (t[i][time_index["t_e"]] - t[i][time_index["t_s"]])

        param: t (tensor of times)

        Returns: indices tensor as described above
        """
        num_samples = len(t)
        # try:
        time_interval_lengths = t[:,time_index["t_e_orig"]] - t[:,time_index["t_s_orig"]] + 1

        # except:
            # pdb.set_trace()            
        indices = numpy.repeat(numpy.arange(num_samples), time_interval_lengths)

        indices = self.func_load_to_gpu(torch.tensor(indices), self.load_to_gpu)

        return indices
    # ------------------------------------------- #



def evaluate(name, ranker, kb, batch_size, predict_time=0, predict_time_pair=0, predict_rel=0, verbose=0,
             hooks=None, save_text=None, load_to_gpu=True, flag_add_reverse=0):  
    """
    Evaluates an entity ranker on a knowledge base, by computing mean reverse rank, mean rank, hits 10 etc\n
    Can also print type prediction score with higher verbosity.\n
    :param name: A name that is displayed with this evaluation on the terminal
    :param ranker: The ranker that is used to rank the entites
    :param kb: The knowledge base to evaluate on.
    :param batch_size: The batch size of each minibatch
    :param verbose: The verbosity level. More info is displayed with higher verbosity
    :param top_count: The number of entities whose details are stored
    :param hooks: The additional hooks that need to be run with each mini-batch
    :return: A dict with the mrr, mr, hits10 and hits1 of the ranker on kb
    """
    if hooks is None:
        hooks = []
    totals = {"e2": {"mrr": 0, "mr": 0, "hits10": 0, "hits1": 0}, "e1": {"mrr": 0, "mr": 0, "hits10": 0, "hits1": 0},
              "m": {"mrr": 0, "mr": 0, "hits10": 0, "hits1": 0}, "r": {
            "mrr": 0, "mr": 0, "hits10": 0, "hits1": 0}, "t": {"mrr": 0, "mr": 0, "hits10": 0, "hits1": 0},
              "tp": {"mrr": 0, "mr": 0, "hits10": 0, "hits1": 0, "iou": 0}}
    start_time = time.time()
    facts = kb.facts  
    if verbose > 0:
        totals["correct_type"] = {"e1": 0, "e2": 0}
        for hook in hooks:
            hook.begin()


    valid_list = []
    ranks_head, ranks_tail, ranks_rel = [], [], []
    top5_tail, top5_head, top3_rel = [], [], []

    # --pickle for testing--#
    scores_t_pickle = []
    # ----------------------#

    for i in range(0, int(facts.shape[0]), batch_size):
        # break

        start = i
        end = min(i + batch_size, facts.shape[0])

        s = facts[start:end, 0]
        r = facts[start:end, 1]
        o = facts[start:end, 2]

        if len(kb.facts_time_tokens) == 0:
            t = facts[start:end, 3:]
        else:
            t = kb.facts_time_tokens[start:end]  # for TA-x models

        t_ids = facts[start:end, 3:]


        if load_to_gpu:
            s = torch.autograd.Variable(torch.from_numpy(
                s).cuda().unsqueeze(1), requires_grad=False)
            r = torch.autograd.Variable(torch.from_numpy(
                r).cuda().unsqueeze(1), requires_grad=False)
            o = torch.autograd.Variable(torch.from_numpy(
                o).cuda().unsqueeze(1), requires_grad=False)
            t = torch.autograd.Variable(torch.from_numpy(
                t).cuda().unsqueeze(1), requires_grad=False)

            t_ids = torch.autograd.Variable(torch.from_numpy(t_ids).cuda(), requires_grad=False)

        else:
            # CPU mode
            s = torch.autograd.Variable(torch.from_numpy(
                s).unsqueeze(1), requires_grad=False)
            r = torch.autograd.Variable(torch.from_numpy(
                r).unsqueeze(1), requires_grad=False)
            o = torch.autograd.Variable(torch.from_numpy(
                o).unsqueeze(1), requires_grad=False)
            t = torch.autograd.Variable(torch.from_numpy(
                t).unsqueeze(1), requires_grad=False)
            t_ids = torch.autograd.Variable(torch.from_numpy(t_ids), requires_grad=False)



        if flag_add_reverse==0:
            scores_o, score_of_expected_o = ranker.forward(
                s, r, o, t, flag_s_o=1, predict_o=True, load_to_gpu=load_to_gpu)
            ranks_o = ranker.filtered_ranks(start, end, scores_o, score_of_expected_o, predict='o', load_to_gpu=load_to_gpu)
            
            
            scores_s, score_of_expected_s = ranker.forward(
                s, r, o, t, flag_s_o=0, predict_s=True, load_to_gpu=load_to_gpu)
            ranks_s = ranker.filtered_ranks(start, end, scores_s, score_of_expected_s, predict='s', load_to_gpu=load_to_gpu)

        else:
            scores_o, score_of_expected_o = ranker.forward(
                s, r, o, t, flag_s_o=1, predict_o=True, load_to_gpu=load_to_gpu)
            ranks_o = ranker.filtered_ranks(start, end, scores_o, score_of_expected_o, predict='o', load_to_gpu=load_to_gpu)
            

            scores_s, score_of_expected_s = ranker.forward(
                o, r + len(kb.datamap.relation_map), s, t, flag_s_o=1, predict_s=True, load_to_gpu=load_to_gpu)
            ranks_s = ranker.filtered_ranks(start, end, scores_s, score_of_expected_s, predict='s', load_to_gpu=load_to_gpu)
            

        if predict_rel:
            scores_r, score_of_expected_r = ranker.forward(
                s, r, o, t, flag_r=1, load_to_gpu=load_to_gpu)
            ranks_r = ranker.filtered_ranks(start, end, scores_r, score_of_expected_r, predict='r', load_to_gpu=load_to_gpu)

            totals['r']['mr'] += ranks_r.sum()
            totals['r']['mrr'] += (1.0 / ranks_r).sum()
            totals['r']['hits10'] += ranks_r.le(11).float().sum()
            totals['r']['hits1'] += ranks_r.eq(1).float().sum()

        if predict_time_pair:  
            if ranker.expand_mode == "start-end-diff-relation":
                score_from_model = ranker.get_start_end_time_scores(
                    s, r, o)
            else:
                raise Exception(
                    "Predict time pair not supported for expand_mode {}".format(ranker.expand_mode))
            # save scores for pickling later
            scores_t_pickle.append(score_from_model)

        # e1,r,?
        totals['e2']['mr'] += ranks_o.sum()
        totals['e2']['mrr'] += (1.0 / ranks_o).sum()
        totals['e2']['hits10'] += ranks_o.le(11).float().sum()
        totals['e2']['hits1'] += ranks_o.eq(1).float().sum()
        # ?,r,e2
        totals['e1']['mr'] += ranks_s.sum()
        totals['e1']['mrr'] += (1.0 / ranks_s).sum()
        totals['e1']['hits10'] += ranks_s.le(11).float().sum()
        totals['e1']['hits1'] += ranks_s.eq(1).float().sum()

        totals['m']['mr'] += (ranks_s.sum() + ranks_o.sum()) / 2.0
        totals['m']['mrr'] += ((1.0 / ranks_s).sum() +
                               (1.0 / ranks_o).sum()) / 2.0
        totals['m']['hits10'] += (ranks_s.le(11).float().sum() +
                                  ranks_o.le(11).float().sum()) / 2.0
        totals['m']['hits1'] += (ranks_s.eq(1).float().sum() +
                                 ranks_o.eq(1).float().sum()) / 2.0


        # '''
        num_facts = len(s)
        scores_s_np = scores_s.data.cpu().numpy()
        score_of_expected_s_np = score_of_expected_s.data.cpu().numpy()

        scores_o_np = scores_o.data.cpu().numpy()
        score_of_expected_o_np = score_of_expected_o.data.cpu().numpy()

        rem = kb.datamap.reverse_entity_map
        rrm = kb.datamap.reverse_relation_map
        rtm = kb.datamap.id2TimeStr


        for i in range(num_facts):
            s_val, r_val, o_val = s[i][0].item(), r[i][0].item(), o[i][0].item()
            valid_list.append(
                (rem[s_val], rrm[r_val], rem[o_val], rtm[t_ids[i][-2].item()]))


            ranks_head.append(ranks_s[i].item())
            ranks_tail.append(ranks_o[i].item())

            # restore original value, which was overwritten during
            # filtering
            scores_s_np[i][s_val] = score_of_expected_s_np[i]
            scores_o_np[i][o_val] = score_of_expected_o_np[i]

            # top5 head
            top5 = []
            for idx in numpy.argsort(scores_s_np[i])[::-1][:5]:
                top5.append(rem[idx])

            top5_head.append(top5)

            # top5 tail
            top5 = []
            for idx in numpy.argsort(scores_o_np[i])[::-1][:5]:
                top5.append(rem[idx])

            top5_tail.append(top5)


        if predict_rel:  # relation scores
            scores_r_np = scores_r.data.cpu().numpy()
            score_of_expected_r_np = score_of_expected_r.data.cpu().numpy()

            for i in range(num_facts):
                s_val, r_val, o_val = s[i][0].item(
                ), r[i][0].item(), o[i][0].item()
                valid_list.append(
                    (rem[s_val], rrm[r_val], rem[o_val], rtm[t_ids[i].item()]))

                ranks_rel.append(ranks_r[i].item())

                # restore original value, which was overwritten during
                # filtering
                scores_r_np[i][r_val] = score_of_expected_r_np[i]

                # top3 rel
                top3 = []
                for idx in numpy.argsort(scores_r_np[i])[::-1][:3]:
                    top3.append(rrm[idx])

                top3_rel.append(top3)

        # ---------------------------#
        # '''

        extra = ""

        utils.print_progress_bar(end, facts.shape[0], "Eval on %s" % name, (("|M| mrr:%3.2f|h10:%3.2f%"
                                                                             "%|h1:%3.2f|e1| mrr:%3.2f|h10:%3.2f%"
                                                                             "%|h1:%3.2f|e2| mrr:%3.2f|h10:%3.2f%"
                                                                             "%|h1:%3.2f|time %5.0f|") %
                                                                            (100.0 * totals['m']['mrr'] / end,
                                                                             100.0 * totals['m']['hits10'] / end,
                                                                             100.0 * totals['m']['hits1'] / end, 100.0 *
                                                                             totals['e1']['mrr'] / end, 100.0 *
                                                                             totals['e1']['hits10'] / end,
                                                                             100.0 * totals['e1']['hits1'] / end,
                                                                             100.0 *
                                                                             totals['e2']['mrr'] / end, 100.0 *
                                                                             totals['e2']['hits10'] / end,
                                                                             100.0 * totals['e2']['hits1'] / end,
                                                                             time.time() - start_time)) + extra,
                                 color="green")

    print("Eval on %s" % name, (("|M| mrr:%3.2f|h10:%3.2f|h1:%3.2f%"
                                 "%|e1| mrr:%3.2f|h10:%3.2f|h1:%3.2f%"
                                 "%|e2| mrr:%3.2f|h10:%3.2f|h1:%3.2f|%"
                                 "%|r | mrr:%3.2f|h10:%3.2f|h1:%3.2f|%"
                                 "%|t | mrr:%3.2f|h10:%3.2f|h1:%3.2f| mr:%3.2f|%"
                                 "%|tp| mrr:%3.2f|h10:%3.2f|h1:%3.2f| mr:%3.2f|%"
                                 "%| iou:%3.2f|%"
                                 "%time %5.0f|") %
                                (100.0 * totals['m']['mrr'] / end, 100.0 * totals['m']['hits10'] / end,
                                 100.0 * totals['m']['hits1'] / end,
                                 100.0 * totals['e1']['mrr'] / end, 100.0 *
                                 totals['e1']['hits10'] / end, 100.0 *
                                 totals['e1']['hits1'] / end,
                                 100.0 * totals['e2']['mrr'] / end, 100.0 *
                                 totals['e2']['hits10'] / end, 100.0 *
                                 totals['e2']['hits1'] / end,
                                 100.0 *
                                 totals['r']['mrr'] / end, 100.0 * totals['r']['hits10'] /
                                 end, 100.0 *
                                 totals['r']['hits1'] / end,
                                 100.0 * totals['t']['mrr'] / end, 100.0 * totals['t']['hits10'] /
                                 end, 100.0 * totals['t']['hits1'] /
                                 end, totals['t']['mr'] / end,
                                 100.0 * totals['tp']['mrr'] / end, 100.0 * totals['tp']['hits10'] /
                                 end, 100.0 * totals['tp']['hits1'] /
                                 end, totals['tp']['mr'] / end,
                                 totals['tp']['iou'] / end,
                                 time.time() - start_time)) + extra)

    # ---save predictions---#
    if save_text is not None:
        print("Saving predictions")
        save_text = "{}_{}_{}".format(save_text, kb.datamap.dataset, name)
        # name is 'valid'/'test'
        save_preds(valid_list, top5_head, top5_tail, top3_rel, ranks_head,
                   ranks_tail, ranks_rel, save_text, rel_preds=predict_rel)
        print("Predictions saved")
    # -------------------- #

    gc.collect()
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.end()
    print(" ")

    totals['m'] = {x: totals['m'][x] / facts.shape[0]
                   for x in totals['m']}
    totals['e1'] = {x: totals['e1'][x] / facts.shape[0]
                    for x in totals['e1']}
    totals['e2'] = {x: totals['e2'][x] / facts.shape[0]
                    for x in totals['e2']}
    totals['r'] = {x: totals['r'][x] / facts.shape[0]
                   for x in totals['r']}
    totals['t'] = {x: totals['t'][x] / facts.shape[0]
                   for x in totals['t']}
    totals['tp'] = {x: totals['tp'][x] / facts.shape[0]
                    for x in totals['tp']}

    return totals
