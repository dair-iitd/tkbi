import sys

sys.path.append('../')

from abc import ABCMeta, abstractmethod

import numpy
import torch

from collections import defaultdict

from utils import func_load_to_gpu

from pairwise.prob_density_scorer import RecurringFactScorer, ProbDensityScorer

import pickle
import pdb

# --Note: Define this in one place from where all files (kb.py included) use it------ #
YEARMIN = 0
YEARMAX = 3000

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}

class Gadget(torch.nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, train_kb, entity_count, relation_count, load_to_gpu=True, max_neighbours = None,
            eval_batch_size=10, use_obj_scores=False, t1_mode="start", t2_mode="start", reg_wt = 0.0):
        """
        :param train_kb: train_kb containing training facts and Datamap (useful to get exact time)
        :param entity_count: number of entities
        :param relation_count: number of relations
        :param load_to_gpu: True if computations to be performed on GPU
        :param max_neighbours: max size of neighbourhood of entity to be sampled, None means no bound
        :param use_obj_scores: True if object neighbourhood is to be considered as well
                             (generally found to have hurt performance, hence default set to False)
        :param t1_mode: "start"/"end", whether to pick start/end time for r1 (query) relation
        :param t2_mode: "start"/"end", whether to pick start/end time for r2 (link) relation
        """
        super(Gadget, self).__init__()
        self.load_to_gpu = load_to_gpu
        print("relation_count:", relation_count)
        self.entity_count, self.relation_count = entity_count, relation_count + 1  # dummy relation added to avoid empty neighbour case

        self.sub_facts, self.sub_degree, self.obj_facts, self.obj_degree = self.init_matrices(train_kb, max_neighbours)

        print("Created entity neighbourhood dict")

        # --build time_emb_matrix to retrieve t1/t2 from interval id-- #
        assert (train_kb.datamap.use_time_interval == False)  # no time binning assumed


        time_map = train_kb.datamap.intervalId2dateYears
        self.t1_emb = torch.nn.Embedding(len(time_map), 1)
        self.t1_emb.weight.requires_grad = False

        self.t2_emb = torch.nn.Embedding(len(time_map), 1)
        self.t2_emb.weight.requires_grad = False

        t1_map = numpy.zeros((len(time_map), 1))
        t2_map = numpy.zeros((len(time_map), 1))

        for time_id in time_map:
            start, end = time_map[time_id]

            if isinstance(start, str):  # hack for now, map shouldn't contain strings ideally
                start = 0

            if isinstance(end, str):
                end = 0

            if start <= 0 and end > 0:
                start = end
            elif end == YEARMAX and start > 0:
                end = start

            t_mode_map = {"start": start, "end": end}

            t1_map[time_id] = t_mode_map[t1_mode]
            t2_map[time_id] = t_mode_map[t2_mode]

        self.t1_emb.weight.data.copy_(torch.from_numpy(t1_map))
        self.t2_emb.weight.data.copy_(torch.from_numpy(t2_map))

        # ---Time vector of all possible times for time prediction----------#
        if not train_kb.datamap.use_time_interval:
            self.times = func_load_to_gpu(torch.from_numpy(train_kb.datamap.id2dateYear_mat), load_to_gpu)
        else:
            self.times = func_load_to_gpu(torch.from_numpy(train_kb.datamap.binId2year_mat), load_to_gpu)

        # --------------------------#

        self.eval_batch_size = eval_batch_size
        self.eval_ids = func_load_to_gpu((torch.arange(self.entity_count)).repeat(self.eval_batch_size),
                                         self.load_to_gpu)
        self.eval_tensors = {'subject': self.get_nbors_indices(self.eval_ids, mode='subject'),
                             'object': self.get_nbors_indices(self.eval_ids, mode='object')}
        print("Constructed tensors for eval")

        self.use_obj_scores = use_obj_scores

        if not use_obj_scores:
            print("Not looking at object (e2) neighbourhood")

        self.train_kb = train_kb
        self.t1_map = t1_map
        self.t2_map = t2_map

        self.reg_wt = reg_wt

        # --init scoring gadgets-- #
        self.init_scoring_gadgets()
        # -----------------------  #



    @abstractmethod
    def init_scoring_gadgets(self):
        '''
        Initializes the scoring gadgets
        '''
        pass

    @abstractmethod
    def compute_scores(self, s, r, o, t, mode='subject', positive_samples=False, eval=False, predict_time=False):
        '''
        Returns scores for the given quadruples, using the scoring gadgets
        '''
        pass


    def get_nbors_indices(self, entities, mode='subject', filter=None):
        if mode == 'subject':
            nbor_dict = self.sub_facts
            degrees = self.sub_degree
        else:
            nbor_dict = self.obj_facts
            degrees = self.obj_degree

        nbors_to_filter = None
        if filter is not None:
            r, nbors, t = filter
            nbors_to_filter = torch.cat((r.unsqueeze(1), nbors.unsqueeze(1), t), dim=-1)

        batch_size = len(entities)

        # extract neighbours of the form (r,e,t) from dict,
        # get a list of tensors and stack(concat) them. Size after concat is say N.
        nbor_list = []
        for idx, i in enumerate(entities):
            nbors_of_i = nbor_dict[i.item()]
            if filter is not None:
                # --filter out the query from neighbour set-- #
                x = (nbors_of_i[:, :] != nbors_to_filter[idx])
                x = (x != 0).sum(dim=1).nonzero().squeeze()
                nbors_of_i = nbors_of_i[x]
                if len(nbors_of_i.shape) == 1:
                    nbors_of_i = nbors_of_i.unsqueeze(0)
                # pdb.set_trace()
                # ------------------------------------------ #

            nbor_list.append(nbors_of_i)

        # if filter is not None:
        #     pdb.set_trace()

        entity_nbors = torch.cat(nbor_list, dim=0)

        # create a tensor of indices (index from 0 to b-1), where
        # index i is repeated num_neighbours[i] times (call it 'indices')
        # higher versions of pytorch (>=1.2) have a
        # repeat_interleave method for this
        entity_degrees = degrees[entities.long()]
        if filter is not None:
            entity_degrees -= 1  # positive sample filtered for each entity, hence reduce degree by 1

        indices = numpy.repeat(numpy.arange(batch_size), entity_degrees.cpu())
        indices = func_load_to_gpu(torch.tensor(indices), self.load_to_gpu)

        return entity_nbors, indices

    def forward(self, s, r, o, t):

        if t is None:  # time prediction
            # pdb.set_trace()
            batch_size = len(s)
            num_times = len(self.times)

            s = s.repeat(1, num_times).flatten()
            r = r.repeat(1, num_times).flatten()
            o = o.repeat(1, num_times).flatten()

            # pdb.set_trace()
            t = (self.times).float()
            t = t.repeat(batch_size)

            sub_scores = self.compute_scores(s, r, o, t, mode='subject', predict_time=True)

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, mode='object', predict_time=True)
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape(batch_size, num_times)

        elif s is None:  # scores over all entities
            batch_size = len(o)

            # s = func_load_to_gpu((torch.arange(self.entity_count)).repeat(batch_size), self.load_to_gpu)
            # s = (torch.arange(self.entity_count)).repeat(batch_size)
            s = self.eval_ids[:batch_size * self.entity_count]

            o = func_load_to_gpu(torch.from_numpy(o.cpu().numpy().repeat(self.entity_count)), self.load_to_gpu)
            r = func_load_to_gpu(torch.from_numpy(r.cpu().numpy().repeat(self.entity_count)), self.load_to_gpu)
            t = func_load_to_gpu(torch.from_numpy(t.cpu().numpy().repeat(self.entity_count, axis=0)), self.load_to_gpu)

            # print("s shape:{}, r shape:{}, o shape:{}, t shape:{}".format(s.shape, r.shape, o.shape, t.shape))

            sub_scores = self.compute_scores(s, r, o, t, mode='subject', eval=True)

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, mode='object', eval=True)
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape((batch_size, self.entity_count))

        elif o is None:  # scores over all entities
            batch_size = len(s)

            # o = func_load_to_gpu((torch.arange(self.entity_count)).repeat(batch_size), self.load_to_gpu)
            # o = (torch.arange(self.entity_count)).repeat(batch_size)
            o = self.eval_ids[:batch_size * self.entity_count]

            s = func_load_to_gpu(torch.from_numpy(s.cpu().numpy().repeat(self.entity_count)), self.load_to_gpu)
            r = func_load_to_gpu(torch.from_numpy(r.cpu().numpy().repeat(self.entity_count)), self.load_to_gpu)
            t = func_load_to_gpu(torch.from_numpy(t.cpu().numpy().repeat(self.entity_count, axis=0)), self.load_to_gpu)

            sub_scores = self.compute_scores(s, r, o, t, mode='subject', eval=True)

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, mode='object', eval=True)
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape((batch_size, self.entity_count))

        elif len(s.shape) > 1 and s.shape[1] > 1:  # negative samples for s
            batch_size, num_neg_samples = s.shape
            o = func_load_to_gpu(torch.from_numpy(o.cpu().numpy().repeat(num_neg_samples)), self.load_to_gpu)
            r = func_load_to_gpu(torch.from_numpy(r.cpu().numpy().repeat(num_neg_samples)), self.load_to_gpu)
            t = func_load_to_gpu(torch.from_numpy(t.cpu().numpy().repeat(num_neg_samples, axis=0)), self.load_to_gpu)

            s = torch.flatten(s)

            # print("s:", s)
            # print("o:", o)

            # print("s shape:{}, r shape:{}, o shape:{}, t shape:{}".format(s.shape, r.shape, o.shape, t.shape))

            sub_scores = self.compute_scores(s, r, o, t, mode='subject')

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, mode='object')
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape((batch_size, num_neg_samples))

        elif len(o.shape) > 1 and o.shape[1] > 1:  # negative samples for o
            batch_size, num_neg_samples = o.shape
            s = func_load_to_gpu(torch.from_numpy(s.cpu().numpy().repeat(num_neg_samples)), self.load_to_gpu)
            r = func_load_to_gpu(torch.from_numpy(r.cpu().numpy().repeat(num_neg_samples)), self.load_to_gpu)
            t = func_load_to_gpu(torch.from_numpy(t.cpu().numpy().repeat(num_neg_samples, axis=0)), self.load_to_gpu)

            o = torch.flatten(o)

            sub_scores = self.compute_scores(s, r, o, t, mode='subject')

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, mode='object')
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape((batch_size, num_neg_samples))

        else:  # positive samples for s
            # print("s shape:{}, r shape:{}, o shape:{}, t shape:{}".format(s.shape, r.shape, o.shape, t.shape))
            # pdb.set_trace()

            sub_scores = self.compute_scores(s, r, o, t, mode='subject', positive_samples=True)

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, mode='object', positive_samples=True)
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

        return scores

    def init_matrices(self, train_kb, max_neighbours=None):
        """
        :param train_kb: train kb, needed for facts
        :param max_neighbours: max_neighbours to be considered for an entity (same value for both subject and object)
        :return: sub_facts, sub_degree, obj_facts, obj_degree
        sub_facts is a dict from entity to torch tensor of neighbours (on GPU if needed)
        Here sub_facts[e,i] stores [rel, obj, time] corresponding to i'th neighbour of e (with e as subject).
        Similar for obj_facts ([rel, sub, time] instead)
        """
        sub_facts_dict = defaultdict(list)
        obj_facts_dict = defaultdict(list)

        for fact in train_kb.facts:
            s, r, o = fact[:3]
            t = fact[3:]

            sub_facts_dict[s].append(numpy.concatenate(([r, o], t)))

            obj_facts_dict[o].append(numpy.concatenate(([r, s], t)))

        if max_neighbours is None:
            max_neighbours = 0
            for key in sub_facts_dict:
                max_neighbours = max(max_neighbours, len(sub_facts_dict[key]))
            for key in obj_facts_dict:
                max_neighbours = max(max_neighbours, len(obj_facts_dict[key]))

        print("Max neighbours:{}".format(max_neighbours))

        num_entities = len(train_kb.datamap.entity_map)

        sub_facts, obj_facts = {}, {}
        sub_degree = torch.zeros(num_entities)
        obj_degree = torch.zeros(num_entities)

        for i in range(num_entities):
            # --avoid zero neighbour case by adding a dummy relation --#
            if len(sub_facts_dict[i]) == 0:
                sub_facts_dict[i] = [numpy.array([self.relation_count - 1, i] + [0] * len(time_index))]

            if len(obj_facts_dict[i]) == 0:
                obj_facts_dict[i] = [numpy.array([self.relation_count - 1, i] + [0] * len(time_index))]
            # ------------------------------------------------------- #

            sub_facts[i] = func_load_to_gpu(torch.tensor(sub_facts_dict[i]), self.load_to_gpu)
            obj_facts[i] = func_load_to_gpu(torch.tensor(obj_facts_dict[i]), self.load_to_gpu)

            sub_degree[i] = len(sub_facts_dict[i])
            obj_degree[i] = len(obj_facts_dict[i])

        sub_degree = func_load_to_gpu(sub_degree, self.load_to_gpu)
        obj_degree = func_load_to_gpu(obj_degree, self.load_to_gpu)

        return sub_facts, sub_degree, obj_facts, obj_degree

    def post_epoch(self):
        return ""

    def regularizer(self, s, r, o, t):
        return self.reg_wt*(self.scoring_gadget['subject'].regularizer() + self.scoring_gadget['object'].regularizer())

# ----------------------------------------------------------------------------------- #

class Pairs(Gadget):
    def __init__(self, train_kb, entity_count, relation_count, load_to_gpu=True, 
            eval_batch_size=10, use_obj_scores=False, reg_wt = 0.0, **kwargs):
        
        ## gadget specific args
        self.scoring_gadget_type = kwargs.get('scoring_gadget_type', 'gaussian')
        self.trainable = kwargs.get('trainable', True)
        self.weight_init = kwargs.get('weight_init',0.1)
        self.min_support = kwargs.get('min_support',1)
        ##

        super(Pairs, self).__init__(train_kb, entity_count, relation_count, load_to_gpu=load_to_gpu, eval_batch_size=eval_batch_size, 
                                use_obj_scores=use_obj_scores, reg_wt = reg_wt)


    def init_scoring_gadgets(self):
        if self.scoring_gadget_type not in ['gaussian', 'laplacian']:
            raise Exception("Unknown gadget type {}".format(self.coring_gadget_type))
            
        if self.scoring_gadget_type == 'gaussian' or self.scoring_gadget_type == 'laplacian':
                # --init scorers-- #
                sub_prob_density = ProbDensityScorer(self.train_kb, self.relation_count, self.t1_map, self.t2_map, min_support=self.min_support, trainable=self.trainable,
                                                distribution=self.scoring_gadget_type, mode='subject', load_to_gpu=self.load_to_gpu)
                obj_prob_density = ProbDensityScorer(self.train_kb, self.relation_count, self.t1_map, self.t2_map, min_support=self.min_support, trainable=self.trainable,
                                                distribution=self.scoring_gadget_type, mode='object', load_to_gpu=self.load_to_gpu)
                # ------------------------ #

                self.scoring_gadget = torch.nn.ModuleDict({'subject': sub_prob_density, 'object': obj_prob_density})
                print("Initialized {} scoring gadget for pairwise".format(self.scoring_gadget_type))
        else:
            raise Exception("Unknown gadget type {}".format(self.scoring_gadget_type))
        
        # --weight matrix as function of relation pairs-- #
        self.W_sub = torch.ones(self.relation_count, self.relation_count) * self.weight_init  # for subject
        self.W_obj = torch.ones(self.relation_count, self.relation_count) * self.weight_init  # for object
        # ---------------------------------------------- #

        self.W_sub = torch.nn.Parameter(self.W_sub)
        self.W_obj = torch.nn.Parameter(self.W_obj)


        return

    def compute_scores(self, s, r, o, t, mode='subject', positive_samples=False, eval=False, predict_time=False):
        """
        Computes score for each sample (s,r,o,t are expected to be of same length)
        mode='subject' means look at subjects' neighbourhood only (similar for mode='object').
        positive_samples=True indicates that we are computing scores for positive samples, so
        removing the query fact from neighbour set may be required.
        eval=True indicates that we are computing scores across all entities, so use already constructed tensors
        """
        s = s.squeeze()
        r = r.squeeze()
        o = o.squeeze()
        t = t.squeeze()

        if mode == 'subject':
            # --compute scores for neighbourhood of s--#
            entities = s
            nbors = o
            #U2_scoring_gadget = self.U2_scoring_gadget[mode]
            #pairwise_scoring_gadget = self.pairwise_scoring_gadget[mode]
            weights = self.W_sub

        elif mode == 'object':
            # --compute scores for neighbourhood of o--#
            entities = o
            nbors = s
            #U2_scoring_gadget = self.U2_scoring_gadget[mode]
            #pairwise_scoring_gadget = self.pairwise_scoring_gadget[mode]
            weights = self.W_obj

        else:
            raise Exception('Unknown mode')

        num_samples = len(entities)

        if not eval:
            filter = None
            if positive_samples:
                filter = (r, nbors, t)
                # pass
            entity_nbors, indices = self.get_nbors_indices(entities, mode=mode, filter=filter)
            # if positive_samples:
            #     pdb.set_trace()
        else:
            entity_nbors, indices = self.eval_tensors[mode]  # use pre-constructed tensors

            batch_size = num_samples / self.entity_count
            all_nbors = len(entity_nbors) / self.eval_batch_size

            entity_nbors = entity_nbors[:int(all_nbors * batch_size)]
            indices = indices[:int(all_nbors * batch_size)]


        # use indices to repeat t and r appropriate number of times
        t_repeated = t[indices]  # torch.index_select(t, 0, indices)
        r_repeated = r[indices]  # torch.index_select(r, 0, indices)
        e_repeated = nbors[indices]  # torch.index_select(r, 0, indices)

        # print("t after repeating:{}, r after repeating:{}".format(t_repeated.shape, r_repeated.shape))

        # compute time diff
        # time_idx = time_index["t_s"]  # pick exact year instead of bin?
        # r_time = entity_nbors[:, 2 + time_idx]
        # query_time = t_repeated[:, time_idx]

        time_idx = time_index["t_i"]  # pick start year from time interval id
        r_time = self.t1_emb(entity_nbors[:, 2 + time_idx])

        if not predict_time:
            query_time = self.t2_emb(t_repeated[:, time_idx])
        else:
            query_time = t_repeated.unsqueeze(1)

        # pdb.set_trace()


        # print("r_time shape:{}, query_time:{}".format(r_time.shape, query_time.shape))
        # print(query_time)
        # print(time_diff)

        # we have all features! Now compute scores (N scores)
        # features are- r_query, r_link, diff, entity
        nbor_index = {"r": 0, "e": 1, "t": 2}
        # print("Computing scores")
        r_query = r_repeated
        r_link = entity_nbors[:, nbor_index["r"]]
        e_query = e_repeated
        e_link = entity_nbors[:, nbor_index["e"]]

        pairwise_scoring_gadget = self.scoring_gadget[mode]
        time_diff = r_time - query_time
        time_diff = time_diff.float()  # .unsqueeze_(-1)

        pairwise_scores = pairwise_scoring_gadget(r_query, r_link, time_diff)

        # we have scores, now compute soft attention weights for them.
        wt = weights[r_link, r_query]

        # '''
        # compute softmax
        wt = wt - torch.max(wt)
        wt = torch.exp(wt)

        wt_sum = func_load_to_gpu(torch.zeros(num_samples), self.load_to_gpu)
        wt_sum = wt_sum.index_add(0, indices, wt)
        wt_sum = wt_sum[indices]
        wt = wt / (wt_sum.squeeze())
        # print("Computed weights with softmax")
        # '''
        

        # weights computed, now multiply with scores output from scoring_gadget and compute summation
        final_pairwise_scores = func_load_to_gpu(torch.zeros(num_samples), self.load_to_gpu)
        final_pairwise_scores = final_pairwise_scores.index_add(0, indices, wt * pairwise_scores.squeeze())


        final_scores = final_pairwise_scores

        return final_scores

    # def regularizer(self, s, r, o, t):
    #     return self.reg_wt*(self.scoring_gadget['subject'].regularizer() + self.scoring_gadget['object'].regularizer() + 
    #             self.W_sub**2 + self.W_obj**2)


class Recurrent(Gadget):
    def __init__(self, train_kb, entity_count, relation_count, load_to_gpu=True, 
            eval_batch_size=10, use_obj_scores=False, reg_wt = 0.0, **kwargs):
        
        # pdb.set_trace()

        ## gadget specific args
        self.scoring_gadget_type = kwargs.get('scoring_gadget_type', 'gaussian')
        self.trainable = kwargs.get('trainable', True)
        self.weight_init = kwargs.get('weight_init',0.1)
        self.min_support = kwargs.get('min_support',1)
        self.offset_init  =kwargs.get('offset_init',-0.1)
        ##

        super(Recurrent, self).__init__(train_kb, entity_count, relation_count, load_to_gpu=load_to_gpu, eval_batch_size=eval_batch_size, 
                                use_obj_scores=use_obj_scores, reg_wt=reg_wt)


    def init_scoring_gadgets(self):
        if self.scoring_gadget_type not in ['gaussian', 'laplacian']:
            raise Exception("Unknown gadget type {}".format(self.scoring_gadget_type))
            
        if self.scoring_gadget_type == 'gaussian' or self.scoring_gadget_type == 'laplacian':
                # --init scorers-- #
                sub_prob_density = RecurringFactScorer(self.train_kb, self.relation_count, self.t1_map, min_support=self.min_support, trainable=self.trainable,
                                                distribution=self.scoring_gadget_type, mode='subject', offset_init=self.offset_init,
                                                load_to_gpu=self.load_to_gpu)
                obj_prob_density = RecurringFactScorer(self.train_kb, self.relation_count, self.t1_map, min_support=self.min_support, trainable=self.trainable,
                                                distribution=self.scoring_gadget_type, mode='object', offset_init=self.offset_init,
                                                load_to_gpu=self.load_to_gpu)
                # ------------------------ #

                self.scoring_gadget = torch.nn.ModuleDict({'subject': sub_prob_density, 'object': obj_prob_density})
                print("Initialized {} scoring gadget for recurrent".format(self.scoring_gadget_type))
        else:
            raise Exception("Unknown gadget type {}".format(self.scoring_gadget_type))
        
        return


    def compute_scores(self, s, r, o, t, mode='subject', positive_samples=False, eval=False, predict_time=False):
        """
        Computes score for each sample (s,r,o,t are expected to be of same length)
        mode='subject' means look at subjects' neighbourhood only (similar for mode='object').
        positive_samples=True indicates that we are computing scores for positive samples, so
        removing the query fact from neighbour set may be required.
        eval=True indicates that we are computing scores across all entities, so use already constructed tensors
        """
        s = s.squeeze()
        r = r.squeeze()
        o = o.squeeze()
        t = t.squeeze()

        if mode == 'subject':
            # --compute scores for neighbourhood of s--#
            entities = s
            nbors = o

        elif mode == 'object':
            # --compute scores for neighbourhood of o--#
            entities = o
            nbors = s

        else:
            raise Exception('Unknown mode')

        scoring_gadget = self.scoring_gadget[mode]


        num_samples = len(entities)

        if not eval:
            filter = None
            if positive_samples:
                filter = (r, nbors, t)
                # pass
            entity_nbors, indices = self.get_nbors_indices(entities, mode=mode, filter=filter)
            # if positive_samples:
            #     pdb.set_trace()
        else:
            entity_nbors, indices = self.eval_tensors[mode]  # use pre-constructed tensors

            batch_size = num_samples / self.entity_count
            all_nbors = len(entity_nbors) / self.eval_batch_size

            entity_nbors = entity_nbors[:int(all_nbors * batch_size)]
            indices = indices[:int(all_nbors * batch_size)]


        # use indices to repeat t and r appropriate number of times
        indices = indices.long()
        t_repeated = t[indices]  # torch.index_select(t, 0, indices)
        r_repeated = r[indices]  # torch.index_select(r, 0, indices)
        e_repeated = nbors[indices]  # torch.index_select(r, 0, indices)

        # print("t after repeating:{}, r after repeating:{}".format(t_repeated.shape, r_repeated.shape))

        # compute time diff
        # time_idx = time_index["t_s"]  # pick exact year instead of bin?
        # r_time = entity_nbors[:, 2 + time_idx]
        # query_time = t_repeated[:, time_idx]

        time_idx = time_index["t_i"]  # pick start year from time interval id
        r_time = self.t1_emb(entity_nbors[:, 2 + time_idx])

        if not predict_time:
            query_time = self.t2_emb(t_repeated[:, time_idx])
        else:
            query_time = t_repeated.unsqueeze(1)

        # pdb.set_trace()

        time_diff = r_time - query_time

        # print("r_time shape:{}, query_time:{}".format(r_time.shape, query_time.shape))
        # print(query_time)
        # print(time_diff)

        # we have all features! Now compute scores (N scores)
        # features are- r_query, r_link, diff, entity
        nbor_index = {"r": 0, "e": 1, "t": 2}
        # print("Computing scores")
        r_query = r_repeated
        r_link = entity_nbors[:, nbor_index["r"]]
        e_query = e_repeated
        e_link = entity_nbors[:, nbor_index["e"]]

        time_diff = time_diff.float()  # .unsqueeze_(-1)

        # 1 if same r,e seen at different time (time_diff is non-zero)

        # if self.gadget_type == 'recurring-fact':
        repeated_fact = (r_link == r_query) & (e_link == e_query) & (time_diff != 0).squeeze()
        # elif self.gadget_type == 'recurring-relation':
        #     repeated_fact = (r_link == r_query) & (time_diff != 0).squeeze()

        repeated_fact = repeated_fact.float()

        # pdb.set_trace()

        # compress query_fact into binary tensor of length num_samples,
        # with 1 for the sample which has at least one repeated fact.
        # use index_add for this. call it eligible_samples
        eligible_samples = func_load_to_gpu(torch.zeros(num_samples), self.load_to_gpu)
        eligible_samples = eligible_samples.index_add(0, indices, repeated_fact)
        eligible_samples[eligible_samples != 0] = 1

        # find smallest absolute time_diff for each sample (closest repeated fact)
        # call it smallest_time_diff
        smallest_time_diff = numpy.ones(num_samples)*1e5
        time_diff = torch.abs(time_diff).squeeze()

        non_repeated = (repeated_fact == 0).nonzero().squeeze()
        time_diff.scatter_(0, non_repeated, 1e5) # so that non-eligible facts don't interfere with the min operation

        time_diff_np = time_diff.cpu().numpy() if self.load_to_gpu else time_diff.numpy()
        indices_np = indices.cpu().numpy() if self.load_to_gpu else indices.numpy()
        numpy.minimum.at(smallest_time_diff, indices_np, time_diff_np) # pytorch doesn't have a function for this yet

        smallest_time_diff = func_load_to_gpu(torch.tensor(smallest_time_diff).float(), self.load_to_gpu) # this takes time if self.load_to_gpu is True! 
                                                                                                          # find an alternative!

        # compute scores for smallest_time_diff
        scores = scoring_gadget(r, smallest_time_diff)


        # multiply scores with eligible_sample, to give zero scores
        # for non-eligible samples (i.e. for which entities have not been seen with same (r,e)
        final_scores = scores * eligible_samples


        # ---------------------------------------#

        return final_scores







