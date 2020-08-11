import sys

sys.path.append('../')

import numpy
import torch

from collections import defaultdict

import pickle
import pdb

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


# ---general helper functions----#
def get_bounds(data, population=50):
    margin = (100.0 - population) / 2.0
    min_bound = numpy.percentile(data, margin)
    max_bound = numpy.percentile(data, population + margin)
    return min_bound, max_bound, len(data)


def get_r_r_stat(r_r_dict, population=50):
    r_r_stat = {}
    for r1 in r_r_dict.keys():
        for r2 in r_r_dict[r1].keys():
            r_r_stat[r1, r2] = get_bounds(r_r_dict[r1][r2], population=population)
    return r_r_stat


def check_date_validity(date):
    start = date.split('-')[0]
    if start.find('#') == -1 and len(start) == 4:
        return int(start)
    else:
        return -1


def get_pairwise_r_dict(ent_rel_dict_t1, ent_rel_dict_t2):
    """
    Given a dict - dic[entity][relation][time], which contains details of all relations
    and time a given set of entities (when in sub/obj position) are seen with
    Returns:
    rxr dict: containg list of all time differences of r1 and r1
    """
    r_r_dict = {}
    for entity in ent_rel_dict_t1:
        for r1 in ent_rel_dict_t1[entity]:
            for r2 in ent_rel_dict_t2[entity]:
                for r1_t in list(ent_rel_dict_t1[entity][r1]):  # t1 time for r1 (can be start/end)
                    for r2_t in list(ent_rel_dict_t2[entity][r2]):  # t2 time for r2 (can be start/end)
                        if not r1 in r_r_dict.keys():
                            r_r_dict[r1] = {}
                        if not r2 in r_r_dict[r1].keys():
                            r_r_dict[r1][r2] = []
                        r_r_dict[r1][r2].append(int(r1_t) - int(r2_t))

    return r_r_dict


def func_load_to_gpu(data, load_to_gpu):
    if load_to_gpu:
        data = data.cuda()
    return data


##########################

# --helper functions for StepFunctionScorer --#
def get_bounds(data, population=50):
    margin = (100.0 - population) / 2.0
    min_bound = numpy.percentile(data, margin)
    max_bound = numpy.percentile(data, population + margin)
    return min_bound, max_bound, len(data)


def get_r_r_stat(r_r_dict, population=50):
    r_r_stat = {}
    for r1 in r_r_dict.keys():
        for r2 in r_r_dict[r1].keys():
            r_r_stat[r1, r2] = get_bounds(r_r_dict[r1][r2], population=population)
    return r_r_stat

def min_max_thresholds(facts, t1_map, t2_map, num_relations, population=90, min_support=10, mode='subject'):
    """
	Input:
	facts:[n x 4], where n is the number of training facts. Each row contains s,r,o,t ids.
	time_map: map from t id to time string
	population: what percentage of data should the 2 thresholds (min & max) cover
	min_support: minimum support for a relation pair to be considered
	mode: 'subject' for subject thresholds, 'object' for object thresholds
    Returns:
	min_r_r:[r x r] where r is the number of relations, min_sub[i,j] denotes the minimum allowed value for time difference of i
			and j (e.i.t- e.j.t)
	max_r_r:[r x r] where r is the number of relations, max_sub[i,j] denotes the maximum allowed value for time difference of i
			and j (e.i.t- e.j.t)
	"""
    t1_map = t1_map.squeeze()
    t2_map = t2_map.squeeze()

    rel_dict_t1 = defaultdict(lambda: defaultdict(set))
    rel_dict_t2 = defaultdict(lambda: defaultdict(set))

    for fact in facts:
        s, r, o = fact[:3]
        t = fact[3:]
        # date = check_date_validity(time_mat[t["t_i"]])
        # if date >= 0:

        date_t1 = t1_map[t[time_index["t_i"]]]
        date_t2 = t2_map[t[time_index["t_i"]]]

        if mode == 'subject':
            rel_dict_t1[s][r].add(date_t1)
            rel_dict_t2[s][r].add(date_t2)

        elif mode == 'object':
            rel_dict_t1[o][r].add(date_t1)
            rel_dict_t2[o][r].add(date_t2)

        else:
            raise Exception("Unknown mode!")

    r_r_dict = get_pairwise_r_dict(rel_dict_t1, rel_dict_t2)

    r_r_stat = get_r_r_stat(r_r_dict, population=population)

    inf = 100000
    min_r_r = torch.zeros(num_relations, num_relations)
    max_r_r = torch.zeros(num_relations, num_relations)

    for i in range(num_relations):
        for j in range(num_relations):
            minn, maxx, sup = r_r_stat.get((i, j), (-inf, inf, 0))
            # minn, maxx, sup=r_r_stat.get((i,j),(inf, -inf,0))

            if sup >= min_support:
                # print("r1:{}, r2:{}, min:{}, max:{}".format(i,j,minn,maxx))
                min_r_r[i, j] = minn
                max_r_r[i, j] = maxx
            else:
                # min_r_r[i,j]=inf
                # max_r_r[i,j]=-inf
                min_r_r[i, j] = -inf
                max_r_r[i, j] = inf

    return min_r_r, max_r_r
# ------------------------------------------- #

# ---helper functions for ProbDensityScorer-- #
def mean_variance(facts, t1_map, t2_map, num_relations, min_support=10, mode='subject', mask = None):
    '''
    Input:
    facts:[n x 4], where n is the number of training facts. Each row contains s,r,o,t ids.
    t1_mat: array storing t1 for each t["t_i"]
    t2_mat: array storing t2 for each t["t_i"]
    (NOTE: Time diff for r1 and r2 is calculated as r1.t1 - r2.t2)
    min_support: minimum support for a relation pair to be considered
    sub_obj: 0 for sub thresholds, 1 for obj thresholds
    Returns:
    mean_r_r:[r x r] where r is the number of relations, mean_r_r[i,j] denotes the mean time difference of i
            and j (e.i.t- e.j.t)
    var_r_r:[r x r] where r is the number of relations, var_r_r[i,j] denotes the variance in time difference of i
            and j (e.i.t- e.j.t)
    '''
    # time_mat = time_mat.squeeze()
    t1_map = t1_map.squeeze()
    t2_map = t2_map.squeeze()

    rel_dict_t1 = defaultdict(lambda: defaultdict(set))
    rel_dict_t2 = defaultdict(lambda: defaultdict(set))

    for fact in facts:
        s, r, o = fact[:3]
        t = fact[3:]
        # date = check_date_validity(time_mat[t["t_i"]])
        # if date >= 0:

        date_t1 = t1_map[t[time_index["t_i"]]]
        date_t2 = t2_map[t[time_index["t_i"]]]

        if mode == 'subject':
            rel_dict_t1[s][r].add(date_t1)
            rel_dict_t2[s][r].add(date_t2)

        elif mode == 'object':
            rel_dict_t1[o][r].add(date_t1)
            rel_dict_t2[o][r].add(date_t2)

        else:
            raise Exception("Unknown mode!")

    r_r_dict = get_pairwise_r_dict(rel_dict_t1, rel_dict_t2)

    r_r_stat = {}
    for r1 in r_r_dict.keys():
        for r2 in r_r_dict[r1].keys():
            data = r_r_dict[r1][r2]
            r_r_stat[r1, r2] = (numpy.mean(data), numpy.var(data), len(data))

    inf = 1000
    min_var = 0.01

    mean_r_r = torch.zeros(num_relations, num_relations)
    var_r_r = torch.zeros(num_relations, num_relations)

    for i in range(num_relations):
        for j in range(num_relations):
            mean, var, sup = r_r_stat.get((i, j), (-inf, min_var, 0))

            var = max(var, min_var)
            # minn, maxx, sup=r_r_stat.get((i,j),(inf, -inf,0))

            if sup >= min_support:
                mean_r_r[i, j] = mean
                var_r_r[i, j] = var
            else:
                # min_r_r[i,j]=inf
                # max_r_r[i,j]=-inf

                mean_r_r[i, j] = -inf
                var_r_r[i, j] = min_var

            if mask is not None:
                if  mean_r_r[i,j] == -inf and var_r_r[i,j] == min_var:
                    mask[i,j] = 0

    return mean_r_r, var_r_r


# -------------------------------------- #

# ---helper functions for RecurringFactScorer-- #
def recurring_mean_variance(facts, t_map, num_relations, min_support=10, mode='subject'):
    '''
    Input:
    facts:[n x 4], where n is the number of training facts. Each row contains s,r,o,t ids.
    t1_mat: array storing t1 for each t["t_i"]
    t2_mat: array storing t2 for each t["t_i"]
    (NOTE: Time diff for r1 and r2 is calculated as r1.t1 - r2.t2)
    min_support: minimum support for a relation pair to be considered
    sub_obj: 0 for sub thresholds, 1 for obj thresholds
    Returns:
    mean_r_r:[r x r] where r is the number of relations, mean_r_r[i,j] denotes the mean time difference of i
            and j (e.i.t- e.j.t)
    var_r_r:[r x r] where r is the number of relations, var_r_r[i,j] denotes the variance in time difference of i
            and j (e.i.t- e.j.t)
    '''

    t_map = t_map.squeeze()

    rel_t_dict = defaultdict(lambda: defaultdict(list))

    for fact in facts:
        s, r, o = fact[:3]
        t = fact[3:]
        # date = check_date_validity(time_mat[t["t_i"]])
        # if date >= 0:

        date = t_map[t[time_index["t_i"]]]

        rel_t_dict[r][(s, o)].append(date)

    r_diffs = defaultdict(list)

    for r in rel_t_dict:
        for so in rel_t_dict[r]:
            times_list = sorted(rel_t_dict[r][so])
            if len(times_list) > 1:
                for idx, time in enumerate(times_list[1:]):
                    r_diffs[r].append(time - times_list[idx - 1])


    r_stat = {}
    for r in r_diffs:
        data = r_diffs[r]
        r_stat[r] = (numpy.mean(data), numpy.var(data), len(data))

    print("r_stats:", r_stat)
    # pdb.set_trace()

    mean_r = torch.zeros(num_relations)
    var_r = torch.zeros(num_relations)

    inf = 1000

    for i in range(num_relations):
        mean, var, sup = r_stat.get(i, (-inf, 0.01, 0))

        var = max(var, 0.01)
        # minn, maxx, sup=r_r_stat.get((i,j),(inf, -inf,0))

        if sup >= min_support:
            mean_r[i] = mean
            var_r[i] = var
        else:
            # min_r_r[i,j]=inf
            # max_r_r[i,j]=-inf

            mean_r[i] = -inf
            var_r[i] = 0.01

    return mean_r, var_r
# -------------------------------------- #


# ---helper functions for RecurringFactScorer-- #
def recurring_relation_mean_variance(facts, t_map, num_relations, min_support=10, mode='subject'):
    '''
    Input:
    facts:[n x 4], where n is the number of training facts. Each row contains s,r,o,t ids.
    t1_mat: array storing t1 for each t["t_i"]
    t2_mat: array storing t2 for each t["t_i"]
    (NOTE: Time diff for r1 and r2 is calculated as r1.t1 - r2.t2)
    min_support: minimum support for a relation pair to be considered
    sub_obj: 0 for sub thresholds, 1 for obj thresholds
    Returns:
    mean_r_r:[r x r] where r is the number of relations, mean_r_r[i,j] denotes the mean time difference of i
            and j (e.i.t- e.j.t)
    var_r_r:[r x r] where r is the number of relations, var_r_r[i,j] denotes the variance in time difference of i
            and j (e.i.t- e.j.t)
    '''

    t_map = t_map.squeeze()


    rel_t_dict = defaultdict(lambda: defaultdict(list))

    # for fact in facts:
    for fact in facts:
        s, r, o = fact[:3]
        t = fact[3:]
        # date = check_date_validity(time_mat[t["t_i"]])
        # if date >= 0:

        date = t_map[t[time_index["t_i"]]]

        if mode=='subject':
            rel_t_dict[r][s].append(date)
        elif mode=='object':
            rel_t_dict[r][o].append(date)


    r_diffs = defaultdict(list)

    for r in rel_t_dict:
        for e in rel_t_dict[r]:
            times_list = sorted(rel_t_dict[r][e])
            if len(times_list) > 1:
                for idx, time in enumerate(times_list[1:]):
                    r_diffs[r].append(time - times_list[idx - 1])

    r_stat = {}
    for r in r_diffs:
        data = r_diffs[r]
        # if len(data) > 10:
        r_stat[r] = (numpy.mean(data), numpy.var(data), len(data))

    print("r_stats:", r_stat)

    mean_r = torch.zeros(num_relations)
    var_r = torch.zeros(num_relations)

    inf = 1000

    for i in range(num_relations):
        mean, var, sup = r_stat.get(i, (-inf, 0.01, 0))

        var = max(var, 0.01)
        # minn, maxx, sup=r_r_stat.get((i,j),(inf, -inf,0))

        if sup >= min_support:
            mean_r[i] = mean
            var_r[i] = var
        else:
            # min_r_r[i,j]=inf
            # max_r_r[i,j]=-inf

            mean_r[i] = -inf
            var_r[i] = 0.01

    return mean_r, var_r
# -------------------------------------- #

