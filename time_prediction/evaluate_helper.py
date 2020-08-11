"""
Helper methods used by evaluate
"""
import sys

sys.path.append('../')

import torch
import numpy
import pickle
from collections import defaultdict

from time_prediction.interval_prediction_methods import *
from time_prediction.interval_metrics import smooth_iou_score, aeiou_score, tac_score, giou_score

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}

YEAR_MAX = 3000
YEAR_MIN = 0


def func_load_to_gpu(data, load_to_gpu):
    if load_to_gpu:
        data = data.cuda()
    return data


def stack_tensor_list(l):
    return torch.stack(l).cpu()


def string_to_year(s, default=YEAR_MAX):
    if s.find('#') == -1:
        return int(s)
    return default


def get_gold_year_interval(t, id2_time_str, id_year_map, load_to_gpu=True):
    assert (t.shape[-1] == len(time_index))
    t_str = t[:, :, time_index["t_str"]]

    # pdb.set_trace()

    t_gold_min = []
    t_gold_max = []
    # ipdb.set_trace()
    for ele in t_str:
        t_gold_min_ele, t_gold_max_ele = id2_time_str[ele[0].item()].split(
            "\t")
        t_gold_min_ele = t_gold_min_ele.split("-")[0]
        t_gold_max_ele = t_gold_max_ele.split("-")[0]

        t_gold_min.append(t_gold_min_ele)
        t_gold_max.append(t_gold_max_ele)

    return t_gold_min, t_gold_max


def prepare_data_iou_scores(t, test_kb, scores_t="", load_to_gpu=True):
    ##
    # pdb.set_trace()
    if not test_kb.datamap.use_time_interval:
        id_year_map = func_load_to_gpu(torch.from_numpy(
            test_kb.datamap.id2dateYear_mat), load_to_gpu)
    else:
        id_year_map = func_load_to_gpu(torch.from_numpy(
            test_kb.datamap.binId2year_mat), load_to_gpu)
    ##

    id2_time_str = test_kb.datamap.id2TimeStr

    # ---final IOU score computation, extract gold time intervals from KB first---#

    t_gold_min, t_gold_max = get_gold_year_interval(
        t, id2_time_str, id_year_map, load_to_gpu=load_to_gpu)
    # ------------------------------------------#

    out_dict = {"scores_t": scores_t, "gold": (t_gold_min, t_gold_max), "map": id_year_map,
                "use_time_interval": test_kb.datamap.use_time_interval,
                "facts": test_kb.facts, "data_folder_full_path": test_kb.datamap.dataset_root}

    return out_dict


def load_pickle(scores_t_dict):
    d=scores_t_dict

    # print(d.keys())
    t_scores = d['scores_t']
    facts = d['facts']
    id_year_map = d['map']
    t_gold_min, t_gold_max = d['gold']

    duration_scores = d.get('duration_scores', None)

    t_gold_min = torch.Tensor([string_to_year(i, default=YEAR_MIN) for i in t_gold_min])
    t_gold_max = torch.Tensor([string_to_year(i, default=YEAR_MAX) for i in t_gold_max])

    t_gold_min = t_gold_min.long()
    t_gold_max = t_gold_max.long()

    use_time_interval = d['use_time_interval']

    default_dataset_root = '../data/YAGO11k/'  # default data path
    dataset_root = d.get('data_folder_full_path', default_dataset_root)

    print("Gold times shape- start: {}, end: {}".format(t_gold_min.shape, t_gold_max.shape))
    print("dataset_root:{}".format(dataset_root))
    # print("Loaded t_scores and facts from {}".format(scores_t_file))

    # t_scores=t_scores.cpu().float()
    if duration_scores is not None:
        duration_scores = duration_scores.cpu().float()

    return t_scores, duration_scores, facts, dataset_root, use_time_interval, id_year_map, t_gold_min, t_gold_max


def relation_iou_scores(method, ktrain, id2rel, use_time_interval, facts, t_scores, duration_scores=None):
    # Analysis of iou scores (across relations, and durations)
    for rel in set(id2rel.values()):
        r1 = 0
        for r in id2rel:
            if id2rel[r] == rel:
                r1 = ktrain.relation_map[r]

        print("\nRelation: {}:{}".format(r1, rel))

        # filter facts based on relation (r1)
        rel_fact_indices = (facts[:, 1] == r1).nonzero()[0]
        rel_facts = facts[rel_fact_indices]

        if len(rel_facts) == 0:
            continue

        rel_t_scores = t_scores[rel_fact_indices, :]
        rel_duration_scores = duration_scores[rel_fact_indices]

        score_func = {"aeIOU": aeiou_score, "IOU": smooth_iou_score}

        score_to_compute = "aeIOU"

        print("\n\nMethod:{}".format(method))

        iou_scores = compute_scores(ktrain, rel_facts, rel_t_scores, method=method, durations=rel_duration_scores,
                                    score_func=score_func[score_to_compute], use_time_interval=use_time_interval,
                                    topk_ranks=10)

        # output best iou @ k
        iouatk = [1, 5, 10]

        for i in iouatk:
            all_scores = torch.stack(iou_scores[:i])
            best_scores, _ = torch.max(all_scores, 0)
            print("Best {} @{}: {}".format(score_to_compute, i, torch.mean(best_scores)))


def get_thresholds(scores_t, valid_facts, test_facts, aggr='mean', verbose=False):
    rel_prob_list = defaultdict(list)

    rel_interval_len = defaultdict(list)

    probs = torch.nn.functional.softmax(scores_t, dim=-1)
    # print("Probabilities shape:", probs.shape)

    for idx, fact in enumerate(valid_facts):
        s, r, o = fact[:3]
        t = fact[3:]

        start, end = t[time_index["t_s_orig"]], t[time_index["t_e_orig"]]

        prob_sum = 0.0
        for i in range(start, end + 1):
            prob_sum += probs[idx, i]

        rel_prob_list[r].append(float(prob_sum))
        rel_interval_len[r].append(end - start + 1)

    print(valid_facts.shape)

    # print("Num relations:", len(rel_prob_list))

    rel_thresh = {}

    for key, val in rel_prob_list.items():
        # rel_thresh[key]=numpy.median(numpy.array(val))
        if aggr == 'mean':
            rel_thresh[key] = numpy.mean(numpy.array(val))
        elif aggr == 'median':
            rel_thresh[key] = numpy.median(numpy.array(val))
        else:
            raise Exception('Unknown aggregate {} for thresholds'.format(aggr))

    '''
    if(verbose):
        print("\nRelations thresholds:")
        for key,val in rel_thresh.items():
            print(key,val)
            print(id2rel[ktrain.reverse_relation_map[key]])
            print(numpy.mean(numpy.array(rel_interval_len[key])), numpy.std(numpy.array(rel_interval_len[key])), numpy.median(numpy.array(rel_interval_len[key])))
            print("Freq:",len(rel_interval_len[key]))
    
            print('\n')
    '''

    thresholds = torch.zeros(len(test_facts))

    thresh_list = [i for _, i in rel_thresh.items()]
    mean_thresh = sum(thresh_list) / len(thresh_list)
    # print("Mean threshold:{}\n".format(mean_thresh))

    for idx, fact in enumerate(test_facts):
        r = fact[1]
        if r in rel_thresh:
            thresholds[idx] = rel_thresh[r]
        else:
            # print("{} relation not in dict".format(r))
            thresholds[idx] = mean_thresh

    return thresholds


def compute_scores(id_year_map, facts, t_gold_min, t_gold_max, scores_t, method=None, score_func=None,
                   thresholds=None, durations=None, topk_ranks=10):
    gold_durations = []
    for ele in facts:
        t1 = ele[3 + time_index["t_s_orig"]]
        t2 = ele[3 + time_index["t_e_orig"]]
        gold_durations.append(t2 - t1)
    gold_durations = torch.tensor(gold_durations)

    print("min duration:{}, max duration:{}".format(torch.min(gold_durations), torch.max(gold_durations)))

    if method == 'greedy-coalescing':
        # compute thresholds using val set, and do greedy coalescing
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        pred_min, pred_max = greedy_coalescing(probs, thresholds, k=topk_ranks)

    elif method == 'greedy-coalescing-duration':
        # greedy coalescing with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = greedy_coalescing_durations(probs, gold_durations, k=topk_ranks)

    elif method == 'duration-gold-greedy':
        # predictions with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_scores(probs, gold_durations, k=topk_ranks)

    elif method == 'duration-gold-exhaustive':
        # exhaustive sweep with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_exhaustive_sweep(probs, gold_durations, k=topk_ranks)
    elif method == 'duration-predicted-greedy':
        # predictions with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_scores(probs, durations, k=topk_ranks)

    elif method == 'duration-predicted-exhaustive':
        # exhaustive sweep with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_exhaustive_sweep(probs, durations, k=topk_ranks)
    elif method == 'start-end-exhaustive-sweep':
        # exhaustive sweep with start and end scores
        start_scores, end_scores = scores_t

        print("start_scores:{}, end_scores:{}".format(start_scores.shape, end_scores.shape))
        # xx=input()
        start_scores = start_scores.cpu()
        end_scores = end_scores.cpu()

        start_probs = torch.nn.functional.softmax(start_scores, dim=-1)
        end_probs = torch.nn.functional.softmax(end_scores, dim=-1)

        pred_min, pred_max = start_end_exhaustive_sweep(start_probs, end_probs, k=topk_ranks)

    else:
        print("Unknown method- {}".format(method))
        raise Exception

    t_pred_min, t_pred_max = (id_year_map[pred_min.long()].float(), id_year_map[pred_max.long()].float())
    t_pred_min, t_pred_max = (t_pred_min.float(), t_pred_max.float())

    t_gold_min = t_gold_min.float()
    t_gold_max = t_gold_max.float()

    # load to cpu
    t_gold_min, t_gold_max = (t_gold_min.cpu(), t_gold_max.cpu())
    t_pred_min, t_pred_max = (t_pred_min.cpu(), t_pred_max.cpu())

    # -------------------------------------#

    # we have t_gold_min,t_gold_max,t_pred_min,t_pred_max- now compute IOU/aeIOU scores for each rank
    iou_scores = []
    for i in range(topk_ranks):
        # pdb.set_trace()
        iou_score = score_func(t_pred_min[:, i], t_pred_max[:, i], t_gold_min, t_gold_max, delta=0)
        #     print(torch.mean(iou_score))
        iou_scores.append(iou_score)
    #     iou_scores.append(float(torch.mean(iou_score)))

    return iou_scores


def compute_preds(id_year_map, facts, t_gold_min, t_gold_max, scores_t, method=None, score_func=None, thresholds=None,
                  durations=None, topk_ranks=10):
    gold_durations = []
    for ele in facts:
        t1 = ele[3 + time_index["t_s_orig"]]
        t2 = ele[3 + time_index["t_e_orig"]]
        gold_durations.append(t2 - t1)
    gold_durations = torch.tensor(gold_durations)

    print("min duration:{}, max duration:{}".format(torch.min(gold_durations), torch.max(gold_durations)))

    if method == 'greedy-coalescing':
        # compute thresholds using val set, and do greedy coalescing
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        pred_min, pred_max = greedy_coalescing(probs, thresholds, k=topk_ranks)

    elif method == 'greedy-coalescing-duration':
        # greedy coalescing with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = greedy_coalescing_durations(probs, gold_durations, k=topk_ranks)

    elif method == 'duration-gold-greedy':
        # predictions with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_scores(probs, gold_durations, k=topk_ranks)

    elif method == 'duration-gold-exhaustive':
        # exhaustive sweep with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_exhaustive_sweep(probs, gold_durations, k=topk_ranks)
    elif method == 'duration-predicted-greedy':
        # predictions with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_scores(probs, durations, k=topk_ranks)

    elif method == 'duration-predicted-exhaustive':
        # exhaustive sweep with gold duration
        probs = torch.nn.functional.softmax(scores_t, dim=-1)
        # probs=scores_t
        pred_min, pred_max = duration_exhaustive_sweep(probs, durations, k=topk_ranks)
    elif method == 'start-end-exhaustive-sweep':
        # exhaustive sweep with start and end scores
        start_scores, end_scores = scores_t

        # print("start_scores:{}, end_scores:{}".format(start_scores.shape, end_scores.shape))
        # xx=input()
        start_scores = start_scores.cpu()
        end_scores = end_scores.cpu()

        start_probs = torch.nn.functional.softmax(start_scores, dim=-1)
        end_probs = torch.nn.functional.softmax(end_scores, dim=-1)

        pred_min, pred_max = start_end_exhaustive_sweep(start_probs, end_probs, k=topk_ranks)

    else:
        print("Unknown method- {}".format(method))
        raise Exception

    t_pred_min, t_pred_max = (id_year_map[pred_min.long()].float(), id_year_map[pred_max.long()].float())
    t_pred_min, t_pred_max = (t_pred_min.float(), t_pred_max.float())

    t_gold_min = t_gold_min.float()
    t_gold_max = t_gold_max.float()

    # load to cpu
    t_gold_min, t_gold_max = (t_gold_min.cpu(), t_gold_max.cpu())
    t_pred_min, t_pred_max = (t_pred_min.cpu(), t_pred_max.cpu())

    # -------------------------------------#

    return t_gold_min, t_gold_max, t_pred_min, t_pred_max
