import pdb
import sys

sys.path.append('../')

import pickle
import torch
import argparse
import numpy

from time_prediction.evaluate_helper import compute_scores, compute_preds, \
    stack_tensor_list, get_thresholds, load_pickle, prepare_data_iou_scores

from time_prediction.interval_metrics import smooth_iou_score, aeiou_score, tac_score, giou_score, precision_score, \
    recall_score

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


def compute_interval_scores(valid_time_scores_dict, test_time_scores_dict, save_time_results=None,
                            method='greedy-coalescing'):
    """
    Takes input time scores stored in test_pickle and valid_pickle (for test and valid KBs respectively)
    Using these time scores, depending on method it predicts intervals for each fact in test kb (a ranking ideally instead of a single interval),
    and returns gIOU/aeIOU/IOU scores @1/5/10
    """
    thresholds = None

    # load from dict
    t_scores, duration_scores, facts, dataset_root, use_time_interval, id_year_map, t_gold_min, t_gold_max = load_pickle(
        test_time_scores_dict)

    print("Using method {}".format(method))

    if method == 'greedy-coalescing':

        valid_t_scores, _, valid_facts, _, _, _, _, _ = load_pickle(valid_time_scores_dict)

        t_scores = stack_tensor_list(t_scores)
        valid_t_scores = stack_tensor_list(valid_t_scores)

        aggr = 'mean'
        thresholds = get_thresholds(valid_t_scores, valid_facts, facts, aggr=aggr)
        print("Computed thresholds (aggr= {})\n".format(aggr))

    elif method in ['start-end-exhaustive-sweep']:  # for time boundary models
        start_t_scores, end_t_scores = t_scores

        start_t_scores = stack_tensor_list(start_t_scores)
        end_t_scores = stack_tensor_list(end_t_scores)

        t_scores = (start_t_scores, end_t_scores)

    # load valid scores

    # compute thresholds

    id_year_map = id_year_map.long()

    id_year_map_dict = {}
    for i, j in enumerate(id_year_map):
        id_year_map_dict[i] = j
    # print(id_year_map_dict)
    # for i, j in zip(t_gold_min[:5], t_gold_max[:5]):
    #     print("gold start:{}, gold end:{}".format(i, j))

    # ----------------------#
    print("**************")
    topk_ranks = 10

    # score_func = {"precision":precision_score, "recall":recall_score, "aeIOU": aeiou_score, "TAC": tac_score, "IOU": smooth_iou_score, "gIOU": giou_score}
    score_func = {"gIOU": giou_score, "aeIOU": aeiou_score, "TAC": tac_score}

    scores_dict = {}  # for saving later

    for score_to_compute in score_func.keys():
        print("\nScore:{}".format(score_to_compute))
        # iou_scores= compute_scores(ktrain, facts, t_scores, method=method, durations=durations,
        # 			  score_func=score_func[score_to_compute], use_time_interval=use_time_interval, topk_ranks=topk_ranks)
        iou_scores = compute_scores(id_year_map, facts, t_gold_min, t_gold_max, t_scores, method=method,
                                    thresholds=thresholds,
                                    score_func=score_func[score_to_compute], topk_ranks=topk_ranks)

        # output best iou @ k
        iouatk = [1, 5, 10]

        for i in iouatk:
            all_scores = torch.stack(iou_scores[:i])
            #     print("all_scores shape:",all_scores.shape)
            best_scores, _ = torch.max(all_scores, 0)

            scores_dict[(i, score_to_compute)] = best_scores

            #     print("best_scores shape:",best_scores.shape)
            print("Best {} @{}: {}".format(score_to_compute, i, torch.mean(best_scores)))
    # '''

    if save_time_results is not None:
        # saves metrics of the form
        pickle_filename = "{}_time_scores_analysis".format(save_time_results)

        gold_min, gold_max, pred_min, pred_max = compute_preds(id_year_map, facts, t_gold_min, t_gold_max, t_scores,
                                                               method=method, thresholds=thresholds,
                                                               score_func=score_func[score_to_compute],
                                                               topk_ranks=topk_ranks)

        with open(pickle_filename, 'wb') as handle:
            pickle.dump({'facts': facts, 'scores_dict': scores_dict, 'time_scores':t_scores, 't_gold_min': gold_min, 't_gold_max': gold_max,
                         't_pred_min': pred_min, 't_pred_max': pred_max}, handle)
            print("\nPickled scores, t_gold_min, t_gold_max, t_pred_min, t_pred_max to {}\n".format(pickle_filename))

    else:
        print("\nNot saving scores")


def get_time_scores(scoring_function, test_kb, method='greedy-coalescing', load_to_gpu=True):
    """
    Returns dict containing time scores for each fact in test_kb (along with some other useful stuff needed later)
    For time-point models, this means scores for each possible time point (t scores for each fact).
    For time-boundary models (not implemented yet), this would mean t start scores and t end scores for each fact.
    """
    facts = test_kb.facts

    if method in ['greedy-coalescing']:  # for time-point models

        scores_t_list = []

        for i in range(0, int(facts.shape[0]), 1):
            fact = facts[i]

            s, r, o = fact[:3]

            start_bin = fact[3 + time_index["t_s_orig"]]

            # start_bin, end_bin=fact[3:5]

            # num_times=end_bin-start_bin+1
            num_times = 2

            if num_times > 1:
                t = numpy.arange(start_bin, start_bin + 2)

                # t=numpy.arange(start_bin, end_bin+1)
            else:
                num_times += 1
                # to avoid batch size of 1
                t = numpy.array([start_bin, start_bin])

            s = numpy.repeat(s, num_times)
            r = numpy.repeat(r, num_times)
            o = numpy.repeat(o, num_times)

            # '''

            if load_to_gpu:
                s = torch.autograd.Variable(torch.from_numpy(
                    s).cuda().unsqueeze(1), requires_grad=False)
                r = torch.autograd.Variable(torch.from_numpy(
                    r).cuda().unsqueeze(1), requires_grad=False)
                o = torch.autograd.Variable(torch.from_numpy(
                    o).cuda().unsqueeze(1), requires_grad=False)
                t = torch.autograd.Variable(torch.from_numpy(
                    t).cuda().unsqueeze(1), requires_grad=False)
            else:
                # CPU
                s = torch.autograd.Variable(torch.from_numpy(
                    s).unsqueeze(1), requires_grad=False)
                r = torch.autograd.Variable(torch.from_numpy(
                    r).unsqueeze(1), requires_grad=False)
                o = torch.autograd.Variable(torch.from_numpy(
                    o).unsqueeze(1), requires_grad=False)
                t = torch.autograd.Variable(torch.from_numpy(
                    t).unsqueeze(1), requires_grad=False)

            # print(facts[i],facts_track_range, i,s.shape, facts_time_chunk, len(numpy.nonzero(facts_track_range==i)))

            scores_t = scoring_function(s, r, o, None).data

            # save for later (all scores_t are same pick any one)
            scores_t_list.append(scores_t[-1])

        # scores_t_pickle=torch.tensor(scores_t_pickle)
        t = torch.from_numpy(facts[:, 3:]).unsqueeze(1)

        data_pickle = prepare_data_iou_scores(
            t, test_kb, scores_t=scores_t_list, load_to_gpu=load_to_gpu)
        data_pickle["facts"] = facts
        data_pickle["data_folder_full_path"] = test_kb.datamap.dataset_root

    elif method in ["start-end-exhaustive-sweep"]:
        num_relations = len(test_kb.datamap.relation_map)
        start_scores_t_list = []
        end_scores_t_list = []

        for i in range(0, int(facts.shape[0]), 1):
            fact = facts[i]
            s, r, o = fact[:3]

            s = numpy.repeat(s, 2)  # to avoid batch size of 1
            r = numpy.repeat(r, 2)
            o = numpy.repeat(o, 2)

            if load_to_gpu:
                s = torch.autograd.Variable(torch.from_numpy(
                    s).cuda().unsqueeze(1), requires_grad=False)
                r = torch.autograd.Variable(torch.from_numpy(
                    r).cuda().unsqueeze(1), requires_grad=False)
                o = torch.autograd.Variable(torch.from_numpy(
                    o).cuda().unsqueeze(1), requires_grad=False)
            else:  # CPU
                s = torch.autograd.Variable(torch.from_numpy(
                    s).unsqueeze(1), requires_grad=False)
                r = torch.autograd.Variable(torch.from_numpy(
                    r).unsqueeze(1), requires_grad=False)
                o = torch.autograd.Variable(torch.from_numpy(
                    o).unsqueeze(1), requires_grad=False)

            start_scores_t = scoring_function(s, r, o, None).data
            end_scores_t = scoring_function(s, r + num_relations, o, None).data

            # save for later (all scores_t are same pick any one)
            start_scores_t_list.append(start_scores_t[-1])
            end_scores_t_list.append(end_scores_t[-1])

        t = torch.from_numpy(facts[:, 3:]).unsqueeze(1)

        data_pickle = prepare_data_iou_scores(
            t, test_kb, scores_t=(start_scores_t_list, end_scores_t_list), load_to_gpu=load_to_gpu)
        data_pickle["facts"] = facts
        data_pickle["data_folder_full_path"] = test_kb.datamap.dataset_root

    else:
        raise Exception("Not implemented")

    return data_pickle


def evaluate(scoring_function, valid_kb, test_kb, time_args, dump_t_scores=None, load_to_gpu=True, save_time_results=None, save_text=""):
    method = time_args['method']

    valid_time_scores_dict = get_time_scores(scoring_function, valid_kb, method, load_to_gpu)
    test_time_scores_dict = get_time_scores(scoring_function, test_kb, method, load_to_gpu)

    # pdb.set_trace()


    if method in ['greedy-coalescing', 'start-end-exhaustive-sweep']:
        compute_interval_scores(valid_time_scores_dict, test_time_scores_dict, method=method, save_time_results=save_time_results)
    else:
        raise Exception("Not implemented")

    if dump_t_scores is not None:
        valid_pickle_filename = "./debug/{}_t_scores_{}".format(
            dump_t_scores, 'valid')
        with open(valid_pickle_filename, 'wb') as handle:
            pickle.dump(valid_time_scores_dict, handle)
            print("Pickled t scores to {}\n".format(valid_pickle_filename))

        test_pickle_filename = "./debug/{}_t_scores_{}".format(
            dump_t_scores, 'test')
        with open(test_pickle_filename, 'wb') as handle:
            pickle.dump(valid_time_scores_dict, handle)
            print("Pickled t scores to {}\n".format(test_pickle_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_file_test', help="test scores file path", required=True)
    parser.add_argument('--scores_file_valid', help="valid scores file path",
                        required=False)  # need for greedy coalescing
    parser.add_argument('--save_scores', default=0, type=int)  # need for greedy coalescing
    parser.add_argument('--save_text', default="", type=str, help="filename prefix for time scores pickle")

    # parser.add_argument('--method', help="scores file path", required=True)
    arguments = parser.parse_args()

    test_pickle = {}
    valid_pickle = {}

    with open(arguments.scores_file_test, 'rb') as handle:
        test_pickle = pickle.load(handle)

    with open(arguments.scores_file_valid, 'rb') as handle:
        valid_pickle = pickle.load(handle)

    compute_interval_scores(valid_pickle, test_pickle, arguments.save_scores, arguments.save_text)
