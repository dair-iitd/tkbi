"""
Contains different inference procedures for interval prediction
"""
import torch

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


def greedy_coalescing(probs, thresholds, k=1):
    """
    params-
    probs: [n x t]- tensor, prob score for each time (t times) for each instance
    thresholds: [n x 1]- threshold for each instance
    k: number of intervals to be returned for each
    returns-
    (start, end), where start and end are [n x k] tensors,
    indicating boundaries for top k intervals for each instance
    """

    batch_size = len(probs)
    num_times = probs.shape[-1]

    # _,best_indices= torch.max(probs,-1)
    indices = torch.argsort(probs, descending=True)[:, :k]

    pred_min = torch.zeros(batch_size, k)
    pred_max = torch.zeros(batch_size, k)

    for i in range(batch_size):
        for idx, best_t in enumerate(indices[i]):
            best_t = int(best_t)
            left, right = best_t, best_t
            tot = probs[i, best_t]

            thresh = thresholds[i]

            # print("\ntot:{}, thresh:{}".format(tot,thresh))

            # print("Initial:",left,right)
            while tot < thresh and (left > 0 or right < num_times - 1):
                next_index = -1
                if (left == 0):
                    right += 1
                    next_index = right
                elif (right == num_times - 1):
                    left -= 1
                    next_index = left
                else:
                    left_score = probs[i, left - 1]
                    right_score = probs[i, right + 1]
                    # print("left_score:{}, right_score:{}".format(left_score,right_score))

                    if (left_score > right_score):
                        left -= 1
                        next_index = left
                    else:
                        right += 1
                        next_index = right

                tot += probs[i, next_index]

            # print("pred_min shape:",pred_min.shape)
            # print("indices shape:",indices.shape)
            pred_min[i, idx] = left
            pred_max[i, idx] = right

        # print("Later:",left,right)

    # print("{}/{} done".format(i,batch_size))

    return pred_min, pred_max


def greedy_coalescing_durations(probs, durations, k=1):
    """
    params-
    probs: [n x t]- tensor, prob score for each time (t times) for each instance
    thresholds: [n x 1]- threshold for each instance
    k: number of intervals to be returned for each
    returns-
    (start, end), where start and end are [n x k] tensors,
    indicating boundaries for top k intervals for each instance
    """

    batch_size = len(probs)
    num_times = probs.shape[-1]

    # _,best_indices= torch.max(probs,-1)
    indices = torch.argsort(probs, descending=True)[:, :k]

    pred_min = torch.zeros(batch_size, k)
    pred_max = torch.zeros(batch_size, k)

    for i in range(batch_size):
        for idx, best_t in enumerate(indices[i]):
            best_t = int(best_t)
            left, right = best_t, best_t
            span = 1

            duration = durations[i]

            # print("\ntot:{}, thresh:{}".format(tot,thresh))

            # print("Initial:",left,right)
            while span <= duration and (left > 0 or right < num_times - 1):
                next_index = -1
                if (left == 0):
                    right += 1
                    next_index = right
                elif (right == num_times - 1):
                    left -= 1
                    next_index = left
                else:
                    left_score = probs[i, left - 1]
                    right_score = probs[i, right + 1]
                    # print("left_score:{}, right_score:{}".format(left_score,right_score))

                    if (left_score > right_score):
                        left -= 1
                        next_index = left
                    else:
                        right += 1
                        next_index = right

                span += 1

            # print("pred_min shape:",pred_min.shape)
            # print("indices shape:",indices.shape)
            pred_min[i, idx] = left
            pred_max[i, idx] = right

        # print("Later:",left,right)

    # print("{}/{} done".format(i,batch_size))

    return pred_min, pred_max


def duration_exhaustive_sweep(probs, durations, k=1):
    """
    compute scores for all (start,end) pairs of given duration.
    return top k scores.
    scores can be raw model scores or softmax output.
    """
    batch_size = len(probs)
    pred_min = torch.zeros(batch_size, k)
    pred_max = torch.zeros(batch_size, k)

    # '''
    # given probs and durations
    for batch_id in range(probs.shape[0]):
        best_score_duration = []
        best_scores = []

        all_scores = probs[batch_id]

        duration = durations[batch_id]

        score_duration = [];
        score_duration_range = []
        range_start = torch.tensor([0])  # minimum possible start
        range_end = probs.shape[-1] - duration - 1  # maximum possible start

        for start_time in range(range_start, range_end + 1, 1):
            end_time = start_time + duration
            score_duration.append(all_scores[start_time:end_time + 1].sum())
            score_duration_range.append((start_time, end_time))

        score_duration = torch.tensor(score_duration)
        best_k = torch.argsort(score_duration, descending=True)[:k]  # pick best k

        for j, idx in enumerate(best_k):  # store start and end boundaries
            start, end = score_duration_range[idx]
            pred_min[batch_id, j] = start
            pred_max[batch_id, j] = end

    return pred_min, pred_max


def start_end_exhaustive_sweep(start_scores, end_scores, k=1):
    """
    compute scores for all possible (start,end) pairs (sum of the 2).
    return top k scores.
    scores can be raw model scores or softmax output.
    """
    batch_size = len(start_scores)
    num_times = start_scores.shape[-1]

    pred_min = torch.zeros(batch_size, k)
    pred_max = torch.zeros(batch_size, k)

    print("num_times:", num_times)

    xx = 0

    for i in range(batch_size):
        durations = []  # store durations- start,end pairs
        duration_scores = []  # store scores for each duration

        # print(start_scores[i,1] + end_scores[i,5])

        sum_scores = start_scores[i, :].unsqueeze(-1) + end_scores[i, :].unsqueeze(-1).t()
        # print(sum_scores[1,5])
        sum_scores = torch.triu(sum_scores)
        sum_scores = sum_scores.flatten()
        # print(sum_scores[1*num_times + 5])
        # xx=input()
        best_k = torch.argsort(sum_scores, descending=True)[:k]  # pick best k

        # for start in range(num_times):
        # 	for end in range(start,num_times):
        # 		duration_scores.append(start_scores[i,start]+end_scores[i,end])
        # 		durations.append((start,end))
        # 	print("start:",start)

        # duration_scores=torch.tensor(duration_scores)
        # best_k=torch.argsort(duration_scores,descending=True)[:k] #pick best k

        for j, idx in enumerate(best_k):  # store start and end boundaries
            end = idx % num_times
            start = (idx - end) / num_times
            # start,end=durations[idx]
            pred_min[i, j] = start
            pred_max[i, j] = end

        if (i % 100 == 0):
            print("{}/{} examples completed".format(i, batch_size))
            print(pred_min[i, 0], pred_max[i, 0])

    return pred_min, pred_max


def duration_scores(probs, durations, k=1):
    """
    computes top k intervals, where i'th best interval is that of given duration containing
    i'th best time score.
    """
    batch_size = len(probs)
    pred_min = torch.zeros(batch_size, k)
    pred_max = torch.zeros(batch_size, k)

    # '''
    # given probs and durations
    for batch_id in range(probs.shape[0]):
        best_score_duration = []
        best_scores = []

        all_scores = probs[batch_id]

        indices = torch.argsort(all_scores, descending=True)[:k]

        duration = durations[batch_id]

        for idx, best_t in enumerate(indices):
            score_duration = [];
            score_duration_range = []
            range_start = torch.max((best_t - duration), torch.tensor([0]))
            # range_end   = torch.min(best_t, torch.tensor(probs.shape[-1]-duration-1))
            range_end = torch.min(best_t, probs.shape[-1] - duration - 1)

            # print("range_start:{}, range_end:{}, duration:{}".format(range_start, range_end, duration))
            for start_time in range(range_start, range_end + 1, 1):
                end_time = start_time + duration
                score_duration.append(all_scores[start_time:end_time + 1].sum())
                score_duration_range.append((start_time, end_time))
            score_duration = torch.tensor(score_duration)
            # try:
            start, end = score_duration_range[torch.argmax(score_duration)]

            pred_min[batch_id, idx] = start
            pred_max[batch_id, idx] = end

        # print("Predicted- start:{}, end:{}".format(start,end))
        # print("duration:{}".format(duration))
        # xx=input()

        # best_score_duration.append(score_duration_range[torch.argmax(score_duration)])
        # best_scores.append(score_duration[torch.argmax(score_duration)])

    # best_score_duration_start = []
    # best_score_duration_end = []
    # for ele in best_score_duration:
    #     s,e = ele
    #     best_score_duration_start.append(kvalid.id2year_mat[s])
    #     best_score_duration_end.append(kvalid.id2year_mat[e])

    # best_score_duration_start = torch.tensor(best_score_duration_start)
    # best_score_duration_end   = torch.tensor(best_score_duration_end)
    return pred_min, pred_max


