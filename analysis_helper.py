import os

from collections import defaultdict

import csv

YEARMIN = -50
YEARMAX = 3000

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


def get_all_maps(file_path, id2ent=None, id2rel=None):
    sro_map = defaultdict(lambda: [])
    srt_map = defaultdict(lambda: [])
    ort_map = defaultdict(lambda: [])

    sr_map = defaultdict(lambda: [])
    or_map = defaultdict(lambda: [])
    e_map = {}
    r_map = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [i.strip('\n').split('\t') for i in lines]
        print(lines[0])

        for l in lines:
            if len(l) == 5:
                s, r, o, t1, t2 = tuple(l)[:5]
                t = t1 + ' ' + t2
            elif len(l) == 4:
                s, r, o, t = tuple(l)[:4]
            else:
                s, r, o = tuple(l)[:3]
                t = '--'

            if id2rel:
                r = id2rel.get(r, r)

            if id2ent:
                s = id2ent.get(s, s)
                o = id2ent.get(o, o)

            sro_map[(s, r, o)].append(t)
            sr_map[(s, r)].append((o, t))
            or_map[(o, r)].append((s, t))
            srt_map[(s, r, t)].append(o)
            ort_map[(o, r, t)].append(s)

    return {'sro': sro_map, 'sr': sr_map, 'srt': srt_map, 'or': or_map, 'ort': ort_map}


def union_map(dicts):
    combined = defaultdict(lambda: [])
    for d in dicts:
        for key in d:
            combined[key] += (d[key])
    return combined


def compare(test, train):
    seen_count = 0
    total = 0
    cnts = []
    for i in test:
        if len(list(set(train[i]))) > 1:  # seen more than once (1 is minimum as it won't be considered otherwise)
            seen_count += len(test[i])
            cnts.append(len(train[i]))
        total += len(test[i])

    if len(cnts) > 0:
        mean_freq = sum(cnts) / len(cnts)
    else:
        mean_freq = 0


def entmap_rmap(data_dir):
    id2rel, id2ent = {}, {}

    print("Data dir:{}".format(data_dir))

    ent2id_file = os.path.join(data_dir, 'readable_entities.txt')
    rel2id_file = os.path.join(data_dir, 'readable_relations.txt')
    with open(ent2id_file, 'r') as f:
        for line in f:
            eid, e = line.strip().split('\t')[:2]
            id2ent[eid] = e

    with open(rel2id_file, 'r') as f:
        for line in f:
            relid, rel = line.strip().split('\t')[:2]
            id2rel[relid] = rel

    return id2ent, id2rel


def get_dataframe(dataset_root, kb, id2ent=None, id2rel=None):
    em, rem = kb.datamap.entity_map, kb.datamap.reverse_entity_map
    rm, rrm = kb.datamap.relation_map, kb.datamap.reverse_relation_map

    rtm = kb.datamap.intervalId2dateYears

    t_str_map=kb.datamap.id2TimeStr

    id2ent, id2rel = entmap_rmap(dataset_root)
    fact_list = []
    for fact in kb.facts:
        s, r, o = fact[:3]
        t = fact[3:]
        # s,r,o,t=fact

        r = rrm[r]
        s = rem[s]
        o = rem[o]

        if id2rel:
            r = id2rel.get(r, r)

        if id2ent:
            s = id2ent.get(s, s)
            o = id2ent.get(o, o)

        # t = str(rtm[t[-1]])
        t_i = str(rtm[t[time_index["t_i"]]])

        t_str=t_str_map[t[time_index["t_str"]]]
        fact_list.append([s, r, o, t_i, t_str])

    import pandas as pd
    df = pd.DataFrame(fact_list, columns=['e1', 'r', 'e2', 't','t_str'])

    return df


def get_year_span(time_str, default=[YEARMIN, YEARMAX]):  # might have to change according to datasets
    v = time_str.split()
    #     if(v[0] in ['occurSince','occurUntil','<occursSince>','<occursUntil>']):#for TA-x datasets
    #         v=[v[1],v[1]]

    #     print(v)
    assert (len(v) == 2)

    start = v[0].split('-')[0]
    end = v[1].split('-')[0]
    if start == '####':
        start = YEARMIN
    elif start.find('#') != -1 or len(start) != 4:
        start = YEARMIN

    if end == '####':
        end = YEARMAX
    elif end.find('#') != -1 or len(end) != 4:
        end = YEARMAX

    start = int(start)
    end = int(end)

    if end == YEARMAX and start != YEARMIN:
        end = start

    return start, end


def save_preds(valid_list, top5_head, top5_tail, top3_rel, ranks_head, ranks_tail, ranks_rel, save_text,
               rel_preds=True):

    save_text = save_text.replace('/', '_') #to handle icews*/large dataset    

    if 'WIKIDATA12k' in save_text:  # hyTE dataset
        dataset = 'WIKIDATA12k'
    elif 'YAGO11k' in save_text:  # hyTE dataset
        dataset = 'YAGO11k'
    elif 'wikidata' in save_text:  # TA-x dataset
        dataset = 'wikidata'
    elif 'icews05-15' in save_text:
        dataset='icews05-15'
    elif 'icews14' in save_text:
        dataset='icews14'
    else:
        print("No dataset found in provided string {}".format(save_text))
        raise Exception

    data_dir = './data/' + dataset

    id2rel, id2ent = {}, {}
    id2ent['<OOV>'] = '<OOV>'
    id2rel['<OOV>'] = '<OOV>'

    print(data_dir)
    ent2id_file = os.path.join(data_dir, 'readable_entities.txt')
    rel2id_file = os.path.join(data_dir, 'readable_relations.txt')
    with open(ent2id_file, 'r') as f:
        for line in f:
            eid, e = line.strip().split('\t')[:2]
            id2ent[eid] = e

    with open(rel2id_file, 'r') as f:
        for line in f:
            relid, rel = line.strip().split('\t')[:2]
            id2rel[relid] = rel

    file_name = os.path.join('./preds/', save_text + '_preds.csv')

    with open(file_name, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(["r", "e1", "e2", "t", "e1Rank", "e2Rank", "relRank", "e1_top5", "e2_top5", "rel_top3"])

        num_facts = len(valid_list)
        for i in range(num_facts):
            e1, r, e2, t = valid_list[i]
            e1Rank, e2Rank = ranks_head[i], ranks_tail[i]
            e1_top5, e2_top5 = top5_head[i], top5_tail[i]

            e1, e2 = id2ent[e1], id2ent[e2]
            r = id2rel[r]

            # print(rel_top3)

            e1_top5 = [id2ent[k] for k in e1_top5]
            e2_top5 = [id2ent[k] for k in e2_top5]

            if rel_preds:  # if model predicts relations as well
                rel_rank = ranks_rel[i]
                rel_top3 = top3_rel[i]
                rel_top3 = [id2rel[k] for k in rel_top3]
            else:
                rel_rank = -1
                rel_top3 = ['-1', '-1', '-1']

            # csv_writer.writerow([r,e1,e2,t,e1Rank,e2Rank,relRank,','.join(e1_top5), ','.join(e2_top5), ','.join(rel_top3)])
            csv_writer.writerow(
                [r, e1, e2, t, e1Rank, e2Rank, rel_rank, '\t'.join(e1_top5), '\t'.join(e2_top5), '\t'.join(rel_top3)])

    return
