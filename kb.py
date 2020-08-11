import numpy
import torch
from collections import defaultdict as dd
import os
import pdb

YEARMIN = 0  # -50
YEARMAX = 3000

YEAR_STR_LEN = 4 
DATASET = ''

# for icews 
DATASET = 'ICEWS'


class Datamap(object):
    """
    Creates and stores entity/relation/time maps for a given dataset
    """
    def __init__(self,dataset, dataset_root, use_time_interval=False):
        self.dataset = dataset
        self.dataset_root = dataset_root
        self.use_time_interval=use_time_interval

        self.unk_time_str="UNK-TIME"

        train_filename=os.path.join(dataset_root, 'train.txt')


        # ---entity/relation maps--- #
        self.entity_map = {}
        self.relation_map = {}
        self.reverse_entity_map = {}
        self.reverse_relation_map = {}

        with open(train_filename) as f:
            lines = f.readlines()
            lines = [l.strip("\n").split("\t") for l in lines]

            for l in lines:  # preparing data
                # --for entity and relation-- #
                if l[0] not in self.entity_map:
                    # eid = int(l[0]) #for Hyte-Partha model init
                    eid = len(self.entity_map)
                    self.entity_map[l[0]] = eid
                    self.reverse_entity_map[eid] = l[0]
                if l[2] not in self.entity_map:
                    # eid = int(l[2])
                    eid = len(self.entity_map)
                    self.entity_map[l[2]] = eid
                    self.reverse_entity_map[eid] = l[2]
                if l[1] not in self.relation_map:
                    # rid = int(l[1])
                    rid = len(self.relation_map)
                    self.relation_map[l[1]] = rid
                    self.reverse_relation_map[rid] = l[1]
                # ----------- #
        # ----------------------------- #


        # ---time maps--- #
        self.dateYear2id, self.id2dateYear, self.dateYears2intervalId, self.intervalId2dateYears, self.timeStr2Id, self.id2TimeStr = self.get_time_info(
            dataset_root)

        self.year2id = {} # needed if use_time_interval is True
        with open(train_filename) as f:
            lines = f.readlines()
            lines = [l.strip("\n").split("\t") for l in lines]
            # ----Mapping of time-interval-tuple to id----#
            if self.use_time_interval and (len(self.year2id) == 0):
                triple_time = dict()
                count = 0
                for line in lines:
                    triple_time[count] = [x.split('-')[0] for x in line[3:5]]
                    count += 1
                self.year2id = self.create_year2id(triple_time, bin_size=300)  # (bin_start, bin_end) to id
            # ------- #

        # time maps converted to a form that can be indexed
        self.id2dateYear_mat = self.convert_dict2mat(self.dateYear2id)
        self.intervalId2dateYears_mat_s, self.intervalId2dateYears_mat_e = self.convert_dict2mat_tup(
            self.dateYears2intervalId)
        self.intervalId2dateYearsId_mat_s = self.convert_year2id(self.intervalId2dateYears_mat_s, self.dateYear2id)
        self.intervalId2dateYearsId_mat_e = self.convert_year2id(self.intervalId2dateYears_mat_e, self.dateYear2id)
        self.binId2year_mat = self.convert_dict2mat(self.year2id)

        # --------------#


    @staticmethod
    def convert_year2id(mat_in, map_y2i):
        # ipdb.set_trace()
        mat_out = numpy.zeros(mat_in.shape)
        for i in range(mat_in.shape[0]):
            if int(mat_in[i]) == -1:
                mat_out[i] = int(map_y2i['UNK-TIME'])
            else:
                mat_out[i] = int(map_y2i[int(mat_in[i])])
        return mat_out

    @staticmethod
    def convert_dict2mat(dict_in):
        dict_mat = numpy.zeros(len(dict_in))
        # ipdb.set_trace()
        try:
            for key in dict_in.keys():
                if type(key) == tuple:
                    if key == ('UNK-TIME', 'UNK-TIME'):
                        dict_mat[int(dict_in[key])] = -1
                    else:
                        dict_mat[int(dict_in[key])] = int(numpy.mean(key))
                else:
                    if key == 'UNK-TIME':
                        dict_mat[int(dict_in[key])] = -1
                    else:
                        dict_mat[int(dict_in[key])] = int(key)
        except:
            pdb.set_trace()
        dict_mat = numpy.array(dict_mat)
        return dict_mat

    @staticmethod
    def convert_dict2mat_tup(dict_in):
        dict_mat_s = numpy.zeros(len(dict_in))
        dict_mat_e = numpy.zeros(len(dict_in))
        # ipdb.set_trace()
        try:
            for key in dict_in.keys():
                if key == ('UNK-TIME', 'UNK-TIME'):
                    dict_mat_s[int(dict_in[key])] = -1
                    dict_mat_e[int(dict_in[key])] = -1
                else:
                    dict_mat_s[int(dict_in[key])] = key[0]
                    dict_mat_e[int(dict_in[key])] = key[1]
        except:
            pdb.set_trace()
        dict_mat_s = numpy.array(dict_mat_s)
        dict_mat_e = numpy.array(dict_mat_e)
        return dict_mat_s, dict_mat_e

    def get_time_info(self, dataset_root=""):
        '''
        Reads all data (train+test+valid) and returns date(year) to id and time interval to id maps
        including their inverse maps
        '''
        files_to_read = ['train.txt', 'test.txt', 'valid.txt']

        all_years = []
        all_intervals = []
        dateYear2id = {}
        id2dateYear = {}
        dateYears2intervalId = {}
        intervalId2dateYears = {}
        timeStr2Id = {}
        id2TimeStr = {}
        time_str=self.unk_time_str
        for filename in files_to_read:
            with open(os.path.join(dataset_root, filename)) as f:
                lines = f.readlines()
                lines = [l.strip("\n").split("\t") for l in lines]
                for l in lines:
                    if len(l) == 4:
                        date = self.check_date_validity(l[3])
                        if date != -1:
                            all_years.append(date)
                            all_intervals.append((date, date))

                        time_str = l[3]

                    elif len(l) == 5:
                        date1 = self.check_date_validity(l[3])
                        if date1 != -1:
                            all_years.append(date1)

                        date2 = self.check_date_validity(l[4])
                        if date2 != -1:
                            all_years.append(date2)

                        if (date1 >= 0 and date2 >= 0) and (date2 < date1):
                            date2 = date1

                        if date1 >= 0 and date2 >= 0:
                            all_intervals.append((date1, date2))
                        elif date1 >= 0:
                            all_intervals.append((date1, YEARMAX))  # date1))
                        elif date2 >= 0:
                            all_intervals.append((YEARMIN, date2))  # ,date2))

                        time_str = '\t'.join(l[3:])

                    elif len(l)==3: # for non-temporal datasets
                        time_str = ''

                    if time_str not in timeStr2Id:
                        newId = len(timeStr2Id)
                        timeStr2Id[time_str] = newId
                        id2TimeStr[newId] = time_str

        if "####-##-##\t####-##-##" not in timeStr2Id:
            timeStr2Id["####-##-##\t####-##-##"] = len(timeStr2Id)
            id2TimeStr[timeStr2Id["####-##-##\t####-##-##"]] = "####-##-##\t####-##-##"

        # all_years.append(self.unk_time_str)
        # all_intervals.append((self.unk_time_str,self.unk_time_str))
        all_years.append(YEARMIN)
        all_years.append(YEARMAX)

        for index, year in enumerate(sorted(list(set(all_years)))):
            dateYear2id[year] = index
            id2dateYear[index] = year

        dateYear2id[self.unk_time_str] = len(dateYear2id)
        id2dateYear[dateYear2id[self.unk_time_str]] = self.unk_time_str

        all_intervals.append((YEARMIN, YEARMAX))
        for index, year_tup in enumerate(sorted(list(set(all_intervals)))):
            dateYears2intervalId[year_tup] = index
            intervalId2dateYears[index] = year_tup

        dateYears2intervalId[(self.unk_time_str, self.unk_time_str)] = len(
            dateYears2intervalId)  ##(year, yearmax) or (yearmin, year)
        intervalId2dateYears[dateYears2intervalId[(self.unk_time_str, self.unk_time_str)]] = (
            self.unk_time_str, self.unk_time_str)  # (0,0)#

        # print("dateYear2id:",dateYear2id)

        return dateYear2id, id2dateYear, dateYears2intervalId, intervalId2dateYears, timeStr2Id, id2TimeStr

    def create_year2id(self, triple_time, bin_size=300):
        year2id = dict()
        freq = dd(int)
        count = 0
        year_list = []

        for k, v in triple_time.items():
            try:
                start = v[0].split('-')[0]
                end = v[1].split('-')[0]
                # if(len(v)!=1):
                #     end = v[1].split('-')[0]
                # else:
                #     end='####'
            except:
                pdb.set_trace()

            if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
            if end.find('#') == -1 and len(end) == 4: year_list.append(int(end))

        year_list.sort()
        for year in year_list:
            freq[year] = freq[year] + 1

        year_class = []
        count = 0
        for key in sorted(freq.keys()):
            count += freq[key]
            if count > bin_size:
                year_class.append(key)
                count = 0
        prev_year = 0
        i = 0
        for i, yr in enumerate(year_class):
            year2id[(prev_year, yr)] = i
            prev_year = yr + 1
        year2id[(prev_year, max(year_list))] = i + 1
        self.year_list = year_list

        return year2id

    def check_date_validity(self, date):
        # if DATASET == 'ICEWS':
        if self.dataset.lower().startswith('icews'):
            year, month, day = date.split('-')
            # return int(year + month + day)
            return int(year)*375 + int(month)*31 + int(day)
        start = date.split('-')[0]
        if start.find('#') == -1 and len(start) == YEAR_STR_LEN:
            return int(start)
        else:
            return -1




class kb(object):
    """
    Stores a knowledge base as an numpy array. Can be generated from a file. Also stores the entity/relation mappings
    (which is the mapping from entity names to entity id) and possibly entity type information.
    """

    def __init__(self, datamap, filename, add_unknowns: bool = True,
                 nonoov_entity_count: int = None,
                 use_time_tokenizer: bool = False) -> object:
        """
        Duh...
        :param filename: The file name to read the kb from
        :param em: Prebuilt entity map to be used. Can be None for a new map to be created
        :param rm: prebuilt relation map to be used. Same as em
        :param add_unknowns: Whether new entities are to be acknowledged or put as <UNK> token.
        """

        self.datamap = datamap
        self.use_time_tokenizer = use_time_tokenizer
        self.filename=filename

        facts_time_tokens = []  # for TA-x models
        facts = []

        if filename is None:
            return

        # --for time--#
        self.unk_time_str = 'UNK-TIME'  # for facts with no time stamp or invalid time stamps
        # ----------- #

        self.nonoov_entity_count = 0 if nonoov_entity_count is None else nonoov_entity_count

        print("KB", filename, add_unknowns)

        with open(filename) as f:
            lines = f.readlines()
            lines = [l.strip("\n").split("\t") for l in lines]

            # ----------- #
            count_missed_facts = 0
            for l in lines:  # preparing data

                # Main Job
                time_str = 'UNK-TIME'

                if len(l)==3: # for non-temporal datasets
                    facts.append([self.datamap.entity_map.get(l[0], len(self.datamap.entity_map) - 1),
                                    self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1),
                                    self.datamap.entity_map.get(l[2], len(self.datamap.entity_map) - 1),
                                    0, 0, 0, 0, 0,0])
                else:
                    if self.datamap.use_time_interval:
                        if len(l) == 5:  # timestamp of the form "occursSince <YEAR>" or "<YEAR1> <YEAR2>"
                            t_start_lbl, t_end_lbl = self.get_span_ids(l[3], l[4])
                            if t_start_lbl == "" or t_end_lbl == "":
                                count_missed_facts += 1
                                continue
                            assert t_end_lbl >= t_start_lbl
                            start, end = self.get_date_range(l)
                            time_interval_str_id = self.datamap.dateYears2intervalId.get((start, end), len(
                                self.datamap.dateYears2intervalId) - 1)  # self.dateYears2intervalId[(start,end)
                            # ipdb.set_trace()

                            time_str = '\t'.join(l[3:])
                            time_str_id = self.datamap.timeStr2Id[time_str]

                            facts.append([self.datamap.entity_map.get(l[0], len(self.datamap.entity_map) - 1),
                                        self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1),
                                        self.datamap.entity_map.get(l[2], len(self.datamap.entity_map) - 1),
                                        t_start_lbl, t_start_lbl, t_end_lbl, t_end_lbl, time_str_id,
                                        time_interval_str_id])

                        elif len(l) != 3:
                            print("Unknown time format")
                            raise Exception
                        else:
                            count_missed_facts += 1
                    else:
                        if len(l) > 3:
                            start, end = self.get_date_range(l)
                            start_id, end_id = (self.datamap.dateYear2id[start], self.datamap.dateYear2id[end])
                            time_interval_str_id = self.datamap.dateYears2intervalId.get((start, end), len(
                                self.datamap.dateYears2intervalId) - 1)  # self.dateYears2intervalId[(start,end)]

                            time_str = '\t'.join(l[3:])  # l[-1]
                            time_str_id = self.datamap.timeStr2Id[time_str]

                            facts.append([self.datamap.entity_map.get(l[0], len(self.datamap.entity_map) - 1),
                                        self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1),
                                        self.datamap.entity_map.get(l[2], len(self.datamap.entity_map) - 1),
                                        start_id, start_id, end_id, end_id, time_str_id, time_interval_str_id])
                        elif len(l) < 3:
                            print("Bad data: Unknown time format")
                            raise Exception

                if self.use_time_tokenizer:
                    # time_tokens=tokenize_time(time,filename)
                    time_tokens = tokenize_time(time_str, filename)
                    facts_time_tokens.append(time_tokens)

        self.facts_time_tokens = numpy.array(facts_time_tokens, dtype='int64')
        self.facts = numpy.array(facts, dtype='int64')

        print("Data Size:", filename, self.facts.shape)


    def expand_data(self, mode="all"):
        '''
        mode = all/start/end/both/start-mid-end
        index3 updated here!
        '''
        new_facts = []
        for fact in self.facts:
            e1, r, e2, t_start, t_start, t_end, t_end, t_str, t_interval = fact
            if mode == "start-end-diff-relation":
                new_facts.append([e1, r, e2, t_start, t_start, t_end, t_end, t_str, t_interval])
                new_facts.append([e1, r + len(self.datamap.relation_map), e2, t_end, t_start, t_end, t_end, t_str, t_interval])
                continue

            if t_start == t_end:
                step = 1
            elif mode == "all":
                step = 1
            elif mode == "both":
                step = t_end - t_start
            elif mode == "start":
                step = t_end
            elif mode == "start-mid-end":
                step = int((t_end - t_start) / 2.0)
            elif mode == "end":
                step = -(t_start + 1)
                x = t_start
                t_start = t_end
                t_end = x - 2
                assert t_end <= x - 1

            for tid in range(t_start, t_end + 1, step):
                new_facts.append([e1, r, e2, tid, t_start, t_end, t_end, t_str, t_interval])

        self.facts = numpy.array(new_facts, dtype='int64')

    def get_all_data(self, dataset_root=""):
        files_to_read = ['train.txt', 'test.txt', 'valid.txt']
        all_data = []
        for filename in files_to_read:
            with open(os.path.join(dataset_root, filename)) as f:
                lines = f.readlines()
                lines = [l.strip("\n").split("\t") for l in lines]
            all_data += lines
        return all_data

    def get_date_range(self, fact):
        if len(fact) == 3:
            t1 = t2 = "###"
        elif len(fact) == 4:
            _, _, _, t1 = fact
            t2 = t1
        else:
            _, _, _, t1, t2 = fact
        # start = self.check_date_validity(t1)
        # end = self.check_date_validity(t2)
        start = self.datamap.check_date_validity(t1)
        end = self.datamap.check_date_validity(t2)

        if (start != -1 and end != -1) and (start > end):
            end = start
        if start == -1 and end != -1:
            start = YEARMIN  # self.unk_time_str#end
        elif start != -1 and end == -1:
            end = YEARMAX  # self.unk_time_str#start
        elif start == -1 and end == -1:
            # start = end = self.unk_time_str
            start = YEARMIN
            # end = YEARMAX
            end = YEARMIN

        return start, end

    def get_span_ids(self, start_in, end_in):
        try:
            start = start_in.split('-')[0]
            end = end_in.split('-')[0]
        except:
            pdb.set_trace()

        if start == '####':
            start = YEARMIN
        elif start.find('#') != -1 or len(start) != YEAR_STR_LEN:
            start = YEARMIN
            # return "",""

        if end == '####':
            end = YEARMAX
        elif end.find('#') != -1 or len(end) != YEAR_STR_LEN:
            end = YEARMAX
            # return "",""

        start = int(start)
        end = int(end)
        if start > end:
            end = YEARMAX

        # ---(1980-####) should have end=start=1980, similar for (####-1980)---#
        # if(start==YEARMIN and end!=YEARMAX):
        #     start=end
        # if(end==YEARMAX and start!=YEARMIN):
        #     end=start
        # -------#

        if start == YEARMIN:
            start_lbl = 0
        else:
            for key, lbl in sorted(self.datamap.year2id.items(), key=lambda x: x[1]):
                if start >= key[0] and start <= key[1]:
                    start_lbl = lbl

        if end == YEARMAX:
            end_lbl = len(self.datamap.year2id.keys()) - 1
        else:
            for key, lbl in sorted(self.datamap.year2id.items(), key=lambda x: x[1]):
                if end >= key[0] and end <= key[1]:
                    end_lbl = lbl
        return start_lbl, end_lbl

    def augment_type_information(self, mapping):
        """
        Augments the current knowledge base with entity type information for more detailed evaluation.\n
        :param mapping: The maping from entity to types. Expected to be a int to int dict
        :return: None
        """
        self.type_map = mapping
        entity_type_matrix = numpy.zeros((len(self.entity_map), 1))
        for x in self.type_map:
            if x not in self.entity_map.keys():  # ignore entities not in training set   28/08/19 (sushant)
                continue
            entity_type_matrix[self.entity_map[x], 0] = self.type_map[x]
        entity_type_matrix = torch.from_numpy(numpy.array(entity_type_matrix))
        self.entity_type_matrix = entity_type_matrix

    def compute_degree(self, out=True):
        """
        Computes the in-degree or out-degree of relations\n
        :param out: Whether to compute out-degree or in-degree
        :return: A numpy array with the degree of ith ralation at ith palce.
        """
        entities = [set() for x in self.relation_map]
        index = 2 if out else 0
        for i in range(self.facts.shape[0]):
            entities[self.facts[i][1]].add(self.facts[i][index])
        return numpy.array([len(x) for x in entities])

    # ----useful methods for charCNN---#

    def charcnn_packaged(self, ls,
                         load_to_gpu=True):  # ls expected of the form (s,r,o,ns,no), any prefix of this tuple works as well
        # print("ls:",ls)
        rem, rrm = self.reverse_entity_map, self.reverse_relation_map
        ls_char = []
        for i, idx_list in enumerate(ls[:3]):  # s,r,o
            char_embeddings = numpy.zeros(shape=(len(idx_list), self.dataset.alphabet_size, self.dataset.max_len),
                                          dtype=float)
            # print("idx_list:",idx_list.shape)
            for j in range(len(idx_list.tolist())):  # .tolist()):
                # print("idx: ",idx)
                idx = idx_list[j]

                if i == 1:  # relations
                    emb = self.dataset.one_hot_encode(rrm[int(idx)])
                else:  # entities
                    emb = self.dataset.one_hot_encode(rem[int(idx)])

                char_embeddings[j, :, :] = emb

            ls_char.append(char_embeddings)
            # ls[i]=(idx_list,char_embeddings)

        for i, idx_list in enumerate(ls[3:], start=3):  # ns,no
            char_embeddings = numpy.zeros(
                shape=(idx_list.shape[0], idx_list.shape[1], self.dataset.alphabet_size, self.dataset.max_len),
                dtype=float)
            # print("idx_list:",idx_list.shape)
            for row, col in zip(range(idx_list.shape[0]), range(idx_list.shape[1])):  # .tolist()):
                # print("idx: ",idx)
                idx = idx_list[row, col]
                if i == 1:  # relations
                    emb = self.dataset.one_hot_encode(rrm[int(idx)])
                else:  # entities
                    emb = self.dataset.one_hot_encode(rem[int(idx)])

                char_embeddings[row, col, :, :] = emb

            ls_char.append(char_embeddings)
            # ls[i]=(idx_list,char_embeddings)

        if load_to_gpu:
            return [torch.autograd.Variable(torch.from_numpy(x).type(torch.FloatTensor).cuda()) for x in ls_char]
        else:
            return [torch.autograd.Variable(torch.from_numpy(x).type(torch.FloatTensor)) for x in ls_char]

        # if load_to_gpu:
        #     return [(torch.autograd.Variable(torch.from_numpy(x[0]).cuda()),
        #         torch.autograd.Variable(torch.from_numpy(x[1]).type(torch.FloatTensor).cuda()) ) for x in ls]
        # else:
        #     return [(torch.autograd.Variable(torch.from_numpy(x[0])),
        #         torch.autograd.Variable(torch.from_numpy(x[1]).type(torch.FloatTensor)) ) for x in ls]

    ##--------------------------##


def dict_union(dicts=[]):
    combined = {}
    for d in dicts:
        for key in d:
            combined[key] = d[key]
    return combined


def union(kb_list):
    """
    Computes a union of multiple knowledge bases\n
    :param kb_list: A list of kb
    :return: The union of all kb in kb_list
    """
    any_kb = kb_list[0]
    k = kb(any_kb.datamap, None)

    l = [k.facts for k in kb_list]
    k.facts = numpy.concatenate(l, 0)

    # --for TA-x models--#
    l = [k.facts_time_tokens for k in kb_list]
    k.facts_time_tokens = numpy.concatenate(l, 0)
    # -------------------#

    return k


def dump_mappings(mapping, filename):
    """
    Stores the mapping into a file\n
    :param mapping: The mapping to store
    :param filename: The file name
    :return: None
    """
    data = [[x, mapping[x]] for x in mapping]
    numpy.savetxt(filename, data)


def dump_kb_mappings(kb, kb_name):
    """
    Dumps the entity and relation mapping in a kb\n
    :param kb: The kb
    :param kb_name: The fine name under which the mappings should be stored.
    :return:
    """
    dump_mappings(kb.entity_map, kb_name + ".entity")
    dump_mappings(kb.relation_map, kb_name + ".relation")


# ----------------------------------#

UNK_TIME_STR = 'UNK-TIME'

tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18, '10m': 19,
    '11m': 20, '12m': 21,
    '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
    'occurSince': 32, 'occurUntil': 33, UNK_TIME_STR: 34, '<occursSince>': 35, '<occursUntil>': 36, '#y': 37, '##m': 38,
    '#d': 39, '<PAD>': 40
}


def tokenize_time(time_str, data_file_path):
    dataset, seq_len = None, None
    if "wikidata" in data_file_path:  # hackish, pass dataset to kb instead
        dataset = "wikidata"
    elif "yago15k" in data_file_path:
        dataset = "yago15k"
    elif "YAGO11k" in data_file_path:
        dataset = "YAGO11k"
    elif "WIKIDATA12k" in data_file_path:
        dataset = "WIKIDATA12k"
    elif "icews05-15" in data_file_path:
        dataset = "icews05-15"
    elif "icews14" in data_file_path:
        dataset = "icews14"
    else:
        print("Unknown dataset for TA-x model")
        raise Exception

    seq_len = None
    if dataset == "wikidata":
        seq_len = 5
    elif dataset == "yago15k":
        seq_len = 8
    elif dataset == "YAGO11k":
        seq_len = 5
    elif dataset == "WIKIDATA12k":
        seq_len = 5
    elif dataset == "icews05-15" or "icews14":
        seq_len = 8

    tem_id_list = []

    if dataset == 'yago15k':
        if time_str == UNK_TIME_STR:  # yago15k has facts without time stamp
            time_str = '<occursSince> "####-##-##"'

        if (time_str.split(' ')[0] in ['<occursSince>',
                                       '<occursUntil>']):  # string of the form '<occursSince> "YEAR"' or '<occursUntil> "YEAR"' (yago15k)
            # print(time_str)
            if len(time_str.split(' ')) == 2:
                descr, x = time_str.split(' ')
            else:  # corner case to handle bad training facts like '<occursSince>' without year
                descr = time_str.split(' ')[0]
                x = '"####-##-##"'

            tem_id_list.append(tem_dict[descr])
            time_str = x[1:-1]

            year, month, day = time_str.split("-")
            for j in range(len(year)):
                token = year[j:j + 1] + 'y'
                tem_id_list.append(tem_dict[token])

            for j in range(1):
                token = month + 'm'
                tem_id_list.append(tem_dict[token])

            for j in range(len(day)):
                token = day[j:j + 1] + 'd'
                tem_id_list.append(tem_dict[token])

    elif dataset == 'wikidata':

        if time_str == UNK_TIME_STR:
            # print("UNKNOWN TIME: {}".format(time_str))
            # print(time_str)
            # xx=input()
            tem_id_list.append(tem_dict[UNK_TIME_STR])
            # return tem_id_list

        elif (time_str.split(' ')[0] in ['occurSince',
                                         'occurUntil']):  # string of the form 'occurSince YEAR' or 'occurUntil YEAR' (wikidata)
            descr, year = time_str.split(' ')
            tem_id_list.append(tem_dict[descr])
            for j in range(len(year)):
                token = year[j:j + 1] + 'y'
                tem_id_list.append(tem_dict[token])
            # print(tem_id_list)
            # return tem_id_list    

    if dataset == 'icews05-15' or dataset == 'icews14':
        # time_str=x[1:-1]

        year, month, day = time_str.split("-")
        for j in range(len(year)):
            token = year[j:j + 1] + 'y'
            tem_id_list.append(tem_dict[token])

        for j in range(1):
            token = month + 'm'
            tem_id_list.append(tem_dict[token])

        for j in range(len(day)):
            token = day[j:j + 1] + 'd'
            tem_id_list.append(tem_dict[token])

    # print("DIFFERENT FORMAT!")
    # xx=input()

    # if(len(tem_id_list)==7):
    # raise Exception
    if len(tem_id_list) < seq_len:
        while len(tem_id_list) != seq_len:
            tem_id_list.append(tem_dict['<PAD>'])

    return tem_id_list
