import numpy
import torch
import torch.autograd

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


class data_loader(object):
    """
    Does th job of batching a knowledge base and also generates negative samples with it.
    """

    def __init__(self, kb, load_to_gpu, first_zero=False, loss=None, flag_add_reverse=None, model="",
                 perturb_time=False):
        """
        Duh..\n
        :param kb: the knowledge base to batch
        :param load_to_gpu: Whether the batch should be loaded to the gpu or not
        :param first_zero: Whether the first entity in the set of negative samples of each fact should be zero
        """
        self.kb = kb
        self.load_to_gpu = load_to_gpu
        self.first_zero = first_zero
        self.flag_add_reverse = flag_add_reverse
        self.loss = loss

        self.model = model

        self.perturb_time = perturb_time


    def sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s = numpy.expand_dims(facts[:, 0], -1)
        r = numpy.expand_dims(facts[:, 1], -1)
        o = numpy.expand_dims(facts[:, 2], -1)
        if len(self.kb.facts_time_tokens) == 0:
            t = numpy.expand_dims(facts[:, 3:], -1)

            if self.perturb_time:
                for i in range(len(t)):
                    start, end = t[i, time_index["t_s_orig"], 0], t[i, time_index["t_e_orig"], 0]
                    t[i, time_index["t_s"], 0] = numpy.random.randint(low=start, high=end + 1)



        else:  # for TA-x model
            t = self.kb.facts_time_tokens[indexes]
            t = numpy.expand_dims(t, -1)
            # print("t shape during training:",t.shape)

        ns = numpy.random.randint(low=0, high=len(self.kb.datamap.entity_map) - 1, size=(batch_size, negative_count))
        no = numpy.random.randint(low=0, high=len(self.kb.datamap.entity_map) - 1, size=(batch_size, negative_count))
        if self.first_zero:
            ns[:, 0] = len(self.kb.datamap.entity_map) - 1
            no[:, 0] = len(self.kb.datamap.entity_map) - 1
        return [s, r, o, t, ns, no]

    def sample_time(self, batch_size=1000, negative_count=10, exclude=None):
        '''
        Generates a random sample of times of size batch_size x negative_count
        :param exclude: Since number of times are small, excluding the correct time is important when not doing full softmax
        '''

        if self.kb.datamap.use_time_interval:
            num_times = len(self.kb.datamap.year2id)
        else:
            num_times = len(self.kb.datamap.dateYear2Id)

        if exclude is None:
            nt = numpy.random.randint(low=0, high=num_times - 1, size=(batch_size, negative_count))
        else:
            # need to implement
            pass

        return nt

    def sample_icml(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s_fwd = numpy.expand_dims(facts[:, 0], -1)
        r_fwd = numpy.expand_dims(facts[:, 1], -1)
        o_fwd = numpy.expand_dims(facts[:, 2], -1)

        ns_fwd = numpy.random.randint(low=0, high=len(self.kb.datamap.entity_map) - 1, size=(batch_size, negative_count))
        no_fwd = numpy.random.randint(low=0, high=len(self.kb.datamap.entity_map) - 1, size=(batch_size, negative_count))

        if len(self.kb.facts_time_tokens) == 0:
            t = numpy.expand_dims(facts[:, 3:], -1)

            if self.perturb_time:
                for i in range(len(t)):
                    start, end = t[i, time_index["t_s_orig"], 0], t[i, time_index["t_e_orig"], 0]
                    t[i, time_index["t_s"], 0] = numpy.random.randint(low=start, high=end + 1)

        else:  # for TA-x model
            t = self.kb.facts_time_tokens[indexes]
            t = numpy.expand_dims(t, -1)
            # print("t shape during training:",t.shape)


        # if self.loss == "crossentropy_loss":
        #     ns_fwd = None;
        #     no_fwd = None
        # else:
        #     ns_fwd = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
        #     no_fwd = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
        num_relations = len(self.kb.datamap.relation_map)
        r_rev = r_fwd + num_relations

        s = numpy.concatenate([s_fwd, o_fwd])
        r = numpy.concatenate([r_fwd, r_rev])
        o = numpy.concatenate([o_fwd, s_fwd])
        t = numpy.concatenate([t, t])

        ns = numpy.concatenate([ns_fwd, no_fwd])  ##to do randomly generate ns_rev and no_rev
        no = numpy.concatenate([no_fwd, ns_fwd])
        if self.first_zero:
            ns[:, 0] = self.kb.nonoov_entity_count - 1
            no[:, 0] = self.kb.nonoov_entity_count - 1

        # if self.loss == "crossentropy_loss":
        #     ns = None
        #     no = None
        # else:
        #     ns = numpy.concatenate([ns_fwd, no_fwd])  ##to do randomly generate ns_rev and no_rev
        #     no = numpy.concatenate([no_fwd, ns_fwd])
        #     if self.first_zero:
        #         ns[:, 0] = self.kb.nonoov_entity_count - 1
        #         no[:, 0] = self.kb.nonoov_entity_count - 1

        return [s, r, o, t, ns, no]

    def tensor_sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as torch tensors. Internally uses sampe\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        if self.flag_add_reverse:
            ls = self.sample_icml(batch_size, negative_count)
        else:
            ls = self.sample(batch_size, negative_count)

        if self.load_to_gpu:
            ls_ret = [torch.autograd.Variable(torch.from_numpy(x).cuda()) for x in ls]
        else:
            ls_ret = [torch.autograd.Variable(torch.from_numpy(x)) for x in ls]

        if self.model.endswith("charCNN"):  # load one-hot character encodings for each entity/relation
            ls_char = self.kb.charcnn_packaged(ls, self.load_to_gpu)
            return list(zip(ls_ret, ls_char))
        else:
            return ls_ret

