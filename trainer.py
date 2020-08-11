import numpy
import time
import evaluate
import torch
import kb
import utils
import os
import random
import sys

import pdb

import losses

from time_prediction.evaluate import evaluate as time_evaluate

time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}


def log_eval_scores(writer, valid_score, test_score, num_iter):
    for metric in ['mrr', 'hits10', 'hits1']:
        writer.add_scalar('{}/valid_m'.format(metric), valid_score['m'][metric], num_iter)
        writer.add_scalar('{}/valid_e1'.format(metric), valid_score['e1'][metric], num_iter)
        writer.add_scalar('{}/valid_e2'.format(metric), valid_score['e2'][metric], num_iter)
        writer.add_scalar('{}/valid_r'.format(metric), valid_score['r'][metric], num_iter)
        writer.add_scalar('{}/valid_t'.format(metric), valid_score['t'][metric], num_iter)

        writer.add_scalar('{}/test_m'.format(metric), test_score['m'][metric], num_iter)
        writer.add_scalar('{}/test_e1'.format(metric), test_score['e1'][metric], num_iter)
        writer.add_scalar('{}/test_e2'.format(metric), test_score['e2'][metric], num_iter)
        writer.add_scalar('{}/test_r'.format(metric), valid_score['r'][metric], num_iter)
        writer.add_scalar('{}/test_t'.format(metric), valid_score['t'][metric], num_iter)

    return


def get_time_facts(t, r):
    t_start = t[:, time_index["t_s_orig"], :]
    t_end = t[:, time_index["t_e_orig"], :]

    positive_r = r.clone()
    positive_r[:, 0] = 0  # 0 relation- less than

    negative_r = r.clone()
    negative_r[:, 0] = 1  # 1 relation- greater than
    return t_start, t_end, positive_r, negative_r


class Trainer(object):
    def __init__(self, scoring_function, scoring_function_arguments, regularizer, loss, optim, train, valid, test,
                 verbose=0, batch_size=1000,
                 hooks=None, eval_batch=100, negative_count=10, gradient_clip=None, regularization_coefficient=0.01,
                 save_dir="./logs", scheduler=None, debug=0, time_neg_samples=False, expand_mode="None",
                 filter_method="time-interval",
                 flag_additional_filter=1, use_time_facts=0, time_loss_margin=5.0,
                 predict_time=0, time_args=None, flag_add_reverse=0, load_to_gpu=True):
        super(Trainer, self).__init__()
        self.scoring_function = scoring_function
        self.scoring_function_arguments = scoring_function_arguments  # needed for model init later on
        self.loss = loss
        self.regularizer = regularizer
        self.train = train
        self.test = test
        self.valid = valid
        self.optim = optim
        self.batch_size = batch_size
        self.negative_count = negative_count
        self.ranker_valid = evaluate.Ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]), kb_data=valid.kb,
                                      expand_mode=expand_mode, filter_method=filter_method,
                                      flag_additional_filter=flag_additional_filter, load_to_gpu=load_to_gpu)
        self.ranker_test = evaluate.Ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]), kb_data=test.kb,
                                        expand_mode=expand_mode, filter_method=filter_method,
                                        flag_additional_filter=flag_additional_filter, load_to_gpu=load_to_gpu)
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid_m": {"mrr": 0.0}, "test_m": {"mrr": 0.0},
                                  "valid_e2": {"mrr": 0.0}, "test_e2": {"mrr": 0.0},
                                  "valid_e1": {"mrr": 0.0}, "test_e1": {"mrr": 0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []
        self.scheduler = scheduler

        self.debug = debug
        self.load_to_gpu=load_to_gpu
        self.time_neg_samples = time_neg_samples

        # self.normalize_time=None
        # if(self.scoring_function.__class__.__name__=='time_complex'): #for hyTE model
        #     self.normalize_time=True

        print("Using regularization_coefficient[:", regularization_coefficient)

        self.use_time_facts = use_time_facts
        if self.use_time_facts:
            print("Training with dummy time facts, loss function margin-pairwise")
            self.time_loss = losses.margin_pairwise_loss(margin=time_loss_margin)

        self.predict_time = predict_time
        self.time_args = time_args
        if self.predict_time:
            print("Time evaluation set to true.")

        self.flag_add_reverse = flag_add_reverse

    def step(self):
        s, r, o, ns, no = [], [], [], [], []

        if self.negative_count == 0:  # use all ent as neg sample
            ns = None
            no = None
            s, r, o, t, _, _ = self.train.tensor_sample(self.batch_size, 1)
        else:
            s, r, o, t, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)

        # print("Data point!: s, r, o, t", s,r,o,t)
        # print("Data point shape!: s:{}, r:{}, o:{}, t:{}, ns:{}, no:{}", s.shape,r.shape,o.shape,t.shape,ns.shape,no.shape)

        flag = random.randint(1, 10001)
        if flag > 9950:
            flag_debug = 1
        else:
            flag_debug = 0

        if flag_debug:
            fp = self.scoring_function(s, r, o, t, flag_debug=flag_debug + 1)
            fno = self.scoring_function(s, r, no, t, flag_debug=flag_debug + 1)
            fns = self.scoring_function(ns, r, o, t, flag_debug=flag_debug + 1)
            # fnt = self.scoring_function(s, r, o, None, flag_debug=flag_debug+1)##
        else:
            fp = self.scoring_function(s, r, o, t, flag_debug=0)
            fno = self.scoring_function(s, r, no, t, flag_debug=0)
            fns = self.scoring_function(ns, r, o, t, flag_debug=0)
            # fnt = self.scoring_function(s, r, o, None, flag_debug=0)##

        '''
        fp = self.scoring_function(s, r, o)
        fns = self.scoring_function(ns, r, o)
        fno = self.scoring_function(s, r, no)
        '''
        if self.flag_add_reverse==0:
            if self.negative_count == 0:  # use all ent as neg sample
                loss = self.loss(s, fns) + self.loss(o, fno)
            else:  # for subset neg sample
                loss = self.loss(fp, fns) + self.loss(fp, fno)
        else:
            if self.negative_count == 0:  # use all ent as neg sample
                loss = self.loss(o, fno) 
            else:  # for subset neg sample
                loss = self.loss(fp, fno) 


        # ---time negative sampling---#
        if self.time_neg_samples:
            print("**Time negative samples")
            nt = []
            fnt = self.scoring_function(s, r, o, None, flag_debug=flag_debug)  # only full softmax for now
            loss = loss + self.loss(t[:, 0, :], fnt)
        # ------------------------------#

        # ---if using time facts as constraints---#
        if self.use_time_facts and t is not None:  # train with dummy facts for time ordering

            t_start, t_end, positive_r, negative_r = get_time_facts(t, r)

            # pdb.set_trace()
            fpt = self.scoring_function.time_forward(t_start, positive_r, t_end)
            fnt = self.scoring_function.time_forward(t_end, positive_r, t_end)

            loss = loss + 0.2 * self.time_loss(fpt, fnt)

            # fpt=self.scoring_function.time_forward(t_end, negative_r, t_start)
            # fnt=self.scoring_function.time_forward(t_start, negative_r, t_end)

        # ----------------------------------------#

        if self.regularization_coefficient is not None:
            # '''
            reg = self.regularizer(s, r, o,
                                   t)  # , reg_val=3) #+ self.regularizer(ns, r, o) + self.regularizer(s, r, no)

            if self.use_time_facts and t is not None:
                t_start, t_end, positive_r, negative_r = get_time_facts(t, r)
                reg += self.scoring_function.time_regularizer(t_start, positive_r, t_end)

            # reg = reg / (self.batch_size * self.scoring_function.embedding_dim)
            loss += self.regularization_coefficient * reg
            # '''
            '''
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
            '''
        else:
            reg = None

        x = loss.item()
        rg = reg.item() if reg is not None else 0
        self.optim.zero_grad()
        loss.backward()
        if self.gradient_clip is not None:
            # print("gradient_clip:",self.gradient_clip)
            # for name, param in self.scoring_function.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            # torch.nn.utils.clip_grad_norm_(self.scoring_function.parameters(), self.gradient_clip)
            pass

        self.optim.step()

        debug = ""

        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def save_state(self, mini_batches, valid_score, test_score):
        state = dict()

        # -------- #
        state['datamap'] = self.train.kb.datamap  # save datamap as well, useful for analysis later on
        state['model_arguments'] = self.scoring_function_arguments
        # ------- #

        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches * self.batch_size / self.train.kb.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        state['valid_score_e2'] = valid_score['e2']
        state['test_score_e2'] = test_score['e2']
        state['valid_score_e1'] = valid_score['e1']
        state['test_score_e1'] = test_score['e1']
        state['valid_score_m'] = valid_score['m']
        state['test_score_m'] = test_score['m']

        state['valid_score_t'] = valid_score['t']
        state['test_score_t'] = test_score['t']

        state['valid_score_r'] = valid_score['r']
        state['test_score_r'] = test_score['r']

        # --not needed, but keeping for backward compatibility-- #
        state['entity_map'] = self.train.kb.datamap.entity_map
        state['reverse_entity_map'] = self.train.kb.datamap.reverse_entity_map
        state['relation_map'] = self.train.kb.datamap.relation_map
        state['reverse_relation_map'] = self.train.kb.datamap.reverse_relation_map
        # --------------- #

        # state['additional_params'] = self.train.kb.additional_params
        state['nonoov_entity_count'] = self.train.kb.nonoov_entity_count

        filename = os.path.join(self.save_directory,
                                "epoch_%.1f_val_%5.2f_%5.2f_%5.2f_test_%5.2f_%5.2f_%5.2f.pt" % (state['epoch'],
                                                                                                state['valid_score_e2'][
                                                                                                    'mrr'],
                                                                                                state['valid_score_e1'][
                                                                                                    'mrr'],
                                                                                                state['valid_score_m'][
                                                                                                    'mrr'],
                                                                                                state['test_score_e2'][
                                                                                                    'mrr'],
                                                                                                state['test_score_e1'][
                                                                                                    'mrr'],
                                                                                                state['test_score_m'][
                                                                                                    'mrr']))

        # torch.save(state, filename)
        def print_tensor(data):
            data2 = {}
            for key in data:
                data2[key] = round(data[key].tolist(), 4)
            return str(data2)

        try:
            if state['valid_score_m']['mrr'] >= self.best_mrr_on_valid["valid_m"]["mrr"]:
                print("Best Model details:\n", "valid_m", str(state['valid_score_m']), "\n", "test_m",
                      str(state["test_score_m"]), "\n\n",
                      "valid", str(state['valid_score_e2']), "\n", "test", str(state["test_score_e2"]), "\n\n",
                      "valid_e1", str(state['valid_score_e1']), "\n", "test_e1", str(state["test_score_e1"]), "\n\n",
                      "valid_r", str(state['valid_score_r']), "\n", "test_r", str(state["test_score_r"]), "\n\n",
                      "valid_t", str(state['valid_score_t']), "\n", "test_t", str(state["test_score_t"]), "\n")
                best_name = os.path.join(self.save_directory, "best_valid_model.pt")
                self.best_mrr_on_valid = {"valid_m": state['valid_score_m'], "test_m": state["test_score_m"],
                                          "valid": state['valid_score_e2'], "test": state["test_score_e2"],
                                          "valid_e1": state['valid_score_e1'], "test_e1": state["test_score_e1"],
                                          "valid_r": state['valid_score_r'], "test_r": state["test_score_r"],
                                          "valid_t": state['valid_score_t'], "test_t": state["test_score_t"]}

                if os.path.exists(best_name):
                    os.remove(best_name)
                torch.save(state, best_name)  # os.symlink(os.path.realpath(filename), best_name)
        except:
            utils.colored_print("red", "unable to save model")

    def load_state(self, state_file):
        state = torch.load(state_file)
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))
        self.scoring_function.load_state_dict(state['model_weights'])
        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current ' +
                                           'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches']

    def compute_charcnn_embed(self, batch_size=200):
        ent_cnt = len(self.train.kb.datamap.entity_map)

        print("Precomputing charCNN embeddings")
        with torch.no_grad():
            for i in range(0, ent_cnt, batch_size):
                utils.print_progress_bar(i, ent_cnt)
                inp = self.train.kb.charcnn_packaged([numpy.arange(i, min(i + batch_size, ent_cnt))])
                self.scoring_function.compute_char_embeddings(i, i + batch_size, inp[0])

        print("charCNN embeddings computed")

    def start(self, steps=50, batch_count=(20, 10), mb_start=0, logs_dir="", predict_time=0,
              time_prediction_method='greedy-coalescing', predict_rel=0):
        start = time.time()
        losses = []
        count = 0

        # CPU
        # self.scoring_function=self.scoring_function.cpu()

        '''
        self.scoring_function.eval()
        if(self.scoring_function.__class__.__name__.endswith('charCNN')): #precompute charcnn embeddings for all ent
            self.compute_charcnn_embed()
        
        valid_score = evaluate.evaluate("valid", self.ranker_valid, self.valid.kb, self.eval_batch, predict_rel = predict_rel,
                                        verbose=self.verbose, hooks=self.hooks, load_to_gpu=self.load_to_gpu, flag_add_reverse=self.flag_add_reverse)
        
        test_score = evaluate.evaluate("test ", self.ranker_test, self.test.kb, self.eval_batch, predict_rel = predict_rel,
                                           verbose=self.verbose, hooks=self.hooks, load_to_gpu= self.load_to_gpu, flag_add_reverse=self.flag_add_reverse)
        if self.predict_time:
            time_evaluate(self.scoring_function,self.valid.kb, self.test.kb, time_args=self.time_args)

        #sys.exit(1)
        self.scoring_function.train()
        #'''


        print("Starting training")

        for i in range(mb_start, steps):
            l, reg, debug = self.step()

            # print("REG:",reg)

            losses.append(l)
            suffix = ("| Current Loss %8.4f | " % l) if len(losses) != batch_count[0] else "| Average Loss %8.4f | " % \
                                                                                           (numpy.mean(losses))
            suffix += "reg %6.3f | time %6.0f ||" % (reg, time.time() - start)
            suffix += debug
            prefix = "Mini Batches %5d or %5.1f epochs" % (i + 1, i * self.batch_size / self.train.kb.facts.shape[0])
            utils.print_progress_bar(len(losses), batch_count[0], prefix=prefix, suffix=suffix)

            # print("Pairwise model weights sub:", self.scoring_function.pairwise_model.W_sub.data)
            # print("Pairwise model weights sub:", self.scoring_function.pairwise_model.scoring_gadget['subject'].
            #       mean_r_r)

            if len(losses) >= batch_count[0]:
                count += 1


                losses = []
                if count == batch_count[1]:

                    if (self.scoring_function.__class__.__name__.endswith(
                            'charCNN')):  # precompute charcnn embeddings for all ent
                        self.compute_charcnn_embed()

                    self.scoring_function.eval()

                    # print("Pairwise model weights sub:", self.scoring_function.pairwise_model_dict["start-start"].W_sub.data)

                    valid_score = evaluate.evaluate("valid", self.ranker_valid, self.valid.kb, self.eval_batch,
                                                    predict_rel=predict_rel, verbose=self.verbose, hooks=self.hooks, load_to_gpu=self.load_to_gpu, 
                                                    flag_add_reverse=self.flag_add_reverse)
                    test_score = evaluate.evaluate("test ", self.ranker_test, self.test.kb, self.eval_batch,
                                                   predict_rel=predict_rel, verbose=self.verbose, hooks=self.hooks, load_to_gpu=self.load_to_gpu,
                                                   flag_add_reverse=self.flag_add_reverse)

                    if self.predict_time:
                        time_evaluate(self.scoring_function, self.valid.kb, self.test.kb, time_args=self.time_args)


                    self.scoring_function.train()
                    self.scheduler.step(valid_score['m']['mrr'])  # Scheduler to manage learning rate added
                    count = 0
                    print()
                    self.save_state(i, valid_score, test_score)
        print()
        print("Ending")
        # print(self.best_mrr_on_valid["valid_m"])
        # print(self.best_mrr_on_valid["test_m"])
        print(self.best_mrr_on_valid)
