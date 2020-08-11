import kb
import data_loader
import trainer
import torch
import losses
import models
import argparse
import os
import datetime
import json
import utils
import sys
import torch.optim.lr_scheduler as lr_scheduler
import re
import pdb
import numpy as np
import torch

from time_prediction.evaluate import evaluate as time_evaluate
import evaluate
import pprint

# set random seeds
torch.manual_seed(32)
np.random.seed(12)

has_cuda = torch.cuda.is_available()
#has_cuda=False
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")


def init_model(model_name, model_arguments, datamap, ktrain, eval_batch_size, flag_time_smooth=None, regularizer=None,
               expand_mode='None', flag_add_reverse=None, batch_norm=False, has_cuda=True):
    """
    Initializes model with appropriate arguments.
    ktrain, eval_batch_size needed for pairwise models.
    """

    # --set model arguments-- #
    model_arguments['entity_count'] = len(datamap.entity_map)
    if datamap.use_time_interval:
        model_arguments['timeInterval_count'] = len(datamap.year2id)
        print("BIN: to year-pair", datamap.year2id)
    else:
        model_arguments['timeInterval_count'] = len(datamap.dateYear2id)  # intervalId2dateYears)#year2id)#dateYear2id
        print("Number of timestamps: ", len(datamap.dateYear2id))
        print(list(datamap.dateYear2id.items())[-10:])

    if flag_time_smooth:
        model_arguments["flag_time_smooth"] = flag_time_smooth

    if regularizer:
        print("Using reg ", regularizer)
        model_arguments['reg'] = regularizer

    if flag_add_reverse:
        model_arguments['relation_count'] = len(datamap.relation_map) * 2
        model_arguments['flag_add_reverse'] = flag_add_reverse
        # print("Using reg 3")
        # model_arguments['reg'] = 3
    elif expand_mode=='start-end-diff-relation':
        #TODO: modify datamap to store 2*num_relations
        # (instead of only changing the model argument)
        model_arguments['relation_count'] = len(datamap.relation_map) * 2
    else:
        model_arguments['relation_count'] = len(datamap.relation_map)

    model_arguments['batch_norm'] = batch_norm

    if model_name in ['TimePlex']:
        model_arguments['train_kb'] = ktrain
        model_arguments['eval_batch_size'] = eval_batch_size


    # if model_name.startswith('time_order_constraint'):
    #     model_arguments['train_kb'] = ktrain

    # if model_name.startswith('Pairwise'):
    #     model_arguments['train_kb'] = ktrain
    #     model_arguments['eval_batch_size'] = eval_batch_size

    # if model_name.startswith('Test'):
    #     model_arguments['train_kb'] = ktrain
    #     model_arguments['eval_batch_size'] = eval_batch_size


    print("final model arguments", model_arguments)

    model_arguments['has_cuda'] = has_cuda
    # ------------- #

    # --init model-- #
    scoring_function = getattr(models, model_name)(**model_arguments)
    # --------------- #

    return scoring_function


def main(mode, dataset, dataset_root, save_dir, tflogs_dir, debug, model_name, model_arguments,
         loss_function,
         learning_rate,
         batch_size,
         regularization_coefficient, regularizer, gradient_clip, optimizer_name, max_epochs, negative_sample_count,
         hooks,
         eval_every_x_mini_batches, eval_batch_size, resume_from_save, introduce_oov, verbose, batch_norm, predict_rel,
         predict_time, time_args, time_neg_samples, expand_mode, flag_bin, flag_time_smooth,
         flag_additional_filter,
         filter_method, perturb_time, use_time_facts, time_loss_margin, dump_t_scores, save_text, save_time_results, patience, flag_add_reverse
         ):
    # --set arguments for different models-- #
    if model_name.startswith('TA'):
        use_time_tokenizer = True
    else:
        use_time_tokenizer = False
    print("Flag: use_time_tokenizer-", use_time_tokenizer)

    print("Flag: flag_add_reverse-", flag_add_reverse)
    # if re.search("_lx", model_name):# or model_name=="TimePlex":
    #     flag_add_reverse = 1
    # else:
    #     flag_add_reverse = 0
    # -------------------------------------- #

    if resume_from_save:
        map_location = None if has_cuda else 'cpu'
        model = torch.load(resume_from_save, map_location = map_location)

        datamap = model['datamap']  # load datamap from saved model
        saved_model_arguments = model['model_arguments']
        if model_arguments is not None:
            for key in model_arguments:
                saved_model_arguments[key] = model_arguments[key]

        model_arguments = saved_model_arguments
        print("model_arguments:", model_arguments)
        # model_arguments = model['model_arguments']  # load model_arguments for model init (argument ignored)

    else:
        # --for HyTE-like binning-- #
        if flag_bin:
            use_time_interval = True
            print("Using hyTE-like chunking\n")
        else:
            use_time_interval = False
        print("Flag: use_time_interval", use_time_interval)
        # ------------------------- #

        # build datamap
        datamap = kb.Datamap(dataset, dataset_root, use_time_interval)

        if introduce_oov:
            if not "<OOV>" in datamap.entity_map.keys():
                eid = len(datamap.entity_map)
                datamap.entity_map["<OOV>"] = eid
                datamap.reverse_entity_map[eid] = "<OOV>"
                datamap.nonoov_entity_count = datamap.entity_map["<OOV>"] + 1

    # ---create train/test/valid kbs for filtering (need to keep this same irrespective of model)--- #
    dataset_root_filter = './data/{}'.format(dataset)

    datamap_filter = kb.Datamap(dataset, dataset_root_filter, use_time_interval=False)

    ranker_ktrain = kb.kb(datamap_filter, os.path.join(dataset_root_filter, 'train.txt'))

    ranker_ktest = kb.kb(datamap_filter, os.path.join(dataset_root_filter, 'test.txt'),
                  add_unknowns=int(not (int(introduce_oov))))

    ranker_kvalid = kb.kb(datamap_filter, os.path.join(dataset_root_filter, 'valid.txt'),
                   add_unknowns=int(not (int(introduce_oov))))
    # --------------------------- #




    # ---create train/test/valid kbs--- #
    ktrain = kb.kb(datamap, os.path.join(dataset_root, 'train.txt'),
                   use_time_tokenizer=use_time_tokenizer)

    ktest = kb.kb(datamap, os.path.join(dataset_root, 'test.txt'),
                  add_unknowns=int(not (int(introduce_oov))),
                  use_time_tokenizer=use_time_tokenizer)

    kvalid = kb.kb(datamap, os.path.join(dataset_root, 'valid.txt'),
                   add_unknowns=int(not (int(introduce_oov))),
                   use_time_tokenizer=use_time_tokenizer)
    # --------------------------- #

    print("Train (no expansion)", ktrain.facts.shape)
    print("Test", ktest.facts.shape)
    print("Valid", kvalid.facts.shape)

    # print("dateYear2id", len(datamap.dateYear2id))
    # print("dateYear2id", datamap.dateYear2id)
    # print("intervalId2dateYears", len(datamap.intervalId2dateYears))

    if not eval_batch_size:
        eval_batch_size = max(40, batch_size * 2 * negative_sample_count // len(datamap.entity_map))

    # init model
    if resume_from_save:
        if 'eval_batch_size' in model_arguments:
            model_arguments['eval_batch_size'] = eval_batch_size
            
        scoring_function = getattr(models, model_name)(
            **model_arguments)  # use model_arguments from saved model, allowing those provided in command to be overridden
    else:
        scoring_function = init_model(model_name, model_arguments, datamap, ktrain, eval_batch_size, flag_time_smooth,
                                      regularizer, expand_mode, flag_add_reverse, batch_norm, has_cuda)

    if has_cuda:
        scoring_function = scoring_function.cuda()

    if mode == 'train':
        # expand data as needed
        if expand_mode != "None":
            ktrain.expand_data(mode=expand_mode)
            print("Expanded training data with mode= {}".format(expand_mode))
        else:
            print("Not expanding training data")

        print("Train (after expansion)", ktrain.facts.shape)

        # ---create dataloaders to be used when training--- #
        dltrain = data_loader.data_loader(ktrain, has_cuda, loss=loss_function, flag_add_reverse=flag_add_reverse,
                                          model=model_name, perturb_time=perturb_time)
        dlvalid = data_loader.data_loader(kvalid, has_cuda, loss=loss_function, #flag_add_reverse=flag_add_reverse,
                                          model=model_name)
        dltest = data_loader.data_loader(ktest, has_cuda, loss=loss_function, #flag_add_reverse=flag_add_reverse,
                                         model=model_name)
        # ------------------------------------------------ #

        # loss, optimiser, scheduler for training
        loss = getattr(losses, loss_function)()
        optim = getattr(torch.optim, optimizer_name)(scoring_function.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'max', factor=0.1, patience=patience, verbose=True)  # mrr tracking

        # init trainer and start training
        tr = trainer.Trainer(scoring_function, model_arguments, scoring_function.regularizer, loss, optim, dltrain,
                             dlvalid, dltest,
                             batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                             save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                             regularization_coefficient=regularization_coefficient, verbose=verbose,
                             scheduler=scheduler,
                             debug=debug, time_neg_samples=time_neg_samples, expand_mode=expand_mode,
                             flag_additional_filter=flag_additional_filter,
                             filter_method=filter_method, use_time_facts=use_time_facts,
                             time_loss_margin=time_loss_margin, predict_time=predict_time, time_args=time_args, flag_add_reverse = flag_add_reverse,
                             load_to_gpu=has_cuda)  # 0.01)
        if resume_from_save:
            mb_start = tr.load_state(resume_from_save)
        else:
            mb_start = 0
        max_mini_batch_count = int(max_epochs * ktrain.facts.shape[0] / batch_size)
        print("max_mini_batch_count: %d, eval_batch_size %d" % (max_mini_batch_count, eval_batch_size))
        tr.start(max_mini_batch_count, [eval_every_x_mini_batches // 20, 20], mb_start, tflogs_dir,
                 )

    elif mode == 'test':
        # if not eval_batch_size:
        #     eval_batch_size = max(40, batch_size * 2 * negative_sample_count // len(datamap.entity_map))

        # Load Model
        map_location = None if has_cuda else 'cpu'
        saved_model = torch.load(resume_from_save, map_location=map_location)  # note: resume_from_save is required for testing

        scoring_function.load_state_dict(saved_model['model_weights'])

        print("valid_score_m", saved_model['valid_score_m'])
        print("valid_score_e1", saved_model['valid_score_e1'])
        print("valid_score_e2", saved_model['valid_score_e2'])
        print("test_score_m", saved_model['test_score_m'])
        print("test_score_e1", saved_model['test_score_e1'])
        print("test_score_e2", saved_model['test_score_e2'])

        # '''
        # ---entity/relation prediction--- #
        print("Scores with {} filtering".format(filter_method))

        # ranker = evaluate.Ranker(scoring_function, kb.union([ktrain, kvalid, ktest]), kb_data=kvalid,
        #                          filter_method=filter_method, flag_additional_filter=flag_additional_filter,
        #                          expand_mode=expand_mode, load_to_gpu=has_cuda)
        ranker = evaluate.Ranker(scoring_function, kb.union([ranker_ktrain, ranker_kvalid, ranker_ktest]), kb_data=ranker_kvalid,
                                 filter_method=filter_method, flag_additional_filter=flag_additional_filter,
                                 expand_mode=expand_mode, load_to_gpu=has_cuda)


        valid_score = evaluate.evaluate("valid", ranker, kvalid, eval_batch_size,
                                        verbose=verbose, hooks=hooks, save_text=save_text,
                                        predict_rel=predict_rel, load_to_gpu=has_cuda, flag_add_reverse=flag_add_reverse)

        # ranker = evaluate.Ranker(scoring_function, kb.union([ktrain, kvalid, ktest]), kb_data=test,
        #                          filter_method=filter_method, flag_additional_filter=flag_additional_filter,
        #                          expand_mode=expand_mode, load_to_gpu=has_cuda)

        ranker = evaluate.Ranker(scoring_function, kb.union([ranker_ktrain, ranker_kvalid, ranker_ktest]), kb_data=ranker_ktest,
                                 filter_method=filter_method, flag_additional_filter=flag_additional_filter,
                                 expand_mode=expand_mode, load_to_gpu=has_cuda)
        test_score = evaluate.evaluate("test", ranker, ktest, eval_batch_size,
                                       verbose=verbose, hooks=hooks, save_text=save_text,
                                       predict_rel=predict_rel,
                                       load_to_gpu=has_cuda, flag_add_reverse=flag_add_reverse)

        print("Valid")
        pprint.pprint(valid_score)
        print("Test")
        pprint.pprint(test_score)
        # ------------------ #
        '''

        # '''
        # ---time prediction--- #
        utils.colored_print("yellow", "\nEvaluating on time prediction\n")

        # create test/valid kbs for subset of data (for which boths start end have been provided)
        ktest_sub = kb.kb(datamap, os.path.join(dataset_root, 'intervals/test.txt'),
                          add_unknowns=int(not (int(introduce_oov))),
                          use_time_tokenizer=use_time_tokenizer)

        kvalid_sub = kb.kb(datamap, os.path.join(dataset_root, 'intervals/valid.txt'),
                           add_unknowns=int(not (int(introduce_oov))),
                           use_time_tokenizer=use_time_tokenizer)

        if predict_time:
            time_evaluate(scoring_function, kvalid_sub, ktest_sub, time_args=time_args, dump_t_scores=dump_t_scores,
                          load_to_gpu=has_cuda, save_time_results= save_time_results)
        # '''
        # ------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        default='train',
                        nargs='?',
                        choices=('train', 'test'),
                        help='either train or test')

    parser.add_argument('-m', '--model', help="model name as in models.py", required=True)
    parser.add_argument('--model_type',
                        default='custom',
                        nargs='?',
                        choices=('time-point-random-sampling', 'time-boundary', 'custom'),
                        help='time-point-random-sampling (time point model, during training time randomly chosen from '
                             'start to end, during testing aggregate score (sum) over all time points) '
                             'OR time-boundary (start and end time only , different relations)'
                             'OR custom (use arguments given from command line only, this is default)'
                        )

    parser.add_argument('-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('--data_repository_root', required=False, default='data')

    # --arguments for mode = train only -- #
    parser.add_argument('-a', '--model_arguments', help="model arguments as in __init__ of "
                                                        "model (Excluding entity and relation count), embedding_dim "
                                                        "argument required for training. "
                                                        "This is a json string", required=False)
    parser.add_argument('-o', '--optimizer', required=False, default='Adagrad')
    parser.add_argument('-l', '--loss', help="loss function name as in losses.py",
                        required=False, default='crossentropy_loss')
    parser.add_argument('-r', '--learning_rate', required=False, default=1e-2, type=float)
    parser.add_argument('-g', '--regularization_coefficient', required=False,
                        type=float)  # changed required to False- 12/10/19
    parser.add_argument('-g_reg', '--regularizer', required=False, default=2.0, type=float)
    parser.add_argument('-c', '--gradient_clip', required=False, type=float)
    parser.add_argument('-e', '--max_epochs', required=False, type=int, default=1000)
    parser.add_argument('-b', '--batch_size', required=False, type=int, default=2000)
    parser.add_argument('-x', '--eval_every_x_mini_batches', required=False, type=int, default=2000)
    parser.add_argument('-y', '--eval_batch_size', required=False, type=int, default=0)
    parser.add_argument('-n', '--negative_sample_count', required=False, type=int, default=200)
    parser.add_argument('-s', '--save_dir', required=False)
    parser.add_argument('-u', '--resume_from_save', required=False)
    parser.add_argument('-v', '--oov_entity', required=False, type=int, default=1)
    parser.add_argument('-q', '--verbose', required=False, default=0, type=int)
    parser.add_argument('-z', '--debug', required=False, default=0, type=int)
    parser.add_argument('-bn', '--batch_norm', required=False, default=0, type=int)
    parser.add_argument('-msg', '--message', required=False)
    parser.add_argument('-bt', '--bin_time', required=False, default=0, type=int)
    parser.add_argument('-tsmooth', '--flag_time_smooth', help="value = 0: no smoothing, k is k path smoothing",
                        required=False, default=0, type=int)
    parser.add_argument('-tn', '--time_neg_samples', required=False, default=0, type=int)
    parser.add_argument('--perturb_time', required=False, default=0, type=int)

    parser.add_argument('--patience', required=False, default=3, type=int) # patience for learning rate scheduler
    parser.add_argument('-tf', '--tflogs_dir', required=False)  # for tensorboard logs
    # -----------------------------------#

    # --for inverse facts-- #
    parser.add_argument('--flag_add_reverse', required=False, default=0, type=int) # 1 if inverse facts are to be added
    # --------------------- #

    # --arguments for time prediction--#
    parser.add_argument('-pt', '--predict_time', required=False, default=0, type=int)
    parser.add_argument('--subset', required=False, help="whether time prediction is to be done only on subset",
                        default=1, type=int)
    parser.add_argument('--time_prediction_method',
                        default='greedy-coalescing',
                        nargs='?',
                        choices=('greedy-coalescing', 'start-end-exhaustive-sweep'),
                        help='inference method for time prediction')
    # -------------------------------- #

    parser.add_argument('-pr', '--predict_rel', required=False, default=0, type=int)

    parser.add_argument('-ed', '--expand_mode', help="Mode of expansion, can be all/ start-end-diff-relation/ None",
                        required=False, default="None")
    parser.add_argument('--filter_method',
                        help="filter method- time-interval/ start-time/ no-filter/ ignore-time/ time-str/ enumerate-time",
                        required=False, default='enumerate-time')
    parser.add_argument('--flag_additional_filter', required=False, default=0, type=int)
    parser.add_argument('--dump_t_scores',
                        help="if time scores are to be pickled, specify prefix for filename here (useful for "
                             "mode=test only)",
                        required=False, default=None)
    parser.add_argument('--save_time_results',
                        help="if time prediction results are to be pickled (score for each fact), specify prefix for filename here (useful for "
                             "mode=test only)",
                        required=False, default=None)

    parser.add_argument('--save_text',
                        help="if predictions are to be saved, specify prefix for filename here (useful for mode=test "
                             "only). "
                             "Should contain the name of dataset, for example (CX_WIKIDATA12k)",
                        required=False)

    parser.add_argument('-k', '--hooks', required=False, default="[]")

    # --for time facts experiment---#
    parser.add_argument('--use_time_facts', required=False, default=0, type=int)
    parser.add_argument('--time_loss_margin', required=False, default=5.0, type=float)
    # ------------------------------#

    arguments = parser.parse_args()

    arguments.hooks = json.loads(arguments.hooks)
    time_args = {'method': arguments.time_prediction_method}  # take dict as argument instead?

    # --appropriate arguments for each model type --#
    if arguments.model_type != 'custom':
        if arguments.model_type == 'time-point-random-sampling':
            arguments.perturb_time = 1
            arguments.expand_mode = 'None'

        elif arguments.model_type == 'time-boundary':
            arguments.perturb_time = 0
            arguments.expand_mode = 'start-end-diff-relation'
    # ------------------------------------------- #

    if arguments.mode == 'train':
        arguments.model_arguments = json.loads(arguments.model_arguments)

        if arguments.save_dir is None:
            arguments.save_dir = os.path.join("logs", "%s_%s_%s_run_on_%s_starting_from_%s" % (arguments.model,
                                                                                               '',#arguments.model_arguments,# str(arguments.model_arguments).replace('/', '_'),
                                                                                               arguments.loss,
                                                                                               arguments.dataset,
                                                                                               str(
                                                                                                   datetime.datetime.now())))
        log_folder = "./models/"

        arguments.save_dir = log_folder + arguments.save_dir

        if not arguments.debug:
            if not os.path.isdir(arguments.save_dir):
                print("Making directory (s) %s" % arguments.save_dir)
                os.makedirs(arguments.save_dir)
            else:
                utils.colored_print("yellow", "directory %s already exists" % arguments.save_dir)
            utils.duplicate_stdout(os.path.join(arguments.save_dir, "log.txt"))

        if arguments.tflogs_dir is None:
            arguments.tflogs_dir = arguments.save_dir
        else:
            arguments.tflogs_dir += datetime.datetime.now().strftime('_%d-%m-%y_%H.%M.%S')
            if not os.path.isdir(arguments.tflogs_dir):
                print("Making directory (s) %s" % arguments.tflogs_dir)
                os.makedirs(arguments.tflogs_dir)
            else:
                utils.colored_print("yellow", "directory %s already exists" % arguments.tflogs_dir)

    elif arguments.mode == 'test':
        if arguments.resume_from_save is None:
            parser.error("--mode test requires -u (saved model path)")

    print(arguments)
    print("User Message:: ", arguments.message)
    print("Command:: ", " ".join(sys.argv))
    dataset_root = os.path.join(arguments.data_repository_root, arguments.dataset)
    main(arguments.mode, arguments.dataset, dataset_root, arguments.save_dir, arguments.tflogs_dir, arguments.debug,
         arguments.model,
         arguments.model_arguments, arguments.loss,
         arguments.learning_rate, arguments.batch_size, arguments.regularization_coefficient,
         arguments.regularizer,
         arguments.gradient_clip,
         arguments.optimizer, arguments.max_epochs, arguments.negative_sample_count, arguments.hooks,
         arguments.eval_every_x_mini_batches, arguments.eval_batch_size, arguments.resume_from_save,
         arguments.oov_entity, arguments.verbose, arguments.batch_norm, arguments.predict_rel,
         arguments.predict_time,
         time_args, arguments.time_neg_samples, arguments.expand_mode, arguments.bin_time,
         arguments.flag_time_smooth, arguments.flag_additional_filter, arguments.filter_method,
         arguments.perturb_time,
         arguments.use_time_facts, arguments.time_loss_margin, arguments.dump_t_scores,
         arguments.save_text, arguments.save_time_results, arguments.patience, arguments.flag_add_reverse)
