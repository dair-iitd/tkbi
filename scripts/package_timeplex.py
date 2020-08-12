'''
Takes in the following argumets-
1, Recurrent gadget wt (optional)
2. Recurrent trained model path
3. Pairwise gadget wt (optional)
4. Pairwise model path
5. save as

And saves a TimePlex model with the 2 gadgets combined (filename provided in 'save as' argument)
This model can then directly be used for evaluation, using an evaluation command of the 
type listed in README.
This script can be used for hyper-tuning gadgets weights for the full TimePlex model.

Example-
python scripts/package_timeplex.py --rec_model_path ./models/yago15k_timeplex_rec/best_valid_model.pt --pairs_model_path ./models/yago15k_timeplex_pair
s/best_valid_model.pt --save_as ./models/yago15k_timeplex/model.pt
'''

import torch
import argparse

def package_TimePlex(rec_wt, pairs_wt, rec_model_path, pairs_model_path, save_as):
    # torch.load models, 
    rec_model=torch.load(rec_model_path)
    print("Loaded recurrent gadget from {}".format(rec_model_path))

    pairs_model=torch.load(pairs_model_path)
    print("Loaded pairs gadget from {}".format(pairs_model_path))

    # one combined model
    final_model = rec_model
    final_model['model_arguments']['pairs_wt'] = pairs_model['model_arguments']['pairs_wt']

    if rec_wt is not None:
        final_model['model_arguments']['recurrent_wt']=rec_wt

    if pairs_wt is not None:
        final_model['model_arguments']['pairs_wt']=pairs_wt

    print("Recurrent wt: ", final_model['model_arguments']['recurrent_wt'])
    print("Pairs wt: ", final_model['model_arguments']['pairs_wt'])


    for key,val in pairs_model['model_weights'].items():
        if key.startswith('pairs.'):
            final_model['model_weights'][key] = val

    # valid & test scores will now be different, so clear these fields
    final_model['valid_score_m']=None
    final_model['valid_score_e1']=None
    final_model['valid_score_e2']=None

    final_model['test_score_m']=None
    final_model['test_score_e1']=None
    final_model['test_score_e2']=None

    torch.save(final_model, save_as)
    print("Saved combined model in {}".format(save_as))



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        default='train',
                        nargs='?',
                        choices=('train', 'test'),
                        help='either train or test')

    parser.add_argument('--rec_wt', help="Weight of the recurrent gadget", required=False, type=float)
    parser.add_argument('--rec_model_path', help="Path to trained model (with recurrent)", required=True)

    parser.add_argument('--pairs_wt', help="Weight of the recurrent gadget", required=False, type=float)
    parser.add_argument('--pairs_model_path', help="Path to trained model (with pairs)", required=True)

    parser.add_argument('--save_as', help="Save combined model as this file", required=True)

    arguments = parser.parse_args()

    package_TimePlex(arguments.rec_wt, arguments.pairs_wt, arguments.rec_model_path, arguments.pairs_model_path, arguments.save_as)

