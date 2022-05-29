# Temporal Knowledge Base Completion: New Algorithms and Evaluation Protocols
This project contains Pytorch implementation of Temporal Knowledge Base Completion (TKBC) models [1]. The code has been developed at Indian Institute of Technology, Delhi (IIT Delhi). The TKBI models in this repository are trained over structured temporal knowledge bases like WIKIDATA12k, YAGO11k, ICEWS05-15, and ICEWS14. You can also add your own KB seamlessly.

[1] "[Temporal Knowledge Base Completion: New Algorithms and Evaluation Protocols](https://arxiv.org/abs/2005.05035)" Jain, Prachi*, Sushant Rathi*, Mausam and Soumen Chakrabarti. EMNLP 2020.

(* equal contribution)

Updated version to be uploaded soon.

# Requirements
```
python>=3.6
pytorch==1.0.0
```

# Dataset
Download the dataset from [here](https://zenodo.org/record/4286007#.X7uL8BMzYWo).

Commands-  
```
# Run from inside the repo dir

wget -O datasets.zip https://zenodo.org/record/4286007/files/share-tkbc-data.zip?download=1

unzip datasets.zip && mv share data
```


# Training commands-

## TimePlex (base):

```
##YAGO11k
python main.py -d YAGO11k -m TimePlex_base -a "{'embedding_dim':200, 'srt_wt':5.0, 'ort_wt':5.0, 'sot_wt':0.0, 'emb_reg_wt':0.03}" -l crossentropy_loss_AllNeg -r 0.1 -b 1500 -x 300 -n 0 -v 1 -q 0 -y 100 -g_reg 2 -g 1.0 --perturb_time 1 --mode train -e 100 --save_dir yago_timeplex_base

##WIKIDATA12k
python main.py -d WIKIDATA12k -m TimePlex_base -a "{'embedding_dim':200, 'srt_wt':5.0, 'ort_wt':5.0, 'sot_wt':5.0, 'emb_reg_wt':0.005}" -l crossentropy_loss_AllNeg -r 0.1 -b 1500 -x 300 -n 0 -v 1 -q 0 -y 100 -g_reg 2 -g 2.0 --perturb_time 1 --mode train --flag_add_reverse 1 -e 100 --save_dir wiki_timeplex_base


##ICEWS05-15
python main.py -d icews05-15 -m TimePlex_base -a "{'embedding_dim':200, 'srt_wt': 5.0, 'ort_wt': 5.0, 'sot_wt': 5.0, 'time_reg_wt':5.0, 'emb_reg_wt':0.005}" -l crossentropy_loss_AllNeg -r 0.1 -b 1000 -x 2000 -n 0 -v 1 -q 0 -y 500 -g_reg 2 -g 1.0 --filter_method time-str -e 250 --flag_add_reverse 1 --save_dir icews05-15_timeplex_base

##ICEWS14
python main.py -d icews14 -m TimePlex_base -a "{'embedding_dim':200, 'srt_wt': 5.0, 'ort_wt': 5.0, 'sot_wt': 5.0, 'time_reg_wt':1.0, 'emb_reg_wt':0.005}" -l crossentropy_loss_AllNeg -r 0.1 -b 1000 -x 2000 -n 0 -v 1 -q 0 -y 500 -g_reg 2 -g 1.0 --filter_method time-str -e 250 --flag_add_reverse 1 --save_dir icews14_timeplex_base
```

## TimePlex-

Once the base model has been trained, we can augment it with either pair/recurrent features.
To train with pair features- 
```
python main.py -d YAGO11k -m TimePlex -a "{'embedding_dim':200, 'model_path':'./models/yago_timeplex_base/best_valid_model.pt', 'pairs_wt':3.0, 'pairs_args':{'reg_wt':0.002}}" -l crossentropy_loss -r 0.05 -b 100 -x 300 -n 100 -v 1 -q 0 -y 40  -g 1.0 -bt 0 --patience 1 -e 2 --save_dir yago_timeplex
```

To train with recurrent features-

```
##YAGO11k-
python  main.py -d YAGO11k -m TimePlex -a "{'embedding_dim':200, 'model_path':'./models/yago_timeplex_base/best_valid_model.pt', 'recurrent_wt':5.0}" -l crossentropy_loss -r 1.0 -b 100 -x 600 -n 100 -v 1 -q 0 -y 40 -g_reg 2 -g 0.0 -bt 0 --patience 1 -e 10 --save_dir yago_timeplex

##WIKIDATA12k-
python main.py -d WIKIDATA12k -m TimePlex -a "{'embedding_dim':200, 'model_path':'./models/wiki_timeplex_base/best_valid_model.pt', 'recurrent_wt':5.0}" -l crossentropy_loss -r 0.1 -b 100 -x 300 -n 100 -v 1 -q 0 -y 40 -g_reg 2 -g 0.0 -bt 0 --patience 1 -e 2 --save_dir wiki_timeplex
```

# Evaluating trained models (for link and time prediction)-

(Note: To evaluate TimePlex_base models, replace `-m TimePlex` with `-m TimePlex_base` and `--resume_from_save` argument to base model path, for example `--resume_from_save "./models/icews14_timeplex_base/best_valid_model.pt"`)

For interval datasets-
```
## YAGO11k- 
python main.py -d YAGO11k -m TimePlex --resume_from_save "./models/yago_timeplex/best_valid_model.pt"  --mode test --predict_time 1 -y 40

## WIKIDATA12k- 
python main.py -d WIKIDATA12k -m TimePlex --resume_from_save "./models/wiki_timeplex/best_valid_model.pt"  --mode test --predict_time 1 -y 40
```

For event datasets-
```
## ICEWS05-15
python main.py -d icews05-15 -m TimePlex --resume_from_save "./models/icews05-15_timeplex/best_valid_model.pt"  --mode test --filter_method time-str -y 40 --flag_add_reverse 1 


## ICEWS14
python main.py -d icews14 -m TimePlex --resume_from_save "./models/icews14_timeplex/best_valid_model.pt"  --mode test --filter_method time-str -y 40 --flag_add_reverse 1 
```


# Results
Link Prediction scores:
![image](https://user-images.githubusercontent.com/20418382/120938804-5eefd400-c732-11eb-8a3b-91932d31d8af.png)
Link Prediction scores (2k dim model):

|      Dataset     | Wikidata12k |        |         | Yago11k |        |         | ICEWS05-15 |        |         | ICEWS14 |        |         |
|:----------------:|:-----------:|:------:|:-------:|:-------:|:------:|:-------:|:----------:|:------:|:-------:|:-------:|:------:|:-------:|
| Methods (2k dim) |     MRR     | HITS@1 | HITS@10 |   MRR   | HITS@1 | HITS@10 |     MRR    | HITS@1 | HITS@10 |   MRR   | HITS@1 | HITS@10 |
|  TIMEPLEX (BASE) |    32.68    |  22.03 |  52.52  |  18.93  |  11.58 |  31.52  |    66.14   |  57.07 |  82.4  |  62.00  |  53.49 |  77.48  |
|     TIMEPLEX     |    33.82    |  22.92 |  53.37  |  23.28  |  16.33 |  36.2  |    66.18   |  57.07 |  82.49  |  62.02  |  53.54 |  77.51  |

Time-interval prediction scores:
![Image-time-interval-prediction-performance](https://github.com/dair-iitd/tkbi/blob/master/images/Time_scores.png?raw=true)



