import torch
import os
import numpy
import argparse

import pdb

def smooth_iou_score(pred_start, pred_end, gold_start, gold_end, delta=0):
	'''
	params-
	pred_start: [n x 1] tensor, predicted start time
	pred_end: [n x 1] tensor, predicted end time
	gold_start: [n x 1] tensor, gold start time
	gold_end: [n x 1] tensor, gold end time
	delta: for smoothing, 0 means no smoothing
	returns-
	[n x 1] tensor indicating smooth iou for each
	'''
	t_pred_max, t_pred_min= pred_end, pred_start
	t_gold_max, t_gold_min= gold_end, gold_start


	numerator   = torch.min(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.max(t_pred_min.squeeze(), t_gold_min.squeeze()) 
	denomerator = torch.max(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.min(t_pred_min.squeeze(), t_gold_min.squeeze())

	# non_zero_inter = (numerator > 0).type("torch.DoubleTensor").nonzero()
	# zero_inter = (numerator <= 0).type("torch.DoubleTensor").nonzero()        

	#--IMPORTANT!- as (t,t) interval length is counted 0---#
	numerator+=1
	denomerator+=1
	#--------------#

	num_facts=len(pred_start)

	iou_score = torch.zeros(num_facts)
	delta = 0 #smooth IOU #1e-1

	non_zero_inter = (numerator > 0).type("torch.DoubleTensor").nonzero()
	if non_zero_inter.shape[0]:
		non_zero_inter = non_zero_inter.squeeze()
		score = numerator/(denomerator+1e-8).type('torch.FloatTensor').squeeze()
		iou_score.scatter_(0,non_zero_inter,score[non_zero_inter]+delta)

	zero_inter = (numerator < 0).type("torch.DoubleTensor").nonzero()        
	if zero_inter.shape[0]:
		zero_inter = zero_inter.squeeze()
		score = delta*torch.exp(-torch.sqrt(numerator**2)).type('torch.FloatTensor')
		iou_score.scatter_(0,zero_inter,score)#0)
	#-------------------#

	# print(len(iou_score))
	# print(len(iou_score.nonzero()))
	# print(torch.mean(iou_score))

	return iou_score


def aeiou_score(pred_start, pred_end, gold_start, gold_end, delta=0):
	'''
	computes aeiou score (affinity enhanced iou) defined as max{1, overlap_time}/(minimum span that covers both)
	params-
	pred_start: [n x 1] tensor, predicted start time
	pred_end: [n x 1] tensor, predicted end time
	gold_start: [n x 1] tensor, gold start time
	gold_end: [n x 1] tensor, gold end time
	returns-
	[n x 1] tensor indicating aeiou for each
	'''
	t_pred_max, t_pred_min= pred_end, pred_start
	t_gold_max, t_gold_min= gold_end, gold_start


	numerator   = torch.min(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.max(t_pred_min.squeeze(), t_gold_min.squeeze()) 
	denomerator = torch.max(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.min(t_pred_min.squeeze(), t_gold_min.squeeze())

	#--IMPORTANT!- as (t,t) interval length is counted 0---#
	numerator+=1
	denomerator+=1
	#--------------#

	#---numerator=max{1, intersection} for aeiou----#
	numerator= torch.max(numerator, torch.ones(numerator.shape)).float()
	#--------#

	num_facts=len(pred_start)

	iou_score = torch.zeros(num_facts)
	delta = 0 #smooth IOU #1e-1

	non_zero_inter = (numerator > 0).type("torch.DoubleTensor").nonzero()
	if non_zero_inter.shape[0]:
		non_zero_inter = non_zero_inter.squeeze()
		score = numerator/(denomerator+1e-8).type('torch.FloatTensor').squeeze()
		iou_score.scatter_(0,non_zero_inter,score[non_zero_inter]+delta)

	zero_inter = (numerator < 0).type("torch.DoubleTensor").nonzero()        
	if zero_inter.shape[0]:
		zero_inter = zero_inter.squeeze()
		score = delta*torch.exp(-torch.sqrt(numerator**2)).type('torch.FloatTensor')
		iou_score.scatter_(0,zero_inter,score)#0)
	#-------------------#

	# print(len(iou_score))
	# print(len(iou_score.nonzero()))
	# print(torch.mean(iou_score))

	return iou_score



def tac_score(pred_start, pred_end, gold_start, gold_end, delta=0):
	'''
	computes tac score  (0.5)*[(1/(1+|gold_start-pred_start|)) + (1/(1+|gold_end-pred_end|))]
	params-
	pred_start: [n x 1] tensor, predicted start time
	pred_end: [n x 1] tensor, predicted end time
	gold_start: [n x 1] tensor, gold start time
	gold_end: [n x 1] tensor, gold end time
	returns-
	[n x 1] tensor indicating tac for each
	'''
	t_pred_max, t_pred_min= pred_end, pred_start
	t_gold_max, t_gold_min= gold_end, gold_start

	min_diff = torch.reciprocal(1+torch.abs(t_pred_min - t_gold_min))
	max_diff = torch.reciprocal(1+torch.abs(t_pred_max - t_gold_max))

	tac_score_val = 0.5 * (min_diff+max_diff)

	return tac_score_val


def giou_score(pred_start, pred_end, gold_start, gold_end, delta=0):
	'''
	computes giou score (affinity enhanced iou) defined as 0.5 * iou_score - ((minimum distance between two interval --hole)/(minimum span that covers both))
	params-
	pred_start: [n x 1] tensor, predicted start time
	pred_end: [n x 1] tensor, predicted end time
	gold_start: [n x 1] tensor, gold start time
	gold_end: [n x 1] tensor, gold end time
	returns-
	[n x 1] tensor indicating aeiou for each
	'''
	t_pred_max, t_pred_min= pred_end, pred_start
	t_gold_max, t_gold_min= gold_end, gold_start

	iou_score = smooth_iou_score(pred_start, pred_end, gold_start, gold_end, delta=0)

	t_pred_max=torch.max(t_pred_min, t_pred_max) #just in case


	# pdb.set_trace()

	numerator   = torch.min(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.max(t_pred_min.squeeze(), t_gold_min.squeeze())
	denomerator = torch.max(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.min(t_pred_min.squeeze(), t_gold_min.squeeze())




	#--IMPORTANT!- as (t,t) interval length is counted 0---#
	numerator+=1
	denomerator+=1
	#--------------#

	numerator[numerator>0]=0

	numerator[numerator<0]*=-1


	# x= (numerator/(denomerator+1e-8).type('torch.FloatTensor').squeeze())
	# incorrect= x>1
	# idx=incorrect[0]
	# print("predicted max:{}, min:{}".format(t_pred_max[idx], t_pred_min[idx]))
	# print("gold max:{}, min:{}".format(t_gold_max[idx], t_gold_min[idx]))
	# pdb.set_trace()


	giou_score_val = 1 + iou_score - (numerator/(denomerator+1e-8).type('torch.FloatTensor').squeeze())

	# try:
	# 	assert(len(torch.nonzero(giou_score_val < 0)) == 0)
	# except Exception as e:
	# 	pdb.set_trace()

	return 0.5*giou_score_val


def precision_score(pred_start, pred_end, gold_start, gold_end, delta=0):
	'''
	computes precision, defined as fraction of hull(pred, gold) included in gold.
	params-
	pred_start: [n x 1] tensor, predicted start time
	pred_end: [n x 1] tensor, predicted end time
	gold_start: [n x 1] tensor, gold start time
	gold_end: [n x 1] tensor, gold end time
	returns-
	[n x 1] tensor indicating precision for each
	'''
	t_pred_max, t_pred_min= pred_end, pred_start
	t_gold_max, t_gold_min= gold_end, gold_start


	# numerator   = torch.min(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.max(t_pred_min.squeeze(), t_gold_min.squeeze()) 
	numerator   = t_gold_max.squeeze() - t_gold_min.squeeze() # gold  
	denomerator = torch.max(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.min(t_pred_min.squeeze(), t_gold_min.squeeze()) # hull 


	#--IMPORTANT!- as (t,t) interval length is counted 0---#
	numerator+=1
	denomerator+=1
	#--------------#

	precision_score= numerator/denomerator

	return precision_score


def recall_score(pred_start, pred_end, gold_start, gold_end, delta=0):
	'''
	computes precision, defined as fraction of hull(pred, gold) included in gold.
	params-
	pred_start: [n x 1] tensor, predicted start time
	pred_end: [n x 1] tensor, predicted end time
	gold_start: [n x 1] tensor, gold start time
	gold_end: [n x 1] tensor, gold end time
	returns-
	[n x 1] tensor indicating precision for each
	'''
	t_pred_max, t_pred_min= pred_end, pred_start
	t_gold_max, t_gold_min= gold_end, gold_start


	numerator   = torch.min(t_pred_max.squeeze(), t_gold_max.squeeze()) - torch.max(t_pred_min.squeeze(), t_gold_min.squeeze()) 
	# numerator   = t_gold_max.squeeze() - t_gold_min.squeeze() # gold  
	denomerator = t_gold_max.squeeze() - t_gold_min.squeeze() # hull 


	#--IMPORTANT!- as (t,t) interval length is counted 0---#
	numerator+=1
	denomerator+=1
	#--------------#

	intersecting= numerator > 0

	# pdb.set_trace()	
	recall_score= (numerator/denomerator)* intersecting.float() 

	return recall_score

