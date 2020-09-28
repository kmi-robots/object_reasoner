from sklearn.metrics import classification_report, accuracy_score, dcg_score, ndcg_score, precision_score
import math
#import pprint
#import os
#import json
import numpy as np
"""
Evaluation script - top-1 eval accuracy
to reproduce results in https://github.com/andyzeng/arc-robot-vision/image-matching
"""

def eval_singlemodel(ReasonerObj):

    all_labels = ReasonerObj.labels
    knownclasses = [v['label'] for v in ReasonerObj.known.values()]
    newclasses = [v['label'] for v in ReasonerObj.novel.values()]
    all_preds = ReasonerObj.predictions[:,0,0].astype('int').astype('str').tolist()
    known_labels = [l for l in all_labels if l in knownclasses]
    new_labels = [l for l in all_labels if l in newclasses]

    all_sum = 0
    k_sum = 0
    n_sum = 0
    for p,l in zip(all_preds,all_labels):
        if p == l:
            all_sum +=1
            if l in knownclasses:
                k_sum +=1
            elif l in newclasses:
                n_sum +=1
    print("Mixed object accuracy: %f" % float(all_sum/len(all_preds)))
    print("Known object accuracy: %f" % float(k_sum / len(known_labels)))
    print("Novel object accuracy: %f" % float(n_sum / len(new_labels)))

    return

def eval_twostage():

    return


def eval_classifier(all_gt_labels, knownclasses, predicted):
    """
    similar to eval_singlemodel but
    can be called even without a ObjectReasoner object
    """
    all_gt_labels = [int(l.item()) for l in all_gt_labels]
    print(all_gt_labels)

    allclasses = set(all_gt_labels) # known + novel
    print(allclasses)
    newclasses = list(allclasses-knownclasses)
    print(newclasses)
    knownclasses = list(knownclasses)

    all_preds = [pred+1 for pred in predicted] # from 0-N to 1-N
    print(all_preds)
    known_labels = [l for l in all_gt_labels if l in knownclasses]
    new_labels = [l for l in all_gt_labels if l in newclasses]
    print(known_labels)
    print(new_labels)
    all_sum = 0
    k_sum = 0
    n_sum = 0
    for p, l in zip(all_preds, all_gt_labels):
        if p == l:
            all_sum += 1
            if l in knownclasses:
                k_sum += 1
            elif l in newclasses:
                n_sum += 1

    print("Mixed object accuracy: %f" % float(all_sum / len(all_preds)))
    print("Known object accuracy: %f" % float(k_sum / len(known_labels)))
    print("Novel object accuracy: %f" % float(n_sum / len(new_labels)))

    return


def eval_KMi(ReasonerObj, depth_aligned=False, K=None):
    """
    Used for KMi test set when all classes are known, based on scikit-learn
    If K is set, it looks at whether the correct answer appears in the top-K ranking
    if depth-aligned is True, evals only on those images which have an accurate (and non null) match for depth
    """
    # eval only on those with a depth img associated
    blacklist = [] #['524132', '409022', '240924', '741394', '109086', '041796', '036939']
    print("Class-wise test results \n")
    if K is None:
        # eval top-1 of each ranking
        y_pred = ReasonerObj.predictions[:, 0, 0].astype('int').astype('str').tolist()
        if depth_aligned:
            y_pred = [y for i,y in enumerate(y_pred)
                      if ReasonerObj.dimglist[i] is not None
                      and ReasonerObj.imglist[i].split('/')[-1].split('_')[-2] not in blacklist]
            y_true = [l for i,l in enumerate(ReasonerObj.labels)
                      if ReasonerObj.dimglist[i] is not None
                      and ReasonerObj.imglist[i].split('/')[-1].split('_')[-2] not in blacklist]
        else: #eval on full RGB test set
            y_true = ReasonerObj.labels
        print(classification_report(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))
    else:
        #eval top-K ranking (ranking quality metrics)
        y_pred = ReasonerObj.predictions[:,:K,0].astype('int').astype('str').tolist()
        y_true = ReasonerObj.labels
        rank_pred = []
        rank_true = []
        precisions = []
        ndcgs = []
        hits = 0
        IDCG = 0. #Ideal DCG
        for n in range(2,K+2):
            IDCG += float(1/math.log(n,2))
        for z, (ranking, gt_label) in enumerate(zip(y_pred, y_true)):
            if depth_aligned and (ReasonerObj.dimglist[z] is None \
                or ReasonerObj.imglist[z].split('/')[-1].split('_')[-2] in blacklist):
                continue
            pred_rank = [1 if r == gt_label else 0 for r in ranking]
            dis_scores = [float(1/math.log(i+2,2)) for i,r in enumerate(ranking) if r == gt_label]
            #true_rank = np.asarray([1 for r in range(K)])
            #rank_pred.append(pred_rank)
            #rank_true.append(true_rank)
            no_hits = pred_rank.count(1)
            precisions.append(float(no_hits/K))
            if no_hits >=1:
                hits+=1 # increment if at least one hit in the ranking
                nDCG = float(sum(dis_scores)/IDCG) # compute nDCG for ranking
                ndcgs.append(nDCG)
        print("Avg ranking Precision@%i: %f " % (K,float(sum(precisions)/len(precisions))))
        print("Avg Normalised DCG @%i: %f" % (K, float(sum(ndcgs)/len(precisions))))
        print("Hit ratio @%i: %f" % (K,float(hits/len(precisions))))
