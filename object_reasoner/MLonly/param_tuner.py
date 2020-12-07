"""
This script recommends a confidence threshold for the Euclidean distance
based on matching a random subsample of test embeddings from test set against the support set.
This distance threshold can be then used as starting point for
fine-tuning on the test set (i.e., in the ML prediction selection model under cli.py)
"""

from sklearn.model_selection import StratifiedKFold
import statistics

def subsample(Reasonerobj, test1_index, test2_index, basemethod):
    # generate test splits so that all contain the same distribution of classes, or as close as possible
    allclasses = Reasonerobj.mapper.values()
    #retain larger split as test set, smaller split is for tuning the epsilon params
    Reasonerobj.labels = [lbl for i, lbl in enumerate(Reasonerobj.labels) if i in test1_index]
    fullpredictions,fullpredictions2 = Reasonerobj.predictions.copy(), Reasonerobj.predictions.copy()
    if Reasonerobj.set == 'lab':
        Reasonerobj.predictions = fullpredictions[test1_index]
        predictions2 = fullpredictions2[test2_index]
        Reasonerobj.predictions_B, predictions_B2 = None, None
    else: #predictions are in a list
        Reasonerobj.predictions = [pred for i, pred in enumerate(fullpredictions) if i in test1_index]
        predictions2 = [pred for i, pred in enumerate(fullpredictions2) if i in test2_index]
        if basemethod =='two-stage':
            fullpredictionsB, fullpredictionsB2 = Reasonerobj.predictions_B.copy(),Reasonerobj.predictions_B.copy()
            Reasonerobj.predictions_B = [pred for i, pred in enumerate(fullpredictionsB) if i in test1_index]
            predictions_B2 = [pred for i, pred in enumerate(fullpredictionsB2) if i in test2_index]
        else: Reasonerobj.predictions_B, predictions_B2 = None, None
    if Reasonerobj.set == 'arc':
        Reasonerobj.tsamples = [s for i, s in enumerate(Reasonerobj.tsamples) if i in test1_index]
    else: Reasonerobj.tsamples = None

    Reasonerobj.dimglist = [imge for i, imge in enumerate(Reasonerobj.dimglist) if i in test1_index]
    Reasonerobj.imglist = [imge for i, imge in enumerate(Reasonerobj.imglist) if i in test1_index]
    Reasonerobj.epsilon_set = estimate_epsilon(predictions2, predictions_B2, allclasses)
    return Reasonerobj

def estimate_epsilon(subsample_preds_algo1, subsample_preds_algo2, classlist):
    """
    Input:  - predictions on test subset by ML algorithm 1
            - predictions on test subset by ML algorithm 2
            - list of N classes
    Output: a 3xN list, with values of the epsilon param for each class and for each algorithm
            + indication of the class label those value refer to
    """
    min_classwise1, min_classwise2 = [],[]
    epsilon_set= []
    for classl in classlist:
        min_predwise1,min_predwise2 = [],[]
        if subsample_preds_algo2 is None: #only one baseline algorithm
            for pred in subsample_preds_algo1:
                try:
                    min_predwise1.append(min([score for l_, score in pred if l_ == classl]))
                except: continue
        else:
            for pred,pred2 in list(zip(subsample_preds_algo1,subsample_preds_algo2)):
                try:
                    min_predwise1.append(min([score for l_,score in pred if l_ ==classl]))
                    min_predwise2.append(min([score for l_,score in pred2 if l_ ==classl]))
                except: continue
            min_classwise2.append(min(min_predwise2))
        min_classwise1.append(min(min_predwise1))
    if subsample_preds_algo2 is None: epsilon_set= (statistics.mean(min_classwise1),None)
    else: epsilon_set = (statistics.mean(min_classwise1),statistics.mean(min_classwise2))
    return epsilon_set
