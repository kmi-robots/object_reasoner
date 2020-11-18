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
    Reasonerobj.predictions = [t for i, t in enumerate(Reasonerobj.predictions) if i in test1_index]
    predictions2 = [t for i, t in enumerate(Reasonerobj.predictions) if i in test2_index]
    if basemethod =='two-stage':
        Reasonerobj.predictions_B = [t for i, t in enumerate(Reasonerobj.predictions_B) if i in test1_index]
        predictions_B2= [t for i, t in enumerate(Reasonerobj.predictions_B) if i in test2_index]
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
    epsilon_set = []
    for classl in classlist:
        mean_predwise1,mean_predwise2 = [],[]
        if subsample_preds_algo2 is None: #only one baseline algorithm
            for pred in subsample_preds_algo1:
                try:
                    mean_predwise1.append(statistics.mean([score for l_, score in pred if l_ == classl]))
                except: continue
            epsilon_set.append((classl, statistics.mean(mean_predwise1), None))
        else:
            for pred,pred2 in list(zip(subsample_preds_algo1,subsample_preds_algo2)):
                try:
                    mean_predwise1.append(statistics.mean([score for l_,score in pred if l_ ==classl]))
                    mean_predwise2.append(statistics.mean([score for l_,score in pred2 if l_ ==classl]))
                except: continue
            epsilon_set.append((classl,statistics.mean(mean_predwise1),statistics.mean(mean_predwise2)))

    return epsilon_set
