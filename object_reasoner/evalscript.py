from sklearn.metrics import classification_report, accuracy_score
#import pprint
import os
import json
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


def eval_KMi(ReasonerObj, args):
    """
    Used for KMi test set when all classes are known, based on scikit-learn
    """
    print("Class-wise test results \n")
    y_pred = ReasonerObj.predictions[:, 0, 0].astype('int').astype('str').tolist()
    with open(os.path.join(args.test_base, 'class_to_index.json')) as jf:
        allclasses = json.load(jf)
        backbook = dict((v, k) for k, v in allclasses.items())  # swap keys with indices

    print(classification_report(ReasonerObj.labels, y_pred))
    print(accuracy_score(ReasonerObj.labels, y_pred))
    """
    pp = pprint.PrettyPrinter(indent=4)
    cdict = classification_report(ReasonerObj.labels, y_pred, output_dict=True)  # , target_names=allclasses))
    remapped_dict = dict(
        (backbook[k], v) for k, v in cdict.items() if k not in ['accuracy', 'macro avg', 'weighted avg'])
    pp.pprint(remapped_dict)
    """
