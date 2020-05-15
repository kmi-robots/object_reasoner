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


def eval_imprinted(all_gt_labels, knownclasses, predicted_proba):
    """
    similar to eval_singlemodel but
    can be called even without a ObjectReasoner object
    """
    allclasses = set([int(l_tensor.item()) for l_tensor in all_gt_labels]) # known + novel
    newclasses = list(allclasses-knownclasses)
    knownclasses = list(knownclasses)

    all_preds = [int(np.argmax(prob_vector)) for prob_vector in predicted_proba]
    known_labels = [l for l in all_gt_labels if l in knownclasses]
    new_labels = [l for l in all_gt_labels if l in newclasses]
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
