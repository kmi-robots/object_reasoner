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
