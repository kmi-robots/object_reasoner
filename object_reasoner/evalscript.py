from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import math

def eval_singlemodel(ReasonerObj,eval_d,method, K=1):
    if K==1 and ReasonerObj.set=='arc':
        """
        Evaluation script - top-1 eval accuracy
        to reproduce results in https://github.com/andyzeng/arc-robot-vision/image-matching
        """
        all_labels = ReasonerObj.labels
        knownclasses = [v['label'] for v in ReasonerObj.known.values()]
        newclasses = [v['label'] for v in ReasonerObj.novel.values()]
        all_preds = [ar[0][0] for ar in  ReasonerObj.predictions] #only top-1 pred
        #all_preds = ReasonerObj.predictions[:,0,0].astype('int').astype('str').tolist()
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
        for k, metr in [('known_acc',float(k_sum / len(known_labels)) ), ('novel_acc',float(k_sum / len(known_labels)) ), \
                         ('mixed_acc',float(all_sum / len(all_preds)))]:
            try: eval_d[method][k].append(metr)
            except KeyError:
                eval_d[method][k] = []
                eval_d[method][k].append(metr)
        return eval_d

    elif K==1 and ReasonerObj.set=='KMi':
        # eval top-1 of each ranking
        y_pred = ReasonerObj.predictions[:, 0, 0].astype('int').astype('str').tolist()
        y_true = ReasonerObj.labels
        global_acc = accuracy_score(y_true, y_pred)
        print(classification_report(y_true, y_pred, digits=4))
        print(global_acc)
        Pu,Ru, F1u, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        Pw, Rw, F1w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        for k,metr in [('accuracy',global_acc),('Punweighted',Pu),('Runweighted',Ru),('F1unweighted',F1u), ('Pweighted',Pw),('Rweighted',Rw),('F1weighted',F1w)]:
            try:eval_d[method][k].append(metr)
            except KeyError:
                eval_d[method][k] =[]
                eval_d[method][k].append(metr)
        return eval_d
    else:#eval quality of top-K ranking
        return eval_ranking(ReasonerObj, K, eval_d,method)


def eval_twostage():
    """Two-stage pipeline, by Zeng et al. (2018). Skipped because it uses gt labels to predict Known v Novel optimal thresholds"""
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


def eval_ranking(ReasonerObj,K,eval_d,method):
    """
    Prints mean Precision@K, mean nDCG@K and hit ratio @ K
    """
    if ReasonerObj.set=='KMi':
        y_pred = ReasonerObj.predictions[:, :K, 0].astype('int').astype('str').tolist()
    else: #ARC: predictions in a list
        y_pred = [ar[:K,0] for ar in ReasonerObj.predictions]
    y_true = ReasonerObj.labels
    precisions = []
    ndcgs = []
    hits = 0
    IDCG = 0.  # Ideal DCG
    for n in range(2, K + 2):
        IDCG += float(1 / math.log(n, 2))
    for z, (ranking, gt_label) in enumerate(zip(y_pred, y_true)):
        pred_rank = [1 if r == gt_label else 0 for r in ranking]
        dis_scores = [float(1 / math.log(i + 2, 2)) for i, r in enumerate(ranking) if r == gt_label]
        no_hits = pred_rank.count(1)
        precisions.append(float(no_hits / K))
        if no_hits >= 1:
            hits += 1  # increment if at least one hit in the ranking
            nDCG = float(sum(dis_scores) / IDCG)  # compute nDCG for ranking
            ndcgs.append(nDCG)
    print("Avg ranking Precision@%i: %f " % (K, float(sum(precisions) / len(precisions))))
    print("Avg Normalised DCG @%i: %f" % (K, float(sum(ndcgs) / len(precisions))))
    print("Hit ratio @%i: %f" % (K, float(hits / len(precisions))))

    for k,metr in [('meanP@K', float(sum(precisions) / len(precisions))), ('meannDCG@K', float(sum(ndcgs) / len(precisions))) \
        , ('hitratio', float(hits / len(precisions)))]:
        try: eval_d[method][k].append(metr)
        except KeyError:
            eval_d[method][k] = []
            eval_d[method][k].append(metr)
    return eval_d
