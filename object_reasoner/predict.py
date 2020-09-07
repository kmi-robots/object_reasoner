import numpy as np
import sys
import torch
import scipy.stats as stats

def pred_singlemodel(ReasonerObj, args):
    """A Python re-writing of part of the procedure followed in
    https://github.com/andyzeng/arc-robot-vision/image-matching/evaluateModel.m"
    """
    #Find NN based on the embeddings of a single model

    if args.baseline =='k-net' or args.baseline =="imprk-net" or args.set=='KMi':
        tgt_space = ReasonerObj.ktest_emb
        prod_space = ReasonerObj.kprod_emb

    elif args.baseline=='n-net':
        tgt_space = ReasonerObj.ntest_emb
        prod_space = ReasonerObj.nprod_emb

    else:
        print("model not supported yet")
        sys.exit(0)

    #Add L2 normalization of vectors to normalise scores
    tgt_space = tgt_space / np.linalg.norm(tgt_space)
    prod_space = prod_space / np.linalg.norm(prod_space)

    # For each test embedding, find Nearest Neighbour in prod space
    # Filter out classes which are not in the current N=20 test sample
    predictions = np.empty((tgt_space.shape[0],prod_space.shape[0],2),dtype="object")

    if args.set=='arc': # ARC2017 (simplified case) - 20 valid classes per run
        avg_predictions = np.empty((tgt_space.shape[0], 20, 2), dtype="object")
        min_predictions = np.empty((tgt_space.shape[0], 20, 2), dtype="object")
        for i,classlist in enumerate(ReasonerObj.tsamples):
            t_emb = tgt_space[i,:] #1x2048
            l2dist = np.linalg.norm(t_emb - prod_space, axis=1)
            all_dists = np.column_stack((ReasonerObj.plabels, l2dist.astype(np.object)))    # keep 2nd column as numeric
            valid_dists = all_dists[np.isin(all_dists, classlist)[:,0]]
            ranking = valid_dists[np.argsort(valid_dists[:, 1])]                            # sort by distance, ascending
            predictions[i,:] = ranking[:5, :].astype(np.object)   # keep track of top 5
            valid_classes = list(np.unique(ranking[:,0]))
            #avg by class
            avg_ranking= np.array([(c,np.mean(ranking[ranking[:,0]==c][:,1])) for c in valid_classes], dtype='object')
            avg_predictions[i,:] = avg_ranking[np.argsort(avg_ranking[:, 1])]  # and redo sorting
            #select min for each class
            min_ranking = np.array([(c, np.min(ranking[ranking[:, 0] == c][:, 1])) for c in valid_classes], dtype='object')
            min_predictions[i, :] = min_ranking[np.argsort(avg_ranking[:, 1])]  # and redo sorting
    else:
        #KMi set case, all classes are valid in all runs
        avg_predictions = None
        min_predictions = None
        for i in range(len(ReasonerObj.imglist)):
            t_emb = tgt_space[i,:] #1x2048/
            l2dist = np.linalg.norm(t_emb - prod_space, axis=1)
            all_dists = np.column_stack((ReasonerObj.plabels, l2dist.astype(np.object)))    # keep 2nd column as numeric
            ranking = all_dists[np.argsort(all_dists[:, 1])]                            # sort by distance, ascending
            #predictions[i,:] = ranking[:5, :].astype(np.object)   # keep track of top 5
            predictions[i, :] = ranking.astype(np.object)
    return predictions, avg_predictions, min_predictions

def pred_twostage(ReasonerObj, args):
    """A Python re-writing of part of the procedure followed in
        https://github.com/andyzeng/arc-robot-vision/image-matching/evaluateTwoStage.m"
    """

    #Decide if object is Known or Novel, based on best threshold
    # Not clear from their code how to do it without GT

    #If assumed to be Known use K-net
    # pred_singlemodel(ReasonerObj, args, model='k-net')
    #Otherwise use N-net
    # pred_singlemodel(ReasonerObj, args, model='n-net')

    return

def pred_by_size(ReasonerObj, dims,current_index):

    """Find NN based on size catalogue"""
    prod_space = ReasonerObj.sizes
    if ReasonerObj.set !='KMi':
        classlist = ReasonerObj.tsamples[current_index]
    #normalize first
    t_emb = dims / np.linalg.norm(dims)  # 1x3
    prod_space = prod_space / np.linalg.norm(prod_space)
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1).astype(np.object)   # keep as type numeric
    all_dists = np.column_stack((ReasonerObj.labelset, l2dist))
    if ReasonerObj.set != 'KMi':
        valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]
        ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending
    else:
        ranking = all_dists[np.argsort(all_dists[:, 1])] # all classes valid across all test runs

    return ranking #[:5, :] # keep track of top 5

def pred_by_vol(ReasonerObj,volume,current_index):

    """Find NN based on volume catalogue"""
    prod_space = ReasonerObj.volumes
    if ReasonerObj.set != 'KMi':
        classlist = ReasonerObj.tsamples[current_index]
    t_emb = volume # 1-dim only
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1).astype(np.object)    # keep as type numeric
    all_dists = np.column_stack((ReasonerObj.labelset, l2dist))   #list(ReasonerObj.KB.keys())
    if ReasonerObj.set != 'KMi':
        valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]        # filter by valid for this dataset only
        ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending
    else:
        ranking = all_dists[np.argsort(all_dists[:, 1])] # all classes valid across all test runs

    return ranking #[:5, :] # keep track of top 5


def pred_vol_proba(ReasonerObj,estimated_volume, dist='mixed', tol=0.0001):

    """Make predictions base on size distributions in KMi object catalogue
    See object_sizes.py for more details on how these distributions are derived
    # volume ranges are computed based on set (fixed) tolerance
    """
    #vol_min, vol_max = float(estimated_volume - tol * estimated_volume), float(
    #    estimated_volume + tol * estimated_volume)
    #avoided percentage of volume for tolerance to not create too much bias
    vol_min, vol_max = float(estimated_volume - tol), float(
        estimated_volume + tol)
    cats, probabilities = [], []
    for k in ReasonerObj.KB.keys():

        cat = k.replace(' ', '_').replace('/', '_')  # copy to adjust based labels in index
        if cat not in ReasonerObj.remapper.values(): continue  # only across training classes (e.g., 60 instead of 65)
        try:
            dist_name = ReasonerObj.KB[k]['distribution']
            if dist=='lognormal' and dist_name !='uniform' and dist_name is not None: #use lognormal representation
                dist_name = stats.lognorm.name
                params = ReasonerObj.KB[k]['lognorm-params']
            elif ReasonerObj.KB[k]['distribution']=='uniform': #not enough data points, it was marked as uniform
                dist_name = ReasonerObj.KB[k]['distribution']
                params = ReasonerObj.KB[k]['params']
            else: #use representation computed as best fit in object_sizes.py
                dist_name = ReasonerObj.KB[k]['distribution']
                params = ReasonerObj.KB[k]['params']

        except KeyError:  # object catalogue with limited fit (only log and uniform)
            #proba = 0.
            if ReasonerObj.KB[k]['lognorm-params'] is not None:
                dist_name = 'lognorm'
                params = ReasonerObj.KB[k]['lognorm-params']
            elif ReasonerObj.KB[k]['uniform-params'] is not None:
                dist_name = 'uniform'
                params = ReasonerObj.KB[k]['uniform-params']
            else: dist_name = None

        if dist_name is not None:  # probability as area under the curve for given volume range
            distmethod = getattr(stats, dist_name)
            proba = distmethod.cdf(vol_max, *params) - \
                    distmethod.cdf(vol_min, *params)
        else:
            proba = 0.  # originally blacklisted object
        cats.append(cat)
        probabilities.append(proba)

    #all_scores = np.column_stack((list(ReasonerObj.KB.keys()), probabilities)])
    dtype = [('class',object),('proba',float)]
    all_scores = np.empty((len(cats),), dtype=dtype)
    all_scores['class'] = np.array(cats)
    all_scores['proba'] = np.array(probabilities)
    return all_scores[np.argsort(all_scores['proba'])[::-1]] # rank by descending probability #[::-1] is used to reverse np.argsort


def predict_classifier(test_data, model, device):
    predictions = []
    with torch.no_grad():
        for i in range(test_data.data.shape[0]):
            data_point = test_data.data[i,:].unsqueeze(0).to(device)
            out_logits = model.forward(data_point, trainmode=False)
            predictions.append(int(np.argmax(out_logits.cpu().numpy()))) #torch.argmax(class_prob, dim=1).tolist())
    return predictions
