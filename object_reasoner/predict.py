import numpy as np
import sys
import torch
import scipy.stats as stats
import copy

def pred_singlemodel(ReasonerObj, args):
    """A Python re-writing of part of the procedure followed in
    https://github.com/andyzeng/arc-robot-vision/image-matching/evaluateModel.m"
    """
    #Find NN based on the embeddings of a single model

    if args.baseline =='k-net' or args.baseline =="imprk-net" or args.set=='lab':
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
    if args.set=='arc': # ARC2017 (simplified case) - 20 valid classes per run
        predictions = []  #np.empty((tgt_space.shape[0], 472, 2), dtype="object")
        for i,classlist in enumerate(ReasonerObj.tsamples):
            t_emb = tgt_space[i,:] #1x2048
            l2dist = np.linalg.norm(t_emb - prod_space, axis=1)
            all_dists = np.column_stack((ReasonerObj.plabels, l2dist.astype(np.object)))    # keep 2nd column as numeric
            valid_dists = all_dists[np.isin(all_dists, classlist)[:,0]]
            ranking = valid_dists[np.argsort(valid_dists[:, 1])]                            # sort by distance, ascending
            #predictions[i,:] = ranking.astype(np.object)
            predictions.append(ranking.astype(np.object)) # variable length, predictions is a list and not an array in this case
    else:
        #lab set case, all classes are valid in all runs
        predictions = np.empty((tgt_space.shape[0], prod_space.shape[0], 2), dtype="object")
        for i in range(len(ReasonerObj.imglist)):
            t_emb = tgt_space[i,:] #1x2048/
            l2dist = np.linalg.norm(t_emb - prod_space, axis=1)
            all_dists = np.column_stack((ReasonerObj.plabels, l2dist.astype(np.object)))    # keep 2nd column as numeric
            ranking = all_dists[np.argsort(all_dists[:, 1])]                            # sort by distance, ascending
            #predictions[i,:] = ranking[:5, :].astype(np.object)   # keep track of top 5
            predictions[i, :] = ranking.astype(np.object)
    return predictions

def pred_twostage(ReasonerObj, args):
    """Based on gt labels in original implementation by Zeng et al.:
        https://github.com/andyzeng/arc-robot-vision/image-matching/evaluateTwoStage.m"
        but here no novel vs known prediction are made, simply both K-net and N-net preds are returned
    """
    args_ = copy.deepcopy(args)
    args_.baseline = 'k-net'
    knet_pred = pred_singlemodel(ReasonerObj, args_)
    args_.baseline = 'n-net'
    nnet_pred = pred_singlemodel(ReasonerObj, args_)
    return knet_pred, nnet_pred

def pred_by_size(ReasonerObj, dims,current_index):
    """Find NN based on size catalogue"""
    prod_space = ReasonerObj.sizes
    if ReasonerObj.set !='lab':
        classlist = ReasonerObj.tsamples[current_index]
    #normalize first
    t_emb = dims / np.linalg.norm(dims)  # 1x3
    prod_space = prod_space / np.linalg.norm(prod_space)
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1).astype(np.object)   # keep as type numeric
    all_dists = np.column_stack((ReasonerObj.labelset, l2dist))
    if ReasonerObj.set != 'lab':
        valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]
        ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending
    else:
        ranking = all_dists[np.argsort(all_dists[:, 1])] # all classes valid across all test runs
    return ranking #[:5, :] # keep track of top 5

def pred_by_vol(ReasonerObj,volume,current_index):

    """Find NN based on volume catalogue"""
    prod_space = ReasonerObj.volumes
    if ReasonerObj.set != 'lab':
        classlist = ReasonerObj.tsamples[current_index]
    t_emb = volume # 1-dim only
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1).astype(np.object)    # keep as type numeric
    all_dists = np.column_stack((ReasonerObj.labelset, l2dist))   #list(ReasonerObj.KB.keys())
    if ReasonerObj.set != 'lab':
        valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]        # filter by valid for this dataset only
        ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending
    else:
        ranking = all_dists[np.argsort(all_dists[:, 1])] # all classes valid across all test runs
    return ranking #[:5, :] # keep track of top 5


def pred_vol_proba(ReasonerObj,estimated_volume, dist='mixed', tol=0.0001):

    """Make predictions base on size distributions in lab object catalogue
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

area_labels = ['XS','small','medium','large','XL']
depth_labels = ['flat','thin','thick','bulky']

def pred_size_qual(dim1, dim2,thresholds=[0.007,0.05,0.35,0.79]): #): #t3=0.19

    estimated_area = np.log(dim1 * dim2)
    if estimated_area < thresholds[0]: return 'XS'
    elif estimated_area >= thresholds[-1]: return 'XL'
    else: #intermediate cases
        for i in range(len(thresholds)-1):
            if (estimated_area>=thresholds[i] and estimated_area < thresholds[i+1]):
                return area_labels[i+1]

def pred_flat(depth, len_thresh = 0.10): #if depth greater than x% of its min dim then non flat
    depth = np.log(depth)
    if depth <= len_thresh: return True
    else: return False

def pred_thinness(depth, cuts=[0.1,0.2,0.4]):
    """
    Rates object thinness/thickness based on measured depth
    """
    depth = np.log(depth)
    if depth <= cuts[0]: return 'flat'
    elif depth > cuts[-1]: return 'bulky'
    else: # intermediate cases
        for i in range(len(cuts)-1):
            if depth > cuts[i] and depth <= cuts[i+1]:
                return depth_labels[i+1]

def pred_proportion(area_qual, mid_measure, depth_measure, cuts=[0.22,0.23,0.65]): #0.15,0.35,0.65
    prop = float(depth_measure/mid_measure)
    if area_qual == 'small':
        if prop <= cuts[0]:  # flat bin common to all measures
            return 'flat'
        else:
            return 'P' # small but not flat -->proportionate
    elif area_qual =='medium': #extra bins for med and large
        if prop <= cuts[0]:  # flat bin common to all measures
            return 'flat'
        elif prop> cuts[0] and prop<= cuts[1]: return 'thin'
        elif prop > cuts[1]: return 'P'
    elif area_qual == 'large':
        if prop <= cuts[0]:  # flat bin common to all measures
            return 'flat'
        elif prop > cuts[0] and prop<= cuts[1]: return 'thin'
        elif prop > cuts[1] and prop<=cuts[2]: return 'thick'
        else: return 'P'

def pred_AR(crop_dims,estim_dims, t=1.4):
    """
    Returns aspect ration based on 2D crop dimensions
    and estimated dimensions
    """
    height, width = crop_dims #used to derive the orientation
    print("crop dimensions are %s x %s" % (str(width), str(height)))
    d1,d2 = estim_dims # of which we do not know orientation
    if height >= width:
        #h = max(d1,d2)
        #w = min(d1,d2)
        AR = height/width
        if AR >= t: return 'TTW'
        else: return 'EQ' #h and w are comparable
    if height < width:
        #h = min(d1, d2)
        #w = max(d1, d2)
        AR = width/height
        if AR >= t: return 'WTT'
        else: return 'EQ' #h and w are comparable
