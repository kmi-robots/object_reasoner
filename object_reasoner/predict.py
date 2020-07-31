import numpy as np
import sys
import torch

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
    predictions = np.empty((tgt_space.shape[0],5,2),dtype="object")

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
            predictions[i,:] = ranking[:5, :].astype(np.object)   # keep track of top 5

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
    classlist = ReasonerObj.tsamples[current_index]
    #normalize first
    t_emb = dims / np.linalg.norm(dims)  # 1x3
    prod_space = prod_space / np.linalg.norm(prod_space)
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1).astype(np.object)   # keep as type numeric
    all_dists = np.column_stack((ReasonerObj.labelset, l2dist))
    valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]
    ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending

    return ranking #[:5, :] # keep track of top 5

def pred_by_vol(ReasonerObj,volume,current_index):

    """Find NN based on volume catalogue"""
    prod_space = ReasonerObj.volumes
    classlist = ReasonerObj.tsamples[current_index]
    t_emb = volume # 1-dim only
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1).astype(np.object)    # keep as type numeric
    all_dists = np.column_stack((ReasonerObj.labelset, l2dist))   #list(ReasonerObj.KB.keys())
    valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]        # filter by valid for this dataset only
    ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending

    return ranking #[:5, :] # keep track of top 5


def predict_classifier(test_data, model, device):
    predictions = []
    with torch.no_grad():
        for i in range(test_data.data.shape[0]):
            data_point = test_data.data[i,:].unsqueeze(0).to(device)
            out_logits = model.forward(data_point, trainmode=False)
            predictions.append(int(np.argmax(out_logits.cpu().numpy()))) #torch.argmax(class_prob, dim=1).tolist())
    return predictions
