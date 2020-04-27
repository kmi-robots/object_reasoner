import numpy as np

def pred_singlemodel(ReasonerObj, args, model=None):
    """A Python re-writing of part of the procedure followed in
    https://github.com/andyzeng/arc-robot-vision/image-matching/evaluateModel.m"
    """
    #Find NN based on the embeddings of a single model
    if model is None:
        model = args.baseline   #use the one specified from cli

    if model =='k-net':
        tgt_space = ReasonerObj.ktest_emb
        prod_space = ReasonerObj.kprod_emb

    elif model=='n-net':
        tgt_space = ReasonerObj.ntest_emb
        prod_space = ReasonerObj.nprod_emb

    # For each test embedding, find Nearest Neighbour in prod space
    # Filter out classes which are not in the current N=20 test sample
    predictions = np.zeros((tgt_space.shape[0],5,2))
    for i,classlist in enumerate(ReasonerObj.tsamples):
        t_emb = tgt_space[i,:] #1x2048
        l2dist = np.linalg.norm(t_emb - prod_space, axis=1)
        all_dists = np.column_stack((ReasonerObj.plabels, l2dist.astype(np.object)))    # keep 2nd column as numeric
        valid_dists = all_dists[np.isin(all_dists, classlist)[:,0]]
        ranking = valid_dists[np.argsort(valid_dists[:, 1])]                            # sort by distance, ascending
        predictions[i,:] = ranking[:5,:]                                                            # keep track of top 5

    return predictions

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
    t_emb = dims  # 1x3
    l2dist = np.linalg.norm(t_emb - prod_space, axis=1)
    all_dists = np.column_stack((ReasonerObj.plabels, l2dist.astype(np.object)))  # keep 2nd column as numeric
    valid_dists = all_dists[np.isin(all_dists, classlist)[:, 0]]
    ranking = valid_dists[np.argsort(valid_dists[:, 1])]  # sort by distance, ascending

    return ranking[:5, :] # keep track of top 5
