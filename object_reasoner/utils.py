"""
Methods used to convert/handle raw input data
"""
import os
import json
import h5py
import numpy as np
import cv2

def init_obj_catalogue(path_to_data):

    obj_dict = {}
    subfols = sorted(os.listdir(os.path.join(path_to_data, 'test-item-data')))
    known_classes = sorted(os.listdir(os.path.join(path_to_data, 'train-item-data')))
    for i,class_fol in enumerate(subfols):
        cname = class_fol.lower()
        try:
            with open(os.path.join(path_to_data, 'test-item-data',class_fol, cname+'.json')) as fin:
                obj_node = json.load(fin) #start from json data given
                obj_node['known'] = True if class_fol in known_classes else False #Known v Novel?

        except FileNotFoundError:
            print("No json file found for object %s" % cname)
            print("Adding empty node")
            obj_node = {}
            obj_node['known'] = True #'Empty' is known at training time

        obj_node['label'] = str(i + 1) # add class label in same format as gt (starts from 1)
        obj_dict[cname] = obj_node

    return obj_dict

def load_emb_space(path_to_hdf5):
    """
    Assumes the input are the HDF5 files
    as produced by the baselines provided at
    https://github.com/andyzeng/arc-robot-vision/image-matching
    """
    tgt_novel = os.path.join(path_to_hdf5, 'snapshots-no-class', 'results-snapshot-8000.h5') #default folder structure by Zeng et al.
    tgt_known = os.path.join(path_to_hdf5, 'snapshots-with-class', 'results-snapshot-170000.h5')

    nnetf = h5py.File(tgt_novel, 'r')
    knetf = h5py.File(tgt_known, 'r')

    return np.array(knetf['prodFeat'], dtype='<f4'), np.array(nnetf['prodFeat'], dtype='<f4'), \
           np.array(knetf['testFeat'], dtype='<f4'), np.array(nnetf['testFeat'], dtype='<f4')


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


def load_depth(paths_to_depths):
    h,w,c = cv2.imread(paths_to_depths[0]).shape
    all_depths = np.zeros((len(paths_to_depths),h,w,c))
    for i,pth in enumerate(paths_to_depths):
        all_depths[i,:] = cv2.imread(pth)
    return all_depths
