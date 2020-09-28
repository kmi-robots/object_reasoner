import torch
from torch import optim
import argparse
import sys
import os
import h5py
import numpy as np
import adabound

from models import ImprintedKNet, KNet, NNet, BaselineNet
import data_loaders
from imprinting import imprint, imprint_fortest
from predict import predict_classifier
from evalscript import eval_classifier
from utils import crop_test, create_class_map
from pytorchtools import EarlyStopping

""" Hardcoded training params
    as set in
    https://github.com/andyzeng/arc-robot-vision/blob/master/image-matching/train.lua
"""
batch_size = 16
epochs = 10000
lr = 0.0001
upper_lr = 0.01
momentum = 0.9
wdecay = 0.000001
patience = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_arc',
                        help='path to root folder where img files are kept'
                             'Expects same folder structure as ARC2017')
    parser.add_argument('mode', choices=['train','predict', 'predict_baseline'],
                        help='run mode')
    parser.add_argument('--numobj', default=60, type=int,
                        help='No. of object classes to train on')
    parser.add_argument('--model', default="imprk-net", choices=['imprk-net', 'k-net','n-net', 'triplet'],
                        help='Image Matching model to use')
    parser.add_argument('--out', default='./data/imprintedKnet',
                        help='path where to save outputs. defaults to data/imprintedKnet')
    parser.add_argument('--set', default='arc', choices=['arc','KMi'],
                        help='Dataset to run on')
    parser.add_argument('--chkp', default=None,
                        help='path to model checkpoint. Required when running in predict mode')
    parser.add_argument('--anns', default=None,
                        help='path to image annotations')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.mode =='train':

        if args.model == 'imprk-net':
            model = ImprintedKNet(num_classes=args.numobj).to(device)
        elif args.model == 'k-net':
            model = KNet(num_classes=args.numobj).to(device)
        elif args.model == 'n-net':
            model = NNet().to(device)
        elif args.model == 'triplet':
            model = BaselineNet().to(device)

        params_to_update = model.parameters()  # all params

        if not os.path.isdir(os.path.join('./data',args.model, 'snapshots-with-class')):
            os.makedirs(os.path.join('./data',args.model, 'snapshots-with-class'), exist_ok=True)
        from train import train
        from validate import validate
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        model.eval() # eval mode before loading embeddings
        print("Loading training data")
        train_loader = torch.utils.data.DataLoader(
            data_loaders.ImageMatchingDataset(model, device, args, randomised=False), batch_size=batch_size,
            shuffle=True)
        print("Train batches loaded!")
        # no validation set in original ARC training code/ dataset
        if args.set =='KMi':
            print("Loading validation data")
            val_loader = torch.utils.data.DataLoader(
                data_loaders.ImageMatchingDataset(model, device, args, randomised=False, load_validation=True), batch_size=batch_size,
                shuffle=False)
            print("Validation batches loaded!")
        model.train() # back to train mode
        # optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
        optimizer = adabound.AdaBound(params_to_update, lr=lr, final_lr=upper_lr)

        if args.model == 'imprk-net':
            model = imprint(model, device, train_loader, num_classes=args.numobj)
            print("Weights have been imprinted based on training classes")

        if args.model == 'n-net' or args.model =='triplet':
            eval_metric='binary' #binary classification
        else:
            eval_metric ='weighted' #multi-class classification (k-net, imprk-net)

        for epoch in range(epochs):

            if early_stopping.early_stop:
                print("Early stopping") # number of epochs without improvement exceeded
                break
            print("Epoch %i of %i starts..." % (epoch+1, epochs))
            train(args, model, device, train_loader, epoch, optimizer, epochs, metric_avg=eval_metric)
            val_improved, early_stopping = validate(args, model, device, val_loader, optimizer, early_stopping, metric_avg=eval_metric)

            if val_improved:
                # print("Validation loss decreased. Saving model...")
                filepath = os.path.join('./data/',args.model,'snapshots-with-class', 'snapshot-'+str(epoch)+'.pth')
                torch.save(model.state_dict(), filepath)

        return 0

    else:
        """ Test/inference stage
        """
        if args.mode !='predict_baseline':
            if args.chkp is None or not os.path.isfile(args.chkp):
                print("Please provide a path to pre-trained model checkpoint")
                return 0
        else:
            args.chkp = os.path.join('./data', args.model,'snapshots-with-class')
            if not os.path.isdir(args.chkp):
                os.makedirs(args.chkp)

        if args.set == 'KMi':
            if not os.path.exists(os.path.join(args.out, '../class_to_index.json')):
                res = create_class_map(os.path.join(args.out, '../class_to_index.json'))
                if res is not None:  # stopping because reference training files are missing
                    return res
            if not os.path.exists(os.path.join(args.out, '../test-imgs.txt')):
                # crop test images to annotate bbox/polygon
                print("Preparing ground truth annotations...")
                ilist = [os.path.join(args.path_to_arc, fname) for fname in os.listdir(args.path_to_arc) if 'depth' not in fname]
                #crop_test(ilist, os.path.join(args.path_to_arc, '../../exported'), args.out)
                crop_test(ilist, args.anns, args.out)

        #All classes at test time: known + novel
        if args.model == 'imprk-net':
            model = ImprintedKNet(feature_extraction=True,num_classes=args.numobj).to(device)
        elif args.model == 'k-net':
            model = KNet(feature_extraction=True,num_classes=args.numobj).to(device)
        elif args.model == 'n-net':
            model = NNet(feature_extraction=True).to(device)
        elif args.model == 'triplet':
            model = BaselineNet(feature_extraction=True).to(device)

        if args.mode !='predict_baseline':

            pretrained_dict = torch.load(args.chkp, map_location=torch.device(device))
            model_dict = model.state_dict()
            if args.model == 'imprk-net':
                # store/keep weight imprinted during training separately
                old_weights = pretrained_dict['fc2.weight']
                # load all pre-trained params except last layer (different num of classes now)
                pretrained_dict = {k: (v if v.size()== model_dict[k].size() else model_dict[k]) for k, v in pretrained_dict.items()}
                # overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
            # load the new state dict
            model.load_state_dict(pretrained_dict)
            print("Loaded pre-trained model")

        model.eval()

        # Extract all product embeddings
        # as well as test imgs embeddings
        test_set = data_loaders.ImageMatchingDataset(model, device, args)
        print("Test data loaded")

        # If KNN matching based on embeddings
        KNN=True
        if not KNN and (args.model == 'imprk-net' or args.model == 'k-net'):
            print("Starting imprinting on both known and novel classes")
            # init weights based on both known and novel classes
            if args.model == 'imprk-net':
                model = imprint_fortest(model, device, test_set, old_weights, num_classes=args.numobj)
                model.eval()
            print("Imprinting complete")
            print("Predicting based on classifier layer")
            test_results = predict_classifier(test_set, model, device) # list of numpy arrays in this case
            # each array is the prob distribution output by the classif layer
            # provide the list of classes seen at training

            with open(os.path.join(args.path_to_arc, 'train-labels.txt')) as txtf:
                knownclasses = set([int(l) for l in txtf.read().splitlines()])
            print("There are %i known classes" % len(knownclasses))
            eval_classifier(test_set.labels, knownclasses, test_results)
            return 0
        else:
            print("Producing embeddings for KNN matching")
            """ Eval done in the ObjectReasoner class"""
            #Save results as HDF5 / (This is the input expected by object_reasoner.py)
            test_results = {}
            test_results['testFeat'] = test_set.data_emb
            test_results['prodFeat'] = test_set.prod_emb
            if args.mode != 'predict_baseline':
                print("saving resulting embeddings under %s" % '/'.join(args.chkp.split('/')[:-1]))
                hfile = h5py.File(os.path.join('/'.join(args.chkp.split('/')[:-1]), 'snapshot-test2-results.h5'),'w')
            else:
                print("saving resulting embeddings under %s" % args.chkp)
                hfile = h5py.File(os.path.join(args.chkp, 'snapshot-test2-results.h5'), 'w')
            for k, v in test_results.items():
                hfile.create_dataset(k, data=np.array(v, dtype='<f4'))
            print("Test embeddings saved")
            return 0

if __name__ == '__main__':
    sys.exit(main())
