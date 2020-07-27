import torch
from torch import optim
import argparse
import sys
import os
import h5py
import numpy as np

from models import ImprintedKNet, KNet, NNet
import data_loaders
from imprinting import imprint, imprint_fortest
from predict import predict_classifier
from evalscript import eval_classifier

""" Hardcoded training params
    as set in
    https://github.com/andyzeng/arc-robot-vision/blob/master/image-matching/train.lua
"""
batch_size = 6
epochs = 10000000
lr = 0.001
momentum = 0.99
wdecay = 0.000001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_arc',
                        help='path to root folder where img files are kept'
                             'Expects same folder structure as ARC2017')
    parser.add_argument('mode', choices=['train','predict'],
                        help='run mode')
    parser.add_argument('--numobj', default=60, type=int,
                        help='No. of object classes to train on')
    parser.add_argument('--model', default="imprk-net", choices=['imprk-net', 'k-net','n-net'],
                        help='Image Matching model to use')
    parser.add_argument('--out', default='./data/imprintedKnet',
                        help='path where to save outputs. defaults to data/imprintedKnet')
    parser.add_argument('--chkp', default=None,
                        help='path to model checkpoint. Required when running in predict mode')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model =='imprk-net': model = ImprintedKNet(num_classes=args.numobj).to(device)
    elif args.model =='k-net': model = KNet(num_classes=args.numobj).to(device)
    elif args.model == 'n-net': model = NNet(num_classes=args.numobj).to(device)

    params_to_update = model.parameters()  # all params

    if args.mode =='train':

        if not os.path.isdir(os.path.join('./data',args.model, 'snapshots-with-class')):
            os.makedirs(os.path.join('./data',args.model, 'snapshots-with-class'), exist_ok=True)
        from train import train
        model.eval() # eval mode before loading embeddings
        print("Loading training data")
        train_loader = torch.utils.data.DataLoader(
            data_loaders.ImageMatchingDataset(model, device, args, randomised=False), batch_size=batch_size,
            shuffle=True)
        print("Train batches loaded!")
        # no validation set in original training code
        model.train() # back to train mode
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
        if args.model == 'imprk-net':
            model = imprint(model, device, train_loader, num_classes=args.numobj)
            print("Weights have been imprinted based on training classes")

        for epoch in range(epochs):
            print("Epoch %i of %i starts..." % (epoch+1, epochs))
            train(args, model, device, train_loader, epoch, optimizer, epochs)
            if epoch % 10 == 0:
                filepath = os.path.join('./data/',args.model,'snapshots-with-class', 'snapshot-'+str(epoch)+'-randomised.pth')
                #save snapshot locally every x - so epochs
                torch.save(model.state_dict(), filepath)

        return 0

    else:
        """ Test/inference stage
        """
        if args.chkp is None or not os.path.isfile(args.chkp):
            print("Please provide a path to pre-trained model checkpoint")
            return 0

        #All classes at test time: known + novel
        if args.model == 'imprk-net':
            model = ImprintedKNet(feature_extraction=True,num_classes=args.numobj).to(device)
        elif args.model == 'k-net':
            model = KNet(feature_extraction=True,num_classes=args.numobj).to(device)
        elif args.model == 'n-net':
            model = NNet(num_classes=args.numobj).to(device)

        pretrained_dict = torch.load(args.chkp)
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
        model.eval()
        print("Loaded pre-trained model")
        # Extract all product embeddings
        # as well as test imgs embeddings
        test_set = data_loaders.ImageMatchingDataset(model, device, args)
        print("Test data loaded")
        print("Starting imprinting on both known and novel classes")
        # init weights based on both known and novel classes
        if args.model == 'imprk-net':
            model = imprint_fortest(model, device, test_set, old_weights,num_classes=args.numobj)
            model.eval()
        print("Imprinting complete")

        # If KNN matching based on embeddings
        KNN=True
        if not KNN and (args.model == 'imprk-net' or args.model == 'k-net'):
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
            print("saving resulting embeddings under %s" % '/'.join(args.chkp.split('/')[:-1]))
            hfile = h5py.File(os.path.join('/'.join(args.chkp.split('/')[:-1]), 'snapshot-test-results.h5'),'w')
            for k, v in test_results.items():
                hfile.create_dataset(k, data=np.array(v, dtype='<f4'))
            return 0


if __name__ == '__main__':
    sys.exit(main())
