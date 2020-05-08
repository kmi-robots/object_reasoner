import torch
from torch import optim
import argparse
import sys
import os
import h5py
import numpy as np

from models import ImprintedKNet
import data_loaders
from imprinting import imprint

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
                        help='path to root folder where ARC2017 files are kept')

    parser.add_argument('mode', choices=['train','predict'],
                        help='run mode')
    parser.add_argument('--numobj', default=41,
                        help='No. of object classes to train on')
    parser.add_argument('--out', default='./data/imprintedKnet',
                        help='path where to save outputs. defaults to data/imprintedKnet')
    parser.add_argument('--chkp', default=None,
                        help='path to model checkpoint. Required when running in predict mode')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ImprintedKNet(feature_extraction=False, num_classes=args.numobj).to(device)
    params_to_update = model.parameters()  # all params

    if args.mode =='train':

        if not os.path.isdir('./data/imprintedKnet/snapshots-with-class'):
            os.makedirs('./data/imprintedKnet/snapshots-with-class', exist_ok=True)
        from train import train
        model.eval() # eval mode before loading embeddings
        train_loader = torch.utils.data.DataLoader(
            data_loaders.ImageMatchingDataset(model, device, args), batch_size=batch_size,
            shuffle=True)
        # no validation set in original training code
        model.train() # back to train mode
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)

        imprint(model, device, train_loader, num_classes=args.numobj)
        print("Weights have been imprinted based on training classes")

        for epoch in range(epochs):
            print("Epoch %i of %i starts..." % (epoch+1, epochs))
            train(model, device, train_loader, epoch, optimizer, epochs)
            if epoch % 1000 == 0:
                filepath = os.path.join('./data/imprintedKnet/snapshots-with-class', 'snapshot-'+str(epoch)+'.pth')
                #save snapshot locally every x - so epochs
                torch.save(model.state_dict(), filepath)

        return 0

    else:
        """ Test/inference stage
        """
        if args.chkp is None or not os.path.isfile(args.chkp):
            print("Please provide a path to pre-trained model checkpoint")
            return 0
        model.load_state_dict(torch.load(args.chkp))
        model.eval()
        # Extract all product embeddings
        # as well as test imgs embeddings
        test_set = data_loaders.ImageMatchingDataset(model, device, args)
        # print(test_set.data.shape)
        # print(test_set.prod_data.shape)

        #Save results as HDF5 / (This is the input expected by object_reasoner.py)
        test_results = {}
        test_results['testFeat'] = test_set.data
        test_results['prodFeat']= test_set.prod_data

        hfile = h5py.File(os.path.join('./data/imprintedKnet/snapshots-with-class', 'snapshot-test-results.h5'))
        for k, v in test_results.items():
            hfile.create_dataset(k, data=np.array(v, dtype='<f4'))
        return 0


if __name__ == '__main__':
    sys.exit(main())
