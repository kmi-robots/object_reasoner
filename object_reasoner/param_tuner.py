"""
This script recommends a confidence threshold for the Euclidean distance
based on matching pre-trained embeddings with support imgs in the training set
This distance threshold can be then used as starting point for
fine-tuning on the test set (i.e., in the ML prediction selection model under cli.py)
"""
import torch
from MLonly import data_loaders
from MLonly import models
from MLonly import main

N = 60
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.KNet(num_classes=N).to(device)
model2 = models.NNet().to(device)
train_loader = data_loaders.ImageMatchingDataset(model, device, main.args, randomised=False)

print("Training data loaded")
