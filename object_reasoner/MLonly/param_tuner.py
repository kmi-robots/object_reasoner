"""
This script recommends a confidence threshold for the Euclidean distance
based on matching the first n embeddings from test set against the support set.
This distance threshold can be then used as starting point for
fine-tuning on the test set (i.e., in the ML prediction selection model under cli.py)
"""
import torch
import os
import sys
import numpy as np
import statistics
import data_loaders
import models
import object_reasoner.preprocessing.utils as utl

#input arguments
n = 10
args = type('args', (object,), {})()
args.set ="KMi"

if args.set=='KMi':
    args.baseline = "k-net"
    args.test_res = "./data"
    kprod_emb, ktest_emb, _,_= utl.load_emb_space(args)
    args.baseline = "n-net"
    nprod_emb, ntest_emb, _, _ = utl.load_emb_space(args)

elif args.set=='arc':
    args.baseline = "two-stage"
    args.test_res = os.path.join("../../arc-robot-vision/image-matching/")
    kprod_emb, ktest_emb, nprod_emb, ntest_emb = utl.load_emb_space(args)

else:
    print("Dataset not supported yet")
    sys.exit(0)

# normalise vectors to normalise scores
ktest_emb = ktest_emb / np.linalg.norm(ktest_emb)
ntest_emb = ntest_emb/ np.linalg.norm(ntest_emb)
kprod_emb = kprod_emb / np.linalg.norm(kprod_emb)
nprod_emb = nprod_emb / np.linalg.norm(nprod_emb)

#pick first n test embeddings
ksubset = ktest_emb[:n,:] #n x 2048
nsubset = ntest_emb[:n,:]

ndists,kdists = [],[]
for i in range(n):
    nemb = nsubset[i,:]
    kemb = ksubset[i,:]
    nl2_dist = np.linalg.norm(nemb - nprod_emb, axis=1)
    kl2_dist = np.linalg.norm(kemb - kprod_emb, axis=1)
    ndists.append(nl2_dist.min())
    kdists.append(kl2_dist.min())

print("Suggested Epsilon for N-net is %f" % (statistics.mean(ndists)))
print("Suggested Epsilon for K-net is %f" % (statistics.mean(kdists)))

