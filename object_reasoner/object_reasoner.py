"""Main module."""
import json
import os
import time
import numpy as np
from utils import init_obj_catalogue, load_emb_space, pred_singlemodel, pred_twostage
from evalscript import eval_singlemodel

class ObjectReasoner():

    def __init__(self, args):

        start = time.process_time()
        try:
            with open('./data/obj_catalogue.json') as fin:
                self.KB = json.load(fin) #where the ground truth knowledge is
        except FileNotFoundError:
            self.KB = init_obj_catalogue(args.test_base)
            with open('./data/obj_catalogue.json', 'w') as fout:
                json.dump(self.KB, fout)
        print("Background KB initialized. Took %f seconds." % float(time.process_time() - start))

        # Filter only known/novel objects
        self.known = dict((k, v) for k, v in self.KB.items() if v["known"] == True)
        self.novel = dict((k, v) for k, v in self.KB.items() if v["known"] == False)
        with open(os.path.join(args.test_base,'test-labels.txt')) as txtf, \
            open(os.path.join(args.test_base,'test-other-objects-list.txt')) as smpf, \
            open(os.path.join(args.test_base,'test-product-labels.txt')) as prf:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
            self.tsamples = [l.split(',') for l in smpf.read().splitlines()]     # test samples (each of 20 classes,10 known v 10 novel, chosen at random)
            self.plabels = prf.read().splitlines()       # product img labels

        #Retrieve results/embeddings from baseline algo
        start = time.process_time()
        if not os.path.isfile(('./data/test_predictions_%s.npy' % args.baseline)):
            # then retrieve from raw embeddings
            self.kprod_emb, self.nprod_emb, self.ktest_emb, self.ntest_emb = load_emb_space(args.test_res)
            if args.baseline == 'two-stage':
                self.predictions = pred_twostage(self, args)
            else:
                self.predictions = pred_singlemodel(self, args)
            np.save(('./data/test_predictions_%s.npy' % args.baseline), self.predictions)
        else:
            self.predictions = np.load(('./data/test_predictions_%s.npy' % args.baseline))

        print("%s detection results retrieved. Took %f seconds." % (args.baseline,float(time.process_time() - start)))
        print("Double checking top-1 accuracies to reproduce baseline...")
        eval_singlemodel(self)

    def run(self, args):

        # TODO correct self.predictions from baseline
        # based on obj size reasoning

        return


