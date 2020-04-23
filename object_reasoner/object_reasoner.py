"""Main module."""
import json
import os
import time
import sys
import numpy as np
import cv2
from utils import init_obj_catalogue, load_emb_space, load_camera_intrinsics
from predict import pred_singlemodel, pred_twostage
from evalscript import eval_singlemodel
from img_processing import extract_foreground, detect_contours

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

        # load metadata from txt files provided
        with open(os.path.join(args.test_base,'test-labels.txt')) as txtf, \
            open(os.path.join(args.test_base,'test-other-objects-list.txt')) as smpf, \
            open(os.path.join(args.test_base,'test-product-labels.txt')) as prf, \
            open(os.path.join(args.test_base,'test-imgs.txt')) as imgf:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
            self.tsamples = [l.split(',') for l in smpf.read().splitlines()]     # test samples (each of 20 classes,10 known v 10 novel, chosen at random)
            self.plabels = prf.read().splitlines()       # product img labels
            self.dimglist = [os.path.join(args.test_res,pth.replace('color','depth')) for pth in imgf.read().splitlines()]       # paths to test depth imgs


        # Camera intrinsics
        # replace values in yaml file for custom setups
        self.camintr = load_camera_intrinsics('./camera_intrinsics.yaml')

        # Load predictions from baseline algo
        start = time.process_time()
        if not os.path.isfile(('./data/test_predictions_%s.npy' % args.baseline)):
            # then retrieve from raw embeddings
            self.kprod_emb, self.nprod_emb, self.ktest_emb, self.ntest_emb = load_emb_space(args.test_res)
            if args.baseline == 'two-stage':
                self.predictions = pred_twostage(self, args)
            else:
                self.predictions = pred_singlemodel(self, args)
            if self.predictions is not None:
                np.save(('./data/test_predictions_%s.npy' % args.baseline), self.predictions)
            else:
                print("Prediction mode not supported yet. Please choose a different one.")
                sys.exit(0)
        else:
            self.predictions = np.load(('./data/test_predictions_%s.npy' % args.baseline))

        print("%s detection results retrieved. Took %f seconds." % (args.baseline,float(time.process_time() - start)))
        print("Double checking top-1 accuracies to reproduce baseline...")
        eval_singlemodel(self)

    def run(self):
        """
        Color images are saved as 24-bit RGB PNG.
        Depth images and heightmaps are saved as 16-bit PNG, where depth values are saved in deci-millimeters (10-4m).
        Invalid depth is set to 0.
        Depth images are aligned to their corresponding color images.
        """
        # TODO correct self.predictions from baseline
        for i in range(len(self.dimglist)):  # for each depth image
            dimage = cv2.imread(self.dimglist[i])
            cluster_bw = extract_foreground(dimage)
            masked_dmatrix = detect_contours(dimage,cluster_bw)  #masks depth img based on largest contour

            # based on obj size reasoning

        # self.predictions = None     #TODO new corrected predictions here
        # print("Evaluating again post correction...")
        # eval_singlemodel(self)

        return


