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
from img_processing import extract_foreground_2D, detect_contours
from pcl_processing import cluster_3D, MatToPCL, PathToPCL, estimate_dims
import open3d as o3d
import matplotlib.pyplot as plt

class ObjectReasoner():

    def __init__(self, args):

        start = time.time()
        try:
            with open('./data/obj_catalogue.json') as fin:
                self.KB = json.load(fin) #where the ground truth knowledge is
        except FileNotFoundError:
            self.KB = init_obj_catalogue(args.test_base)
            with open('./data/obj_catalogue.json', 'w') as fout:
                json.dump(self.KB, fout)
        print("Background KB initialized. Took %f seconds." % float(time.time() - start))

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
            self.imglist = [os.path.join(args.test_res, pth) for pth in imgf.read().splitlines()]
            self.dimglist = [p.replace('color','depth') for p in self.imglist]       # paths to test depth imgs

        # Camera intrinsics
        # replace values for custom setups
        self.camintr = load_camera_intrinsics(os.path.join(args.test_res,'./camera-intrinsics.txt'))
        self.camera = o3d.camera.PinholeCameraIntrinsic()
        self.camera.set_intrinsics(640,480,self.camintr[0],self.camintr[4],self.camintr[2],self.camintr[5])

        # Load predictions from baseline algo
        start = time.time()
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

        print("%s detection results retrieved. Took %f seconds." % (args.baseline,float(time.time() - start)))
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
            dimage = cv2.imread(self.dimglist[i],cv2.IMREAD_UNCHANGED)
            plt.imshow(cv2.imread(self.imglist[i],cv2.IMREAD_UNCHANGED)) #, cmap='Greys_r')
            plt.show()
            # origpcl = PathToPCL(self.dimglist[10], self.camera)
            cluster_bw = extract_foreground_2D(dimage)
            masked_dmatrix = detect_contours(dimage,cluster_bw)  #masks depth img based on largest contour
            obj_pcl = MatToPCL(masked_dmatrix, self.camera)
            cluster_pcl = cluster_3D(obj_pcl)
            height,width,depth = estimate_dims(cluster_pcl,obj_pcl)
            current_top = self.predictions[i,:]          # baseline top 5 predictions as (label, distance)
            current_pred = current_top[0,:]  #top 1
            for k, v in self.KB.items():
                if v["label"] == str(int(current_pred[0])) \
                    and str(int(current_pred[0])) != self.labels[i]: # if actually incorrect
                    #TODO remove above check on gt labels afterwards
                    gth,gtw,gtd = v['dimensions'] # ground truth dims, in meters
                    break
            #compare estimated dims with ground truth dims
            continue
            # based on obj size reasoning

        # self.predictions = None     #TODO new corrected predictions here
        # print("Evaluating again post correction...")
        # eval_singlemodel(self)

        return


