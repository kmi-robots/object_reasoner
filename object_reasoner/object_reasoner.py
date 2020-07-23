"""Main module."""
import open3d as o3d
import json
import os
import time
import sys
import numpy as np
import cv2
from collections import Counter
from utils import init_obj_catalogue, load_emb_space, load_camera_intrinsics
from predict import pred_singlemodel, pred_twostage, pred_by_vol
from evalscript import eval_singlemodel
from img_processing import extract_foreground_2D, detect_contours
from pcl_processing import cluster_3D, MatToPCL, PathToPCL, estimate_dims

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

        # KB sizes as 3D vectors
        # self.sizes = dict((k,np.array(v['dimensions'])) for k, v in self.KB.items())
        self.sizes = np.empty((len(self.KB.keys()),3))
        self.volumes = np.empty((len(self.KB.keys()),1))
        self.labelset = []
        for i, (k, v) in enumerate(self.KB.items()):
            self.sizes[i] = np.array(v['dimensions'])
            self.volumes[i] = v['dimensions'][0]*v['dimensions'][1]*v['dimensions'][2]
            self.labelset.append(v['label'])

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
            self.kprod_emb, self.ktest_emb, self.nprod_emb, self.ntest_emb = load_emb_space(args)
            if args.baseline == 'two-stage':
                self.predictions = pred_twostage(self, args)
            else:
                self.predictions,self.avg_predictions, self.min_predictions = pred_singlemodel(self, args)
            if self.predictions is not None:
                np.save(('./data/test_predictions_%s.npy' % args.baseline), self.predictions)
                np.save(('./data/test_avg_predictions_%s.npy' % args.baseline), self.avg_predictions)
                np.save(('./data/test_min_predictions_%s.npy' % args.baseline), self.min_predictions)
            else:
                print("Prediction mode not supported yet. Please choose a different one.")
                sys.exit(0)
        else:
            self.predictions = np.load(('./data/test_predictions_%s.npy' % args.baseline),allow_pickle=True)
            self.avg_predictions = np.load(('./data/test_avg_predictions_%s.npy' % args.baseline), allow_pickle=True)
            self.min_predictions = np.load(('./data/test_min_predictions_%s.npy' % args.baseline), allow_pickle=True)

        print("%s detection results retrieved. Took %f seconds." % (args.baseline,float(time.time() - start)))
        print("Double checking top-1 accuracies to reproduce baseline...")
        eval_singlemodel(self)

        """
        print("Results if matches are average by class / across views")
        # print("Results if min across views is taken")
        tmp_copy = self.predictions
        self.predictions = self.avg_predictions
        # self.predictions = self.min_predictions
        eval_singlemodel(self)
        self.predictions = tmp_copy
        del tmp_copy
        sys.exit(0)
        """

    def run(self):
        """
        Color images are saved as 24-bit RGB PNG.
        Depth images and heightmaps are saved as 16-bit PNG, where depth values are saved in deci-millimeters (10-4m).
        Invalid depth is set to 0.
        Depth images are aligned to their corresponding color images.
        """
        print("Reasoning for correction ... ")
        start = time.time()
        non_processed_pcls = 0
        no_corrected =0
        for i in range(len(self.dimglist)):  # for each depth image

            dimage = cv2.imread(self.dimglist[i],cv2.IMREAD_UNCHANGED)

            """
            plt.imshow(dimage, cmap='Greys_r')
            plt.show()
            # origpcl = PathToPCL(self.dimglist[i], self.camera)
            o3d.visualization.draw_geometries([origpcl])
            """

            cluster_bw = extract_foreground_2D(dimage)
            if cluster_bw is None: #problem with 2D clustering
                non_processed_pcls += 1
                print("There was a problem extracting a relevant cluster for img no %i, skipping correction" % i)
                gt_label = list(self.KB.keys())[int(self.labels[i]) - 1]
                print(gt_label)
                continue  # skip correction
            masked_dmatrix = detect_contours(dimage,cluster_bw)  #masks depth img based on largest contour
            obj_pcl = MatToPCL(masked_dmatrix, self.camera)
            cluster_pcl = cluster_3D(obj_pcl)
            if cluster_pcl is None: #or with 3D clustering
                non_processed_pcls+=1
                print("There was a problem extracting a relevant cluster for img no %i, skipping correction" % i)
                gt_label = list(self.KB.keys())[int(self.labels[i])-1]
                print(gt_label)
                continue #skip correction

            d1,d2,depth,volume, orientedbox = estimate_dims(cluster_pcl,obj_pcl)
            current_ranking = self.predictions[i, :]  # baseline predictions as (label, distance)
            # current_avg_ranking = self.avg_predictions[i,:]
            current_min_ranking = self.min_predictions[i, :]
            current_prediction = int(self.predictions[i,0,0]) - 1   # class labels start at 1 but indexing starts at 0
            current_label = list(self.KB.keys())[current_prediction]
            gt_label = list(self.KB.keys())[int(self.labels[i])-1]
            pr_volume = self.volumes[current_prediction]    # ground truth volume and dims for current prediction
            # pr_dims = self.sizes[current_prediction]

            alpha = 4
            volOnly = True   # if True, dims based rankings are excluded
            combined = False # if True, all types of ranking used
            novision = True # if True, vision based ranking is excluded

            if current_label != gt_label: # and abs(volume - pr_volume) > alpha*pr_volume :
                no_corrected += 1
                # If we detect a volume alpha times or more larger/smaller
                # than object predicted by baseline, then hypothesise object needs correction
                # actually need to correct, gives upper bound
                # TODO remove 1st check condition afterwards
                # create ranking by nearest volume
                # gt_volume = self.volumes[int(self.labels[i]) - 1]
                # plt.imshow(cv2.imread(self.imglist[i], cv2.IMREAD_UNCHANGED))  # , cmap='Greys_r')
                # plt.show()
                # o3d.visualization.draw_geometries([obj_pcl, orientedbox])
                gt_dims = self.sizes[int(self.labels[i]) - 1]
                from predict import pred_by_size
                # compare estimated dims with ground truth dims
                # first possible permutation
                dims_ranking = pred_by_size(self, np.array([d1,d2,depth]),i)
                #second possible permutation
                p2_rank = pred_by_size(self, np.array([d2, d1, depth]),i)
                vol_ranking = pred_by_vol(self,volume, i)
                # Set of classes from both rankings
                """
                union_list = list(current_ranking[:,0])+list(dims_ranking[:,0])+\
                                    list(p2_rank[:,0]) + list(vol_ranking[:,0])
                class_set= list(set(union_list))
                """
                class_set = list(np.unique(current_min_ranking[:,0]))

                final_rank= Counter()
                for cname in class_set:

                    try:
                        # base_score = current_ranking[current_ranking[:, 0] == cname][:, 1][0]
                        base_score = current_min_ranking[current_min_ranking[:, 0] == cname][:, 1][0]
                    except:
                        #class is not in the base top-5
                        base_score = 0.

                    if combined:
                        if novision:
                            dim_p1_score = dims_ranking[dims_ranking[:, 0] == cname][:, 1][0]
                            dim_p2_score = p2_rank[p2_rank[:, 0] == cname][:, 1][0]
                            vol_score = vol_ranking[vol_ranking[:, 0] == cname][:, 1][0]
                            final_rank[cname] = sum([dim_p1_score, dim_p2_score, vol_score]) / 3
                        else:
                            dim_p1_score = dims_ranking[dims_ranking[:, 0] == cname][:, 1][0]
                            dim_p2_score = p2_rank[p2_rank[:, 0] == cname][:, 1][0]
                            vol_score = vol_ranking[vol_ranking[:, 0] == cname][:, 1][0]
                            final_rank[cname] = sum([base_score,dim_p1_score,dim_p2_score,vol_score])/4
                    else:
                        if volOnly and not novision:
                            vol_score = vol_ranking[vol_ranking[:, 0] == cname][:, 1][0]
                            final_rank[cname] = sum([base_score, vol_score]) / 2
                        elif volOnly and novision:
                            final_rank[cname] = vol_ranking[vol_ranking[:, 0] == cname][:, 1][0]
                        elif not volOnly and novision:
                            dim_p1_score = dims_ranking[dims_ranking[:, 0] == cname][:, 1][0]
                            dim_p2_score = p2_rank[p2_rank[:, 0] == cname][:, 1][0]
                            final_rank[cname] = sum([dim_p1_score, dim_p2_score]) / 2
                        else:
                            dim_p1_score = dims_ranking[dims_ranking[:, 0] == cname][:, 1][0]
                            dim_p2_score = p2_rank[p2_rank[:, 0] == cname][:, 1][0]
                            final_rank[cname] = sum([base_score, dim_p1_score, dim_p2_score]) / 3

                final_rank = final_rank.most_common()[:-6:-1] # nearest 5 from new ranking
                self.predictions[i, :] = final_rank

        print("Took % fseconds." % float(time.time() - start)) #global proc time
        print("Re-evaluating post size correction...")
        eval_singlemodel(self)
        print("%s image predictions were corrected " % str(no_corrected))
        print("%s images were skipped and kept the same " % str(non_processed_pcls))
        return


