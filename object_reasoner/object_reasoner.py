"""Main module."""
import open3d as o3d
import json
import os
import time
import sys
import numpy as np
import cv2
from collections import Counter
from utils import init_obj_catalogue, load_emb_space, load_camera_intrinsics_txt, call_python_version
from predict import pred_singlemodel, pred_twostage, pred_by_vol, pred_vol_proba, pred_by_size
from evalscript import eval_singlemodel, eval_KMi
from img_processing import extract_foreground_2D, detect_contours
from pcl_processing import cluster_3D, MatToPCL, PathToPCL, estimate_dims

import matplotlib.pyplot as plt

class ObjectReasoner():

    def __init__(self, args):

        self.set = args.set
        start = time.time()
        if args.set =='arc':
            try:
                with open('./data/arc_obj_catalogue.json') as fin:
                    self.KB = json.load(fin) #where the ground truth knowledge is
            except FileNotFoundError:
                self.KB = init_obj_catalogue(args.test_base)
                with open('./data/arc_obj_catalogue.json', 'w') as fout:
                    json.dump(self.KB, fout)
            # Filter only known/novel objects
            self.known = dict((k, v) for k, v in self.KB.items() if v["known"] == True)
            self.novel = dict((k, v) for k, v in self.KB.items() if v["known"] == False)

            # KB sizes as 3D vectors
            # self.sizes = dict((k,np.array(v['dimensions'])) for k, v in self.KB.items())
            self.sizes = np.empty((len(self.KB.keys()), 3))
            self.volumes = np.empty((len(self.KB.keys()), 1))
            self.labelset = []
            for i, (k, v) in enumerate(self.KB.items()):
                self.sizes[i] = np.array(v['dimensions'])
                self.volumes[i] = v['dimensions'][0] * v['dimensions'][1] * v['dimensions'][
                    2]  # size catalogue in meters
                self.labelset.append(v['label'])
        elif args.set == 'KMi':
            try:
                with open('./data/KMi_obj_catalogue.json') as fin,\
                    open('./data/KMi-set-2020/class_to_index.json') as cin:
                    self.KB = json.load(fin) #where the ground truth knowledge is
                    self.mapper = json.load(cin)
                    self.remapper =dict((v, k) for k, v in self.mapper.items())  # swap keys with indices
            except FileNotFoundError:
                print("No KMi catalogue or class-to-label index found - please refer to object_sizes.py for expected catalogue format")
                sys.exit(0)
        print("Background KB initialized. Took %f seconds." % float(time.time() - start))

        # load metadata from txt files provided
        with open(os.path.join(args.test_base,'test-labels.txt')) as txtf, \
            open(os.path.join(args.test_base,'test-product-labels.txt')) as prf, \
            open(os.path.join(args.test_base,'test-imgs.txt')) as imgf:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
              # test samples (each of 20 classes,10 known v 10 novel, chosen at random)
            self.plabels = prf.read().splitlines()       # product img labels
            self.imglist = [os.path.join(args.test_base,'..','..', pth) for pth in imgf.read().splitlines()]

        if args.set =='KMi':
            if args.bags is None or not os.path.isdir(args.bags):
                print("Print provide a valid path to the bag files storing depth data")
                sys.exit(0)
            self.tsamples = None
            #if no depth imgs stored locally, generate from bag
            if not os.path.exists(self.imglist[0][:-4]+'depth.png'): # checking on 1st img of list for instance
                if args.regions is None or not os.path.isdir(args.bags):
                    print("Print provide a valid path to the region annotation files")
                    sys.exit(0)
                # call python2 method from python3 script
                print("Starting depth image extraction from bag files... It may take long to complete")
                self.dimglist = call_python_version("2.7", "bag_processing", "extract_from_bag", [self.imglist,args.bags,args.regions]) # retrieve data from bag
                print("Depth files creation complete.. Imgs saved under %s" % os.path.join(args.test_base,'test-imgs'))
                print("Empty files:")
                print("%s out of %s" % (len([d for d in self.dimglist if d is None]), len(self.dimglist)))

            else:
                self.dimglist = [cv2.imread(p[:-4]+'depth.png', cv2.IMREAD_UNCHANGED) for p in
                                 self.imglist]
                print("Empty depth files:")
                print("%s out of %s" % (len([d for d in self.dimglist if d is None]), len(self.dimglist)))
            self.scale = 1000.0 #depth values in mm

        else: #supports ARC set
            self.dimglist = [cv2.imread(p.replace('color','depth'),cv2.IMREAD_UNCHANGED) for p in self.imglist]       # paths to test depth imgs
            self.scale = 10000.0  # depth values in deci-mm
            with open(os.path.join(args.test_base, 'test-other-objects-list.txt')) as smpf:
                self.tsamples = [l.split(',') for l in smpf.read().splitlines()]

        # Camera intrinsics
        if not os.path.exists(os.path.join(args.test_base,'./camera-intrinsics.txt'))\
            and not os.path.exists(os.path.join(args.test_res,'./camera-intrinsics.txt')):
            bagpath = [os.path.join(args.bags, bagname) for bagname in os.listdir(args.bags) if bagname[-4:] == '.bag'][0]
            self.camintr = self.dimglist = call_python_version("2.7", "bag_processing", "load_intrinsics",[bagpath,\
                                                                os.path.join(args.test_base,'./camera-intrinsics.txt')])
        elif not os.path.exists(os.path.join(args.test_base,'./camera-intrinsics.txt')) \
            and os.path.exists(os.path.join(args.test_res, './camera-intrinsics.txt')):
            self.camintr = load_camera_intrinsics_txt(os.path.join(args.test_res, './camera-intrinsics.txt'))
        else:
            self.camintr = load_camera_intrinsics_txt(os.path.join(args.test_base, './camera-intrinsics.txt'))

        self.camera = o3d.camera.PinholeCameraIntrinsic()
        self.camera.set_intrinsics(640, 480, self.camintr[0], self.camintr[4], self.camintr[2], self.camintr[5])

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
            else:
                print("Prediction mode not supported yet. Please choose a different one.")
                sys.exit(0)
            if self.avg_predictions is not None:
                np.save(('./data/test_avg_predictions_%s.npy' % args.baseline), self.avg_predictions)
            if self.min_predictions is not None:
                np.save(('./data/test_min_predictions_%s.npy' % args.baseline), self.min_predictions)

        else:
            self.predictions = np.load(('./data/test_predictions_%s.npy' % args.baseline),allow_pickle=True)
            try:
                self.avg_predictions = np.load(('./data/test_avg_predictions_%s.npy' % args.baseline), allow_pickle=True)
                self.min_predictions = np.load(('./data/test_min_predictions_%s.npy' % args.baseline), allow_pickle=True)
            except FileNotFoundError:
                self.avg_predictions = None
                self.min_predictions = None

        print("%s detection results retrieved. Took %f seconds." % (args.baseline,float(time.time() - start)))
        print("Double checking top-1 accuracies to reproduce baseline...")
        if args.set == 'KMi': #class-wise report
            eval_KMi(self, depth_aligned=True)
        else: #separate eval for known vs novel
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
        ARC set: Color images are saved as 24-bit RGB PNG.
        Depth images and heightmaps are saved as 16-bit PNG, where depth values are saved in deci-millimeters (10-4m).
        Invalid depth is set to 0.
        Depth images are aligned to their corresponding color images.
        """
        """
        KMi set: Picked nearest depth frame to RGB image, based on timestamp.
        Depth images are saved as 16-bit PNG, where depth values are saved in millimeters (10-3m).
        Invalid depth is set to 0.
        """
        print("Reasoning for correction ... ")
        start = time.time()
        non_processed_pcls = 0
        non_depth_aval = 0
        no_corrected =0
        nfallbacks = 0
        alpha = 4
        volOnly = True  # if True, dims based rankings are excluded
        combined = False  # if True, all types of ranking used
        novision = True  # if True, vision based ranking is excluded
        knowledge_only = False
        foregroundextract = True
        pclcluster = True
        reranking = False

        if self.set == 'KMi':
            # only based on volume
            combined = False
            volOnly = True
            novision = False
            knowledge_only = False
            foregroundextract = True
            pclcluster = True
            reranking = True
            epsilon = 0.0001 # conf threshold for size probas
            all_predictions = self.predictions  # copy to store all similarity scores, not just top 5
            self.predictions = self.predictions[:, :5, :]

        pred_counts = Counter()
        for i,dimage in enumerate(self.dimglist):  # for each depth image
              # baseline predictions as (label, distance)
            if self.min_predictions is not None:
                # current_avg_ranking = self.avg_predictions[i,:]
                current_min_ranking = self.min_predictions[i, :]
            else:
                current_min_ranking = None

            current_ranking = self.predictions[i, :]
            current_prediction = self.predictions[i, 0, 0]
            if self.set == 'KMi':
                current_label = self.remapper[current_prediction]
                gt_label = self.remapper[self.labels[i]]
            else:
                current_label = list(self.KB.keys())[current_prediction]
                gt_label = list(self.KB.keys())[int(self.labels[i]) - 1]
                pr_volume = self.volumes[current_prediction]  # ground truth volume and dims for current prediction
                # pr_dims = self.sizes[current_prediction]
            if current_label != gt_label: # and abs(volume - pr_volume) > alpha*pr_volume :
                no_corrected += 1
            if dimage is None:
                # no synchronised depth data found
                print("No depth data available for this RGB frame... Skipping size-based correction")
                non_depth_aval += 1
                continue
            """
            plt.imshow(dimage, cmap='Greys_r')
            plt.show()
            plt.imshow(cv2.imread(self.imglist[i]))
            plt.show()
            # origpcl = PathToPCL(self.dimglist[i], self.camera)
            o3d.visualization.draw_geometries([origpcl])
            """
            if foregroundextract:
                cluster_bw = extract_foreground_2D(dimage)
                #plt.imshow(cluster_bw, cmap='Greys_r')
                #plt.show()
                if cluster_bw is None: #problem with 2D clustering
                    #non_processed_pcls += 1
                    #print("There was a problem extracting a relevant 2D cluster for img no %i, skipping correction" % i)
                    #Revert to full image
                    obj_pcl = MatToPCL(dimage, self.camera, scale=self.scale)
                else: #mask depth img based on largest contour
                    masked_dmatrix = detect_contours(dimage,cluster_bw)
                    #plt.imshow(masked_dmatrix, cmap='Greys_r')
                    #plt.show()
                    obj_pcl = MatToPCL(masked_dmatrix, self.camera, scale=self.scale)
            else:
                obj_pcl = MatToPCL(dimage, self.camera, scale=self.scale)
            pcl_points = np.asarray(obj_pcl.points).shape[0]
            if pcl_points <=1:
                print("Empty pcl, skipping")
                non_processed_pcls += 1
                continue
            #o3d.visualization.draw_geometries([obj_pcl])
            if pclcluster:
                cluster_pcl = cluster_3D(obj_pcl)
                #o3d.visualization.draw_geometries([cluster_pcl])
                if cluster_pcl is None: #or with 3D clustering
                    print("There was a problem extracting a relevant 3D cluster for img no %i, proceeding with original pcl" % i)
                    cluster_pcl = obj_pcl #original pcl used instead

            else: cluster_pcl = obj_pcl
            try:
                d1,d2,depth,volume, orientedbox = estimate_dims(cluster_pcl,obj_pcl)
            except TypeError:
                print("Still not enough points..skipping")
                non_processed_pcls += 1
                continue

            if current_label != gt_label: # and abs(volume - pr_volume) > alpha*pr_volume :
                # If we detect a volume alpha times or more larger/smaller
                # than object predicted by baseline, then hypothesise object needs correction
                # actually need to correct, gives upper bound
                # TODO remove 1st check condition afterwards
                # create ranking by nearest volume
                # gt_volume = self.volumes[int(self.labels[i]) - 1]
                #plt.imshow(dimage, cmap='Greys_r')
                #plt.show()
                #plt.imshow(cv2.imread(self.imglist[i], cv2.IMREAD_UNCHANGED))  # , cmap='Greys_r')
                #plt.show()
                # o3d.visualization.draw_geometries([obj_pcl, orientedbox])
                if self.set =='KMi':
                    vol_ranking = pred_vol_proba(self,volume,dist='lognormal')
                else:
                    gt_dims = self.sizes[int(self.labels[i]) - 1]
                    #from predict import pred_by_size
                    # compare estimated dims with ground truth dims
                    # first possible permutation
                    dims_ranking = pred_by_size(self, np.array([d1,d2,depth]),i)
                    #second possible permutation
                    p2_rank = pred_by_size(self, np.array([d2, d1, depth]),i)
                    vol_ranking = pred_by_vol(self,volume, i)

                if self.set =='KMi' and not knowledge_only:
                    class_set = list(np.unique(current_ranking[:,0]))
                elif self.set =='KMi' and knowledge_only:
                    for h,cat in enumerate(list(vol_ranking['class'])):
                        clabel = self.mapper[cat]
                        final_rank[clabel] = vol_ranking['proba'][h]
                    #print(final_rank.most_common())
                    final_rank = final_rank.most_common(5)
                    winclass = self.remapper[final_rank[0][0]]
                    winproba = final_rank[0][1]
                    if winproba <= epsilon:
                        # no matching measurement could be found
                        # or not confident enough
                        # skip and fallback to vision ranking
                        nfallbacks +=1
                        #print("Fell back due to this result")
                        #print((winclass, winproba))
                        continue

                    try: pred_counts[winclass] +=1
                    except KeyError: pred_counts[winclass] =1
                    self.predictions[i, :] = final_rank
                    continue

                else: class_set = list(np.unique(current_min_ranking[:,0]))

                final_rank = Counter()
                if reranking:
                    clist = current_ranking.tolist()
                    vision_rank = Counter((self.remapper[k], score) for k, score in clist)
                    size_plausible_rank = Counter()
                    readable_rank = Counter() #same as above, but with readable labels
                    # if vision_rank includes class with near-zero prob do not include as plausible
                    for label,vision_score in vision_rank:
                        size_idx = np.argwhere(vol_ranking['class'] == label)[0][0]
                        size_proba = vol_ranking['proba'][size_idx]
                        if size_proba > epsilon:
                            try:
                                size_plausible_rank[self.mapper[label]] += vision_score
                                readable_rank[label] += vision_score
                            except KeyError:
                                size_plausible_rank[self.mapper[label]] = vision_score
                                readable_rank[label] = vision_score
                    # for each removed one in descending score, replace with one class from vol_ranking (descending order)
                    delta = len(vision_rank) - len(size_plausible_rank.keys())
                    if delta >0:
                        for d in range(delta):
                            # add class name based on size ranking
                            o,prob = vol_ranking[d]
                            if prob > epsilon:
                                # but add score based on vision similarity score
                                numeric_label = self.mapper[o]
                                #find first occurrence of that label in Vision predictions
                                tgti = np.where(all_predictions[i,:,0] == numeric_label)[0][0]
                                size_plausible_rank[numeric_label] = all_predictions[i,tgti,1]
                                readable_rank[o] = all_predictions[i,tgti,1]
                    # init final_rank with this modified ranking
                    # in the end, results will be re-sorted again, based on vision similarity score
                    final_rank = size_plausible_rank
                else:
                    for cname in class_set:
                        try:
                            if self.set =='KMi':
                                base_score = current_ranking[current_ranking[:, 0] == cname][:, 1][0]
                            else:
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
                                if self.set =='KMi':
                                    try:
                                        clabel = self.remapper[cname]
                                        vol_score = vol_ranking[vol_ranking['class'] == clabel]["proba"][0]
                                        if vol_score <=epsilon:
                                            #no size match or not confident enough for this class,fallback to Vision score
                                            final_rank[cname] = base_score
                                            continue
                                    except IndexError:
                                        #object is in Vision ranking but no size available
                                        # fallback to vision score
                                        final_rank[cname] = base_score
                                        continue
                                else:
                                    vol_score = vol_ranking[vol_ranking[:, 0] == cname][:, 1][0]

                                final_rank[cname] = sum([base_score, vol_score]) / 2

                            elif volOnly and novision:
                                if self.set == 'KMi': #only based on size prediction
                                    clabel = self.remapper[cname]
                                    vol_score = vol_ranking[vol_ranking['class']==clabel]["proba"][0]
                                    if vol_score <= epsilon:
                                        # no size match or not confident enough, fallback to Vision score
                                        final_rank[cname] = base_score
                                    else:
                                        final_rank[cname] = vol_score
                                else:
                                    final_rank[cname] = vol_ranking[vol_ranking[:, 0] == cname][:, 1][0]
                            elif not volOnly and novision:
                                dim_p1_score = dims_ranking[dims_ranking[:, 0] == cname][:, 1][0]
                                dim_p2_score = p2_rank[p2_rank[:, 0] == cname][:, 1][0]
                                final_rank[cname] = sum([dim_p1_score, dim_p2_score]) / 2
                            else:
                                dim_p1_score = dims_ranking[dims_ranking[:, 0] == cname][:, 1][0]
                                dim_p2_score = p2_rank[p2_rank[:, 0] == cname][:, 1][0]
                                final_rank[cname] = sum([base_score, dim_p1_score, dim_p2_score]) / 3

                if self.set =='KMi': final_rank = final_rank.most_common(5)
                else: final_rank = final_rank.most_common()[:-6:-1] # nearest 5 from new ranking
                delta = 5 - len(final_rank)
                if delta> 0:
                     final_rank.extend([(None,None) for i in range(delta)]) #fill up the space
                try:
                    winclass = self.remapper[final_rank[0][0]]
                except KeyError:
                    #no plausible re-ranking was found, fall back to original vision ranking/prediction
                    # skip correction
                    nfallbacks+=1
                    continue
                try: pred_counts[winclass] +=1
                except KeyError: pred_counts[winclass] =1
                self.predictions[i, :] = final_rank


        print("Took % fseconds." % float(time.time() - start)) #global proc time
        print("Re-evaluating post size correction...")
        if self.set == 'KMi':  # class-wise report
            pred_counts = list(pred_counts.most_common()) # used to check class imbalances in number of predictions
            eval_KMi(self, depth_aligned=True)
        else:  # separate eval for known vs novel
            eval_singlemodel(self)
        print("{} out of {} images need correction ".format(no_corrected,len(self.dimglist)))
        print("{} out of {} images were not corrected, no depth data available ".format(non_depth_aval,len(self.dimglist)))
        print("{} images {} not corrected due to processing issues".format(non_processed_pcls,len(self.dimglist)))
        print("for {} out of {} images size predictor was not confident enough, fallback to ML score".format(nfallbacks,len(self.dimglist)))
        return


