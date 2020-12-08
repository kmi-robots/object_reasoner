"""Main module."""
import open3d as o3d
import json
import os
import time
import sys
import statistics
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import preprocessing.utils as utl
import preprocessing.depth_img_processing as dimgproc
from preprocessing.rgb_img_processing import crop_test
import preprocessing.pcl_processing as pclproc
import predict as predictors
from evalscript import eval_singlemodel

class ObjectReasoner():
    def __init__(self, args):
        self.set = args.set
        self.verbose = args.verbose
        self.scenario = args.scenario
        self.baseline = args.baseline
        self.p_to_preds = args.preds
        start = time.time()
        self.obj_catalogue(args)
        print("Background KB initialized. Took %f seconds." % float(time.time() - start))
        # load metadata from txt files provided
        self.init_txt_files(args)
        self.init_depth_imglist(args)
        self.init_camera_intrinsics(args)
        # Load predictions from baseline algo
        start = time.time()
        self.init_ML_predictions(args)
        print("%s recognition results retrieved. Took %f seconds." % (args.baseline,float(time.time() - start)))

    def obj_catalogue(self,args):
        if self.set =='arc':
            try:
                with open('./data/arc_obj_catalogue_autom.json') as fin: #
                    self.KB = json.load(fin) #where the ground truth knowledge is
            except FileNotFoundError:
                print("Please copy the arc_obj_catalogue.json file under the ./data folder")
                sys.exit(0)
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
                self.volumes[i] = v['dimensions'][0] * v['dimensions'][1] * v['dimensions'][2]  # size catalogue in meters
                self.labelset.append(v['label'])
            self.mapper = dict((k, self.KB[k]['label']) for k in self.KB.keys())
        elif self.set == 'lab':
            try:
                with open('data/lab_obj_catalogue_autom_valid.json') as fin,\
                    open('data/lab_obj_catalogue.json') as fin2,\
                    open('data/Lab-set/class_to_index.json') as cin:
                    self.KB = json.load(fin) #where the ground truth knowledge is
                    #self.refKB = json.load(fin2) #manually sorted objects
                    self.mapper = json.load(cin)
            except FileNotFoundError:
                print("No lab catalogue or class-to-label index found - please refer to object_sizes.py for expected catalogue format")
                sys.exit(0)
        self.remapper = dict((v, k) for k, v in self.mapper.items())  # swap keys with indices

    def init_txt_files(self,args):
        with open(os.path.join(args.test_base,'test-labels.txt')) as txtf, \
            open(os.path.join(args.test_base,'test-product-labels.txt')) as prf, \
            open(os.path.join(args.test_base,'test-imgs.txt')) as imgf:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
            # test samples (each of 20 classes,10 known v 10 novel, chosen at random)
            self.plabels = prf.read().splitlines()       # product img labels
            if args.set =='lab': self.imglist = imgf.read().splitlines()
            else: self.imglist = [os.path.join(args.test_res,pp) for pp in imgf.read().splitlines()] #ARC support

    def init_depth_imglist(self,args):
        if args.set =='lab':
            """
                    lab set: Picked nearest depth frame to RGB image, based on timestamp.
                    Depth images are saved as 16-bit PNG, where depth values are saved in millimeters (10-3m).
                    Invalid depth is set to 0.
            """
            self.tsamples = None
            #if no depth imgs stored locally, generate from bag
            self.dimglist = utl.list_depth_filenames(os.path.join(args.test_base, 'test-imgs'))
            if self.dimglist is None: # create depth crops first
                if args.bags is None or not os.path.isdir(args.bags):
                    print("Print provide a valid path to the bag files storing depth data")
                    sys.exit(0)
                if args.regions is None or not os.path.isdir(args.bags):
                    print("Print provide a valid path to the region annotation files")
                    sys.exit(0)
                if os.path.isdir(args.origin):
                    #Are there any full size depth images already matched to rgb?
                    original_full_depth = [os.path.join(args.origin, fname) for fname in os.listdir(args.origin) if 'depth' in fname]
                if len(original_full_depth)>0:
                    #if yes, proceed with cropping directly
                    tempdimglist = ['_'.join(p.split('_')[:-1]) + 'depth_' + p.split('_')[-1] for p in self.imglist]
                    crop_test(original_full_depth, args.regions, os.path.join(args.test_base,'test-imgs'),'depth')
                    self.dimglist = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in tempdimglist]
                else:
                    #otherwise, start from temporal img matching too
                    # call python2 method from python3 script
                    print("Starting depth image extraction from bag files... It may take long to complete")
                    try:
                        utl.call_python_version("2.7", "bag_processing", "extract_from_bag", [self.imglist,args.bags,args.regions]) # retrieve data from bag
                    except Exception as e:
                        print(str(e))
                        sys.exit(0)
                    print("Depth files creation complete.. Imgs saved under %s" % os.path.join(args.test_base,'test-imgs'))
                    self.dimglist = [cv2.imread(p[:-4]+'depth.png', cv2.IMREAD_UNCHANGED) for p in self.imglist]
            else:
                tempdimglist = ['_'.join(p.split('_')[:-1]) + 'depth_' + p.split('_')[-1] for p in self.imglist]
                self.dimglist = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in tempdimglist]
            print("Empty depth files:")
            print("%s out of %s" % (len([d for d in self.dimglist if d is None]), len(self.dimglist)))
            self.scale = 1000.0 #depth values in mm

        else: #supports ARC set
            """
                    ARC set: Color images are saved as 24-bit RGB PNG.
                    Depth images and heightmaps are saved as 16-bit PNG, where depth values are saved in deci-millimeters (10-4m).
                    Invalid depth is set to 0.
                    Depth images are aligned to their corresponding color images.
            """
            self.dimglist = [cv2.imread(p.replace('color','depth'),cv2.IMREAD_UNCHANGED) for p in self.imglist]       # paths to test depth imgs
            self.scale = 10000.0  # depth values in deci-mm
            with open(os.path.join(args.test_base, 'test-other-objects-list.txt')) as smpf:
                self.tsamples = [l.split(',') for l in smpf.read().splitlines()]

    def init_camera_intrinsics(self, args):
        if not os.path.exists(os.path.join(args.test_base,'./camera-intrinsics.txt')) and\
            not os.path.exists(os.path.join(args.test_res,'./camera-intrinsics.txt')):
            bagpath = [os.path.join(args.bags, bagname) for bagname in os.listdir(args.bags) if bagname[-4:] == '.bag'][0]
            self.camintr = self.dimglist = utl.call_python_version("2.7", "bag_processing", "load_intrinsics",[bagpath,\
                                                                os.path.join(args.test_base,'./camera-intrinsics.txt')])
        elif not os.path.exists(os.path.join(args.test_base,'./camera-intrinsics.txt')) \
            and os.path.exists(os.path.join(args.test_res, './camera-intrinsics.txt')):
            self.camintr = utl.load_camera_intrinsics_txt(os.path.join(args.test_res, './camera-intrinsics.txt'))
        else:
            self.camintr = utl.load_camera_intrinsics_txt(os.path.join(args.test_base, './camera-intrinsics.txt'))

        self.camera = o3d.camera.PinholeCameraIntrinsic()
        self.camera.set_intrinsics(640, 480, self.camintr[0], self.camintr[4], self.camintr[2], self.camintr[5])

    def init_ML_predictions(self,args,fname='snapshot-test2-results.h5'):
        if not os.path.isfile(('%s/test_predictions_%s.npy' % (args.preds, args.baseline)))\
            and not os.path.isfile(('%s/test_predictions_%s.txt' % (args.preds, args.baseline))):
            if not os.path.isdir(args.preds):
                os.mkdir(args.preds)
            # # First time loading preds from raw embeddings
            self.kprod_emb, self.ktest_emb, self.nprod_emb, self.ntest_emb = utl.load_emb_space(args,fname)

            if args.baseline == 'two-stage': #both k-net and n-net pred are loaded
                self.predictions, self.predictions_B = predictors.pred_twostage(self, args)
            else:
                self.predictions = predictors.pred_singlemodel(self, args)
                self.predictions_B = None

            #Save predictions locally
            if self.predictions is not None and args.set=='lab':
                if args.baseline !='two-stage':
                    np.save(('%s/test_predictions_%s.npy' % (args.preds, args.baseline)), self.predictions)
                else:
                    np.save(('%s/test_predictions_%s.npy' % (args.preds, 'k-net')), self.predictions)
                    np.save(('%s/test_predictions_%s.npy' % (args.preds, 'n-net')), self.predictions_B)
            elif self.predictions is not None and args.set=='arc':
                if args.baseline !='two-stage':
                    with open(('%s/test_predictions_%s.txt' % (args.preds, args.baseline)),'wb') as fp:
                        pickle.dump(self.predictions, fp)
                else:
                    with open(('%s/test_predictions_%s.txt' % (args.preds, 'k-net')),'wb') as fp, \
                    open(('%s/test_predictions_%s.txt' % (args.preds, 'n-net')),'wb') as fp2:
                        pickle.dump(self.predictions, fp)
                        pickle.dump(self.predictions_B, fp2)
            else:
                print("Prediction mode not supported yet. Please choose a different one.")
                sys.exit(0)

        else: #Load from local npy file
            if args.set =='lab':
                if args.baseline != 'two-stage':
                    self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.preds, args.baseline)), allow_pickle=True)
                    self.predictions_B = None
                else:
                    self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.preds, 'k-net')),allow_pickle=True)
                    self.predictions_B = np.load(('%s/test_predictions_%s.npy' % (args.preds, 'n-net')),allow_pickle=True)
            else:
                if args.baseline != 'two-stage':
                    with open(('%s/test_predictions_%s.txt' % (args.preds, args.baseline)),'rb') as fp:
                        self.predictions = pickle.load(fp) #ARC set support
                        self.predictions_B = None
                else:
                    with open(('%s/test_predictions_%s.txt' % (args.preds, 'k-net')), 'rb') as fp,\
                    open(('%s/test_predictions_%s.txt' % (args.preds, 'n-net')), 'rb') as fp2:
                        self.predictions = pickle.load(fp)
                        self.predictions_B = pickle.load(fp2)

    def run(self, eval_dictionary):
        """ Evaluate ML predictions before hybrid reasoning"""
        print("Evaluating ML baseline...")
        eval_dictionary = eval_singlemodel(self,eval_dictionary, 'MLonly')
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'MLonly', K=5)

        print("Reasoning for correction ... ")
        """Data stats and monitoring vars for debugging"""
        start = time.time()
        non_processed_pcls = 0
        non_processed_fnames = []
        non_depth_aval = 0
        no_corrected =0
        nfallbacks = 0

        pclcluster = True
        all_predictions = self.predictions # copy to store all similarity scores, not just top 5
        all_predictions_B = self.predictions_B

        if self.set == 'lab':
            # segmented areas are higher quality and foreground extract is skipped
            foregroundextract = False
            self.predictions = self.predictions[:, :5, :]
            if self.predictions_B: self.predictions_B = self.predictions_B[:, :5, :]
            T= [-4.149075426919093,-2.776689935975939, -1.4043044450327855, -0.0319189540896323]
            lam = [-2.0244465762356794, -1.0759355070093815, -0.12742443778308354]

        elif self.set =='arc':
            foregroundextract = True
            self.predictions = [ar[:5,:] for ar in self.predictions] #only top-5 ranking
            if self.predictions_B: self.predictions_B = [ar[:5,:] for ar in self.predictions_B]
            T = [-7.739329757837735, -6.268319143699288, -4.797308529560841, -3.326297915422394]
            lam = [-3.5866347222619455, -2.5680992585358005, -1.5495637948096554]

        estimated_sizes = {}
        sizequal_copy = self.predictions.copy()
        flat_copy = self.predictions.copy()
        thin_copy = self.predictions.copy()
        thinAR_copy = self.predictions.copy()

        """For each depth image ... """
        for i,dimage in enumerate(self.dimglist):
            full_vision_rank_B, read_current_rank_B = None, None

            if self.set == 'lab':
                current_ranking = self.predictions[i, :]
                current_prediction = self.predictions[i, 0, 0]
                full_vision_rank = all_predictions[i, :]
            else:
                current_ranking = self.predictions[i]
                current_prediction = self.predictions[i][0][0]
                full_vision_rank = all_predictions[i]

            read_current_rank = [(self.remapper[current_ranking[z, 0]], current_ranking[z, 1]) for z in
                                 range(current_ranking.shape[0])]

            if self.predictions_B:
                current_ranking_B = self.predictions_B[i]
                current_prediction_B = self.predictions_B[i][0][0]
                current_label_B = self.remapper[current_prediction_B]
                full_vision_rank_B = all_predictions_B[i]

            current_label = self.remapper[current_prediction]
            gt_label = self.remapper[self.labels[i]]

            if current_label != gt_label:
                no_corrected += 1
            if dimage is None:
                # no synchronised depth data found
                print("No depth data available for this RGB frame... Skipping size-based correction")
                non_depth_aval += 1

            print("%s predicted as %s" % (gt_label, current_label))

            """# 1. ML prediction selection module"""
            if self.scenario == 'selected':
                if self.set=='lab' or self.baseline!='two-stage':
                    if current_label == 'empty':
                        sizeValidate=False
                    else:
                        sizeValidate,_= self.ML_predselection(read_current_rank,self.epsilon_set)
                else:# ARC case: two ML baselines could be leveraged
                    if current_label ==current_label_B or current_label=='empty':
                        # if there is agreement between the 2 Nets, retain predictions as-is
                        # or object categorised as empty (i.e., size is misleading)
                        sizeValidate=False
                        if current_label=='empty':
                            print("empty image, skipping size reasoning")
                        else: print("Agreement ..retain ML ranking")
                        print("ML based ranking - K-net")
                        print(read_current_rank)
                        print("================================")
                    else: #disagreement, proceed with size validation
                        sizeValidate = True

            elif self.scenario=='best':
                if current_label!= gt_label: sizeValidate = True
                else: sizeValidate = False
            elif self.scenario =='worst': sizeValidate = True

            if not sizeValidate:
                print("================================")
                continue #skip correction

            else: #current_label != gt_label: #if
                """Uncomment to visually inspect images/debug"""
                """plt.imshow(dimage, cmap='Greys_r')
                plt.show()
                plt.imshow(cv2.imread(self.imglist[i]))
                plt.title(gt_label+ " - "+self.imglist[i].split("/")[-1].split(".png")[0])
                plt.show()"""

                """ 2. Depth image to pointcloud conversion
                    3. OPTIONAL foreground extraction """
                obj_pcl = self.depth2PCL(dimage, foregroundextract)
                # o3d.visualization.draw_geometries([obj_pcl])
                pcl_points = np.asarray(obj_pcl.points).shape[0]
                if pcl_points <= 1:
                    print("Empty pcl, skipping")
                    non_processed_pcls += 1
                    # do not consider that obj region in eval
                    non_processed_fnames.append(self.imglist[i].split('/')[-1])
                    self.dimglist[i] = None
                    continue
                """ 4. noise removal """
                cluster_pcl = self.PCL_3Dprocess(obj_pcl, pclcluster)
                #cluster_pcl.paint_uniform_color(np.array([0., 0., 0.]))
                #o3d.visualization.draw_geometries([obj_pcl, cluster_pcl])
                """ 5. object size estimation """
                try:
                    d1, d2, d3, volume, orientedbox, aligned_box = pclproc.estimate_dims(cluster_pcl, obj_pcl)
                except TypeError:
                    print("Still not enough points..skipping")
                    non_processed_pcls += 1
                    # do not consider that obj region in eval
                    non_processed_fnames.append(self.imglist[i].split('/')[-1])
                    self.dimglist[i] = None
                    continue
                # o3d.visualization.draw_geometries([orig_pcl, cluster_pcl, orientedbox])
                try:
                    estimated_sizes[gt_label]['d1'].append(d1)
                    estimated_sizes[gt_label]['d2'].append(d2)
                    estimated_sizes[gt_label]['d3'].append(d3)
                except KeyError:
                    estimated_sizes[gt_label] = {}
                    estimated_sizes[gt_label]['d1'] = []
                    estimated_sizes[gt_label]['d2'] = []
                    estimated_sizes[gt_label]['d3'] = []

                    estimated_sizes[gt_label]['d1'].append(d1)
                    estimated_sizes[gt_label]['d2'].append(d2)
                    estimated_sizes[gt_label]['d3'].append(d3)

                print("Estimated dims oriented %f x %f x %f m" % (d1, d2, d3))

                res = self.size_reasoner(self.KB, dimage, [d1, d2, d3], T, lam)
                candidates_num, candidates_num_flat, candidates_num_thin, candidates_num_flatAR, candidates_num_thinAR = res
                valid_rank_flatAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flatAR for z in range(full_vision_rank.shape[0])]]
                valid_rank_thinAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thinAR for z in range(full_vision_rank.shape[0])]]

                valid_rank = full_vision_rank[[full_vision_rank[z, 0] in candidates_num for z in range(full_vision_rank.shape[0])]]
                valid_rank_flat = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flat for z in range(full_vision_rank.shape[0])]]
                valid_rank_thin = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thin for z in range(full_vision_rank.shape[0])]]

                if full_vision_rank_B is not None:  # or in other baseline rank (two-stage pipeline)
                    # add n-net predictions and resort by ascending distance
                    valid_rank_B = full_vision_rank_B[[full_vision_rank_B[z, 0] in candidates_num for z in range(full_vision_rank_B.shape[0])]]
                    valid_rank_B = valid_rank_B[np.argsort(valid_rank_B[:, 1])]

                    valid_rank_flat_B = full_vision_rank_B[[full_vision_rank_B[z, 0] in candidates_num_flat for z in range(full_vision_rank_B.shape[0])]]
                    valid_rank_flat_B = valid_rank_flat_B[np.argsort(valid_rank_flat_B[:, 1])]

                    valid_rank_thin_B = full_vision_rank_B[[full_vision_rank_B[z, 0] in candidates_num_thin for z in range(full_vision_rank_B.shape[0])]]
                    valid_rank_thin_B = valid_rank_thin_B[np.argsort(valid_rank_thin_B[:, 1])]

                    #Is the k-net prediction size-validated? Is the n-net one?
                    if (str(current_prediction) in candidates_num and str(current_prediction_B) in candidates_num) \
                        or (str(current_prediction) not in candidates_num and str(current_prediction_B) not in candidates_num):
                        print("Both or neither predictions are area-validated..picking most confident one")

                        topscore_Knet = valid_rank[0][1]
                        topscore_Nnet = valid_rank_B[0][1]
                        #conf thresh is class-wise and algorithm-wise
                        Kconf_thresh, Nconf_thresh = self.epsilon_set[0],self.epsilon_set[1]

                        #distance between top score and ideal epsilon as suggested by param tuner
                        dis_Knet = abs(topscore_Knet - Kconf_thresh)
                        dis_Nnet = abs(topscore_Nnet - Nconf_thresh)

                        if topscore_Nnet < Nconf_thresh and topscore_Knet >= Kconf_thresh:
                            print("N-net more confident")
                            valid_rank = valid_rank_B
                            valid_rank_flat = valid_rank_flat_B
                            valid_rank_thin = valid_rank_thin_B
                        elif topscore_Knet < Kconf_thresh and topscore_Nnet >= Nconf_thresh: print("K-net more confident") # keep ranking as-is/do nothing

                        elif topscore_Knet < Kconf_thresh and topscore_Nnet < Nconf_thresh: #both equally confident neither of the two is confident
                            #pick the one that is smallest compared to (i.e., most distant from) their ideal threshold
                            if dis_Knet>dis_Nnet: print("K-net more confident") # keep ranking as-is/do nothing
                            else:
                                print("N-net more confident")
                                valid_rank = valid_rank_B
                                valid_rank_flat = valid_rank_flat_B
                                valid_rank_thin = valid_rank_thin_B
                        else: #neither of the two is confident
                            if dis_Knet<dis_Nnet: print("K-net more confident") # keep ranking as-is/do nothing
                            else: #pick the one that is least distant from their ideal threshold
                                print("N-net more confident")
                                valid_rank = valid_rank_B
                                valid_rank_flat = valid_rank_flat_B
                                valid_rank_thin = valid_rank_thin_B

                    elif (str(current_prediction) not in candidates_num and str(current_prediction_B) in candidates_num):
                        print("Only N-net is size validated") # use N-net's validated ranking
                        valid_rank = valid_rank_B
                        valid_rank_flat = valid_rank_flat_B
                        valid_rank_thin = valid_rank_thin_B
                    else: print("Only K-net is size validated") # keep ranking as-is/do nothing

                    if candidates_num_flatAR is not None:
                        valid_rank_flatAR = full_vision_rank_B[[full_vision_rank_B[z, 0] in candidates_num_flatAR for z in range(full_vision_rank_B.shape[0])]]
                        valid_rank_flatAR = valid_rank_flatAR[np.argsort(valid_rank_flatAR[:, 1])]
                    if candidates_num_thinAR is not None:
                        valid_rank_thinAR= full_vision_rank_B[[full_vision_rank_B[z, 0] in candidates_num_thinAR for z in range(full_vision_rank_B.shape[0])]]
                        valid_rank_thinAR = valid_rank_thinAR[np.argsort(valid_rank_thinAR[:, 1])]

                #convert rankings to readable labels
                read_res = self.makereadable(valid_rank, valid_rank_flat, valid_rank_thin, valid_rank_flatAR, valid_rank_thinAR)
                read_rank, read_rank_flat, read_rank_thin, read_rank_flatAR, read_rank_thinAR = read_res

                if self.set =='lab':
                    if len(valid_rank_flatAR)>0: self.predictions[i, :] = valid_rank_flatAR[:5, :]
                    if len(valid_rank_flatAR)>0: thinAR_copy[i, :] = valid_rank_thinAR[:5, :]
                    thin_copy[i, :] = valid_rank_thin[:5, :]
                    sizequal_copy[i, :] = valid_rank[:5, :]  # _thin[:5,:]
                    flat_copy[i, :] = valid_rank_flat[:5, :]
                else: #ARC support
                    # len check in this case, because some of the size-validation predictions
                    # may be empty if the plausible candidates are not in the 20 sample classes
                    if len(valid_rank_flatAR)>0: self.predictions[i] = valid_rank_flatAR[:5, :]
                    if len(valid_rank_thinAR)>0: thinAR_copy[i] = valid_rank_thinAR[:5, :]
                    if len(valid_rank_thin)>0: thin_copy[i] = valid_rank_thin[:5, :]
                    if len(valid_rank)>0: sizequal_copy[i] = valid_rank[:5, :]
                    if len(valid_rank_flat)>0: flat_copy[i] = valid_rank_flat[:5, :]

                print("ML based ranking")
                print(read_current_rank[:5])
                print("Knowledge validated ranking (area)")
                print(read_rank[:5])

                if self.verbose:
                    print("Knowledge validated ranking (area + flat)")
                    print(read_rank_flat[:5])
                    print("Knowledge validated ranking (area + thin)")
                    print(read_rank_thin[:5])
                    if candidates_num_flatAR is not None and self.set!='arc':
                        print("Knowledge validated ranking (area + flat + AR)")
                        print(read_rank_flatAR[:5])

                    if candidates_num_thinAR is not None and self.set!='arc':
                        print("Knowledge validated ranking (area + thin + AR)")
                        print(read_rank_thinAR[:5])
                print("================================")

        print("Took % fseconds." % float(time.time() - start)) #global proc time
        print("Re-evaluating post size correction...")
        if self.verbose:
            """# Summary stats about predicted values"""
            size_summary = estimated_sizes.copy()
            for k in list(estimated_sizes.keys()):
                sub_dict = estimated_sizes[k]
                for subk, v in list(sub_dict.items()):
                    try:
                        size_summary[k]['mean-%s' % subk] = statistics.mean(v)
                    except:  # not enough data points
                        size_summary[k]['mean-%s' % subk] = None
                    try:
                        size_summary[k]['std-%s' % subk] = statistics.stdev(v)
                    except:  # not enough data points
                        size_summary[k]['std-%s' % subk] = None
                    size_summary[k]['min-%s' % subk] = min(v)
                    size_summary[k]['max-%s' % subk] = max(v)
            with open(os.path.join(self.p_to_preds,"../logged_stats.json"), 'w') as fout:
                json.dump(size_summary, fout)

        print("Knowledge-corrected (size qual + flat + AR)")
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+flat+AR')
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+flat+AR',K=5)
        print("Knowledge-corrected (size qual)")
        self.predictions = sizequal_copy
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area')
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area',K=5)
        print("Knowledge-corrected (size qual+flat)")
        self.predictions = flat_copy
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+flat')
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+flat',K=5)
        print("Knowledge-corrected (size qual+thin)")
        self.predictions = thin_copy
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+thin')
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+thin',K=5)
        print("Knowledge-corrected (size qual + thin + AR)")
        self.predictions = thinAR_copy
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+thin+AR')
        eval_dictionary = eval_singlemodel(self,eval_dictionary,'area+thin+AR',K=5)

        print("{} out of {} images need correction ".format(no_corrected,len(self.dimglist)))
        print("{} out of {} images were not corrected, no depth data available ".format(non_depth_aval,len(self.dimglist)))
        print("{} images {} not corrected due to processing issues".format(non_processed_pcls,len(self.dimglist)))
        print("Names of discarded img files: %s" % str(non_processed_fnames))
        print("for {} out of {} images size predictor was not confident enough, fallback to ML score".format(nfallbacks,len(self.dimglist)))
        return eval_dictionary

    def depth2PCL(self,dimage,foregroundextract):
        if foregroundextract:
            #extract foreground before converting
            cluster_bw = dimgproc.extract_foreground_2D(dimage)
            if cluster_bw is None:  # problem with 2D clustering
                # Revert to full image
                obj_pcl = pclproc.MatToPCL(dimage, self.camera, scale=self.scale)
            else:  # mask depth img based on largest contour
                masked_dmatrix = dimgproc.detect_contours(dimage, cluster_bw)
                obj_pcl = pclproc.MatToPCL(masked_dmatrix, self.camera, scale=self.scale)
        else:# just convert full image
            obj_pcl = pclproc.MatToPCL(dimage, self.camera, scale=self.scale)
        return obj_pcl

    def PCL_3Dprocess(self,obj_pcl,pclcluster):
        if pclcluster:
            # remove statistical outliers from pcl
            # obj_pcl = pclproc.pcl_remove_outliers(obj_pcl)
            # cluster_pcl = pclproc.cluster_3D(obj_pcl, downsample=False)
            cluster_pcl = pclproc.pcl_remove_outliers(obj_pcl)
            if cluster_pcl is None:
                # revert to full pcl instead
                cluster_pcl = obj_pcl
        else:
            cluster_pcl = obj_pcl
        return cluster_pcl

    def ML_predselection(self,read_current_rank,distance_ts):

        dis = read_current_rank[0][1]  # distance between test embedding and prod embedding
        conf_thresh = distance_ts[0]
        if dis < conf_thresh:  # lower distance/higher conf
            # ML is confident, keep as is
            #print("%s predicted as %s" % (gt_label, current_label))
            print("ML based ranking")
            print(read_current_rank)
            print("ML confident, skipping size-based validation")
            print("================================")
            return (False,None)
        else:
            return (True,None)

    def size_reasoner(self,KB,dimage, estimated_dims,T,lam):
        depth = min(estimated_dims)
        estimated_dims.remove(depth)
        d1, d2 = estimated_dims

        """ 6. size quantization """
        qual = predictors.pred_size_qual(d1, d2, thresholds=T)
        flat = predictors.pred_flat(depth, len_thresh=lam[0])
        flat_flag = 'flat' if flat else 'non flat'
        # Aspect ratio based on crop
        aspect_ratio = predictors.pred_AR(dimage.shape, (d1, d2))
        thinness = predictors.pred_thinness(depth, cuts=lam)
        cluster = qual + "-" + thinness

        print("Detected size is %s" % qual)
        print("Object is %s" % flat_flag)
        print("Object is %s" % aspect_ratio)
        print("Object is %s" % thinness)

        """ 7. Hybrid (area) """
        candidates = [oname for oname in KB.keys() if qual in str(
            KB[oname]["has_size"])]  # len([s for s in self.KB[oname]["has_size"] if s.startswith(qual)])>0]
        candidates_num = [self.mapper[oname.replace(' ', '_')] for oname in candidates]

        """ 7. Hybrid (area + flat) """
        candidates_flat = [oname for oname in candidates if str(flat) in str(KB[oname]["is_flat"])]
        candidates_num_flat = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat]

        """ 7. Hybrid (area + thin) """
        try:
            candidates_thin = [oname for oname in candidates if thinness in str(KB[oname]["thinness"])]

        except KeyError:  # annotation format variation
            candidates_thin = [oname for oname in candidates if thinness in str(KB[oname]["has_size"])]

        candidates_num_thin = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin]

        """7. Hybrid (area + flat+AR) """
        candidates_flat_AR = [oname for oname in candidates_flat if aspect_ratio in str(KB[oname]["aspect_ratio"])]
        candidates_num_flatAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat_AR]

        """ 7. Hybrid (area + thin +AR) """
        candidates_thin_AR = [oname for oname in candidates_thin if aspect_ratio in str(KB[oname]["aspect_ratio"])]
        candidates_num_thinAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin_AR]

        return [candidates_num, candidates_num_flat, candidates_num_thin, candidates_num_flatAR, candidates_num_thinAR]

    def makereadable(self,valid_rank, valid_rank_flat, valid_rank_thin, valid_rank_flatAR, valid_rank_thinAR):

        read_rank = [(self.remapper[valid_rank[z, 0]], valid_rank[z, 1]) for z in
                     range(valid_rank.shape[0])]
        read_rank_flat = [(self.remapper[valid_rank_flat[z, 0]], valid_rank_flat[z, 1]) for z in
                          range(valid_rank_flat.shape[0])]
        read_rank_thin = [(self.remapper[valid_rank_thin[z, 0]], valid_rank_thin[z, 1]) for z in
                          range(valid_rank_thin.shape[0])]
        if len(valid_rank_flatAR)>0:
            read_rank_flatAR = [(self.remapper[valid_rank_flatAR[z, 0]], valid_rank_flatAR[z, 1]) for z in
                                range(valid_rank_flatAR.shape[0])]
        else: read_rank_flatAR =[]
        if len(valid_rank_thinAR)>0:
            read_rank_thinAR = [(self.remapper[valid_rank_thinAR[z, 0]], valid_rank_thinAR[z, 1]) for z in
                            range(valid_rank_thinAR.shape[0])]
        else: read_rank_thinAR=[]

        return [read_rank, read_rank_flat, read_rank_thin, read_rank_flatAR, read_rank_thinAR]

