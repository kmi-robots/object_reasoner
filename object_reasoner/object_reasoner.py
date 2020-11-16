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
from evalscript import eval_singlemodel, eval_KMi

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
        elif self.set == 'KMi':
            try:
                with open('./data/KMi_obj_catalogue_autom_valid.json') as fin,\
                    open('./data/KMi_obj_catalogue.json') as fin2,\
                    open('./data/KMi-set-2020/class_to_index.json') as cin:
                    self.KB = json.load(fin) #where the ground truth knowledge is
                    self.refKB = json.load(fin2) #manually sorted objects
                    self.mapper = json.load(cin)
            except FileNotFoundError:
                print("No KMi catalogue or class-to-label index found - please refer to object_sizes.py for expected catalogue format")
                sys.exit(0)
        self.remapper = dict((v, k) for k, v in self.mapper.items())  # swap keys with indices

    def init_txt_files(self,args):
        with open(os.path.join(args.test_base,'test-labels.txt')) as txtf, \
            open(os.path.join(args.test_base,'test-product-labels.txt')) as prf, \
            open(os.path.join(args.test_base,'test-imgs.txt')) as imgf:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
            # test samples (each of 20 classes,10 known v 10 novel, chosen at random)
            self.plabels = prf.read().splitlines()       # product img labels
            if args.set =='KMi': self.imglist = imgf.read().splitlines()
            else: self.imglist = [os.path.join(args.test_res,pp) for pp in imgf.read().splitlines()] #ARC support

    def init_depth_imglist(self,args):
        if args.set =='KMi':
            """
                    KMi set: Picked nearest depth frame to RGB image, based on timestamp.
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
            if self.predictions is not None and args.set=='KMi':
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
            if args.set =='KMi':
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

    def run(self):
        """ Evaluate ML predictions before hybrid reasoning"""
        print("Double checking top-1 accuracies for ML baseline...")
        if self.set == 'KMi':  # class-wise report
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True, K=5)
        else:  # separate eval for known vs novel
            eval_singlemodel(self)
            eval_singlemodel(self,K=5)

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

        if self.set == 'KMi':
            # segmented areas are higher quality and foreground extract is skipped
            foregroundextract = False
            self.predictions = self.predictions[:, :5, :]
            if self.predictions_B: self.predictions_B = self.predictions_B[:, :5, :]
            T= [-4.149075426919093,-2.776689935975939, -1.4043044450327855, -0.0319189540896323]#[-4.149075426919093, -2.776689935975939, -1.4043044450327855, -0.0319189540896323] #[0.007, 0.05, 0.35, 0.79]
            T_view2 = None #[-4.304882321457492, -3.0883037250527376, -1.8717251286479835, -0.6551465322432293]
            T_view3 = None #[-5.693662433822248, -4.048786672569905, -2.4039109113175616, -0.7590351500652188]
            lam = [-2.0244465762356794, -1.0759355070093815, -0.12742443778308354] #[-3.3145883831019756, -2.02400658021586, -0.7334247773297449]#[-4.3963762539301765, -2.7451984941013277, -1.0940207342724788] #[0.1, 0.2, 0.4]
            lam_view2 = None #[-2.583024931358944, -1.477170408603952, -0.37131588584896]
            lam_view3 = None #[-2.0244465762356794, -1.0759355070093815, -0.12742443778308354]
            epsilon = 0.040554 #0.04
            N=3

        elif self.set =='arc':
            foregroundextract = True
            self.predictions = [ar[:5,:] for ar in self.predictions] #only top-5 ranking
            if self.predictions_B: self.predictions_B = [ar[:5,:] for ar in self.predictions_B]
            T =  [-7.739329757837735, -6.268319143699288, -4.797308529560841, -3.326297915422394] #[0.0085, 0.01326, 0.0208, 0.033]
            T_view2 =  None #[0.0021,0.0037,0.0049,0.0092] #objects can be grasped from different angles
            T_view3 = None #[0.00203,0.0037,0.0066,0.013]
            lam = [-3.5866347222619455, -2.5680992585358005, -1.5495637948096554] #[0.022,0.033,0.063]
            lam_view2 = None #[0.075,0.126,0.185] #objects can be grasped from different angles
            lam_view3 = None #[0.083,0.11,0.16]
            epsilon = (0.033357,0.021426) #(Nnet, Knet) #0.03
            if self.set == 'arc' and self.baseline == 'k-net':
                epsilon = epsilon[1]
            elif self.set == 'arc' and self.baseline == 'n-net':
                epsilon = epsilon[0]
            N = 3

        estimated_sizes = {}
        sizequal_copy = self.predictions.copy()
        flat_copy = self.predictions.copy()
        thin_copy = self.predictions.copy()
        thinAR_copy = self.predictions.copy()

        """For each depth image ... """
        for i,dimage in enumerate(self.dimglist):
            full_vision_rank_B, read_current_rank_B = None, None

            if self.set == 'KMi':
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
                read_current_rank_B = [(self.remapper[current_ranking_B[z, 0]], current_ranking_B[z, 1]) for z in
                                 range(current_ranking_B.shape[0])]

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
                if self.set=='KMi' or self.baseline!='two-stage':
                    if current_label == 'empty':
                        sizeValidate=False
                    else:
                        sizeValidate,_= self.ML_predselection(read_current_rank,current_label,gt_label,distance_t=epsilon,n=N)
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

                if T_view2 is None: #single view/config case, e.g., KMi set case
                    #results with autom generated KB
                    res = self.size_reasoner_singleview(self.KB, dimage, [d1, d2, d3], T, lam)
                    candidates_num, candidates_num_flat, candidates_num_thin, candidates_num_flatAR, candidates_num_thinAR = res
                    valid_rank_flatAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flatAR for z in range(full_vision_rank.shape[0])]]
                    valid_rank_thinAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thinAR for z in range(full_vision_rank.shape[0])]]

                    """"# results with manual sorting KB
                    res = self.size_reasoner_singleview(self.refKB, dimage, [d1, d2, d3], T, lam)
                    candidates_num2, candidates_num_flat2, candidates_num_thin2, candidates_num_flatAR2, candidates_num_thinAR2 = res
                    valid_rank2 = full_vision_rank[[full_vision_rank[z, 0] in candidates_num2 for z in range(full_vision_rank.shape[0])]]
                    valid_rank_flat2 = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flat2 for z in range(full_vision_rank.shape[0])]]
                    valid_rank_thin2 = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thin2 for z in range(full_vision_rank.shape[0])]]
                    valid_rank_flatAR2 = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flatAR2 for z in range(full_vision_rank.shape[0])]]
                    valid_rank_thinAR2 = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thinAR2 for z in range(full_vision_rank.shape[0])]]
                    """
                else: #multi-view case, e.g., ARC set
                    res = self.size_reasoner_multiview((d1, d2, d3), (T, T_view2, T_view3),(lam, lam_view2, lam_view3))
                    candidates_num,candidates_num_flat,candidates_num_thin = res
                    candidates_num_flatAR, candidates_num_thinAR = None, None # Aspect Ratio is not relevant if multiple orientations are possible
                    valid_rank_flatAR, valid_rank_thinAR = [], []

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

                        #distance between top score and ideal epsilon as suggested by param tuner
                        dis_Knet = abs(topscore_Knet - epsilon[1])
                        dis_Nnet = abs(topscore_Nnet - epsilon[0])

                        if topscore_Nnet < epsilon[0] and topscore_Knet > epsilon[1]:
                            print("N-net more confident")
                            valid_rank = valid_rank_B
                            valid_rank_flat = valid_rank_flat_B
                            valid_rank_thin = valid_rank_thin_B
                        elif topscore_Knet < epsilon[1] and topscore_Nnet > epsilon[0]: print("K-net more confident") # keep ranking as-is/do nothing
                        elif topscore_Nnet<epsilon[0] and topscore_Knet<epsilon[1]: #equally confident
                            #pick the one that is smallest compared to (i.e., most distant from) their ideal threshold
                            if dis_Knet>dis_Nnet: print("K-net more confident") # keep ranking as-is/do nothing
                            else:
                                print("N-net more confident")
                                valid_rank = valid_rank_B
                                valid_rank_flat = valid_rank_flat_B
                                valid_rank_thin = valid_rank_thin_B
                        else: #neither is confident
                            #pick the one with the least distance from ideal threshold
                            if dis_Knet<dis_Nnet: print("K-net more confident") # keep ranking as-is/do nothing
                            else:
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

                #read_res2 = self.makereadable(valid_rank2, valid_rank_flat2, valid_rank_thin2, valid_rank_flatAR2, valid_rank_thinAR2)
                #read_rank2, read_rank_flat2, read_rank_thin2, read_rank_flatAR2, read_rank_thinAR2 = read_res2

                if self.set =='KMi':
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
                #print("**Ranking with manual sorting KB:**")
                #print("Knowledge validated ranking (area)")
                #print(read_rank2[:5])
                if self.verbose:
                    print("Knowledge validated ranking (area + flat)")
                    print(read_rank_flat[:5])
                    #print("**Ranking with manual sorting KB:**")
                    #print("Knowledge validated ranking (area + flat)")
                    #print(read_rank_flat2[:5])
                    print("Knowledge validated ranking (area + thin)")
                    print(read_rank_thin[:5])
                    #print("**Ranking with manual sorting KB:**")
                    #print("Knowledge validated ranking (area + thin)")
                    #print(read_rank_thin2[:5])
                    if candidates_num_flatAR is not None and self.set!='arc':
                        print("Knowledge validated ranking (area + flat + AR)")
                        print(read_rank_flatAR[:5])
                        #print("**Ranking with manual sorting KB:**")
                        #print("Knowledge validated ranking (area + flat + AR)")
                        #print(read_rank_flatAR2[:5])
                    if candidates_num_thinAR is not None and self.set!='arc':
                        print("Knowledge validated ranking (area + thin + AR)")
                        print(read_rank_thinAR[:5])
                        #print("**Ranking with manual sorting KB:**")
                        #print("Knowledge validated ranking (area + thin + AR)")
                        #print(read_rank_thinAR2[:5])
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
        if self.set == 'KMi':
            print("Knowledge-corrected (size qual + flat + AR)")
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True,K=5)
            print("Knowledge-corrected (size qual)")
            self.predictions = sizequal_copy
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True, K=5)
            #print("Knowledge-corrected (size qual+prop)")
            print("Knowledge-corrected (size qual+flat)")
            self.predictions = flat_copy
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True, K=5)

            print("Knowledge-corrected (size qual+thin)")
            self.predictions = thin_copy
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True, K=5)
            print("Knowledge-corrected (size qual + thin + AR)")
            self.predictions = thinAR_copy
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True, K=5)

        else:  # separate eval for known vs novel
            print("Knowledge-corrected (size qual + flat + AR)")
            eval_singlemodel(self)
            eval_singlemodel(self,K=5)
            print("Knowledge-corrected (size qual)")
            self.predictions = sizequal_copy
            eval_singlemodel(self)
            eval_singlemodel(self, K=5)
            # print("Knowledge-corrected (size qual+prop)")
            print("Knowledge-corrected (size qual+flat)")
            self.predictions = flat_copy
            eval_singlemodel(self)
            eval_singlemodel(self, K=5)
            print("Knowledge-corrected (size qual+thin)")
            self.predictions = thin_copy
            eval_singlemodel(self)
            eval_singlemodel(self, K=5)
            print("Knowledge-corrected (size qual + thin + AR)")
            self.predictions = thinAR_copy
            eval_singlemodel(self)
            eval_singlemodel(self, K=5)

        print("{} out of {} images need correction ".format(no_corrected,len(self.dimglist)))
        print("{} out of {} images were not corrected, no depth data available ".format(non_depth_aval,len(self.dimglist)))
        print("{} images {} not corrected due to processing issues".format(non_processed_pcls,len(self.dimglist)))
        print("Names of discarded img files: %s" % str(non_processed_fnames))
        print("for {} out of {} images size predictor was not confident enough, fallback to ML score".format(nfallbacks,len(self.dimglist)))
        return

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

    def ML_predselection(self,read_current_rank,current_label,gt_label,distance_t=0.04,n=3):

        MLclasses = [l[0] for l in read_current_rank]
        l_, c_ = Counter(MLclasses).most_common()[0]
        dis = read_current_rank[0][1]  # distance between test embedding and prod embedding
        if dis < distance_t and c_ >= n:  # lower distance/higher conf and class appears at least three times
            # ML is confident, keep as is
            #print("%s predicted as %s" % (gt_label, current_label))
            print("ML based ranking")
            print(read_current_rank)
            print("ML confident, skipping size-based validation")
            print("================================")
            return (False,None)
        else:
            return (True,None)

    def size_reasoner_singleview(self,KB,dimage, estimated_dims,T,lam):
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

        # candidates_thin = [oname for oname in candidates if thinness in str(self.KB[oname]["has_size"])]
        # candidates_thin = [oname for oname in self.KB.keys() if cluster in self.KB[oname]["has_size"]]
        # candidates_thin = [oname for oname in self.KB.keys() if (qual in self.KB[oname]["has_size"]
        #                     and thinness in str(self.KB[oname]["thinness"]))]
        candidates_num_thin = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin]

        """7. Hybrid (area + flat+AR) """
        candidates_flat_AR = [oname for oname in candidates_flat if aspect_ratio in str(KB[oname]["aspect_ratio"])]
        candidates_num_flatAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat_AR]

        """ 7. Hybrid (area + thin +AR) """
        candidates_thin_AR = [oname for oname in candidates_thin if aspect_ratio in str(KB[oname]["aspect_ratio"])]
        candidates_num_thinAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin_AR]

        return [candidates_num, candidates_num_flat, candidates_num_thin, candidates_num_flatAR, candidates_num_thinAR]

    def size_reasoner_multiview(self, estimated_dims, area_thresholds, depth_thresholds):
        """Size reasoning in a case where multiple orientation configurations can be observed
        """
        d1,d2,d3 = estimated_dims
        T1,T2,T3 = area_thresholds
        lam1,lam2,lam3 = depth_thresholds

        """ 6. size quantization (config 1)"""
        qual = predictors.pred_size_qual(d1, d2, thresholds=T1)
        flat = predictors.pred_flat(d3, len_thresh=lam1[0])
        thinness = predictors.pred_thinness(d3, cuts=lam1)
        cluster = qual + "-" + thinness
        """ 6. size quantization (config 2)"""
        qual2 = predictors.pred_size_qual(d1, d3, thresholds=T2)
        flat2 = predictors.pred_flat(d2, len_thresh=lam2[0])
        thinness2 = predictors.pred_thinness(d2, cuts=lam2)
        cluster2 = qual2 + "-" + thinness2
        """ 6. size quantization (config 3)"""
        qual3 = predictors.pred_size_qual(d2, d3, thresholds=T3)
        flat3 = predictors.pred_flat(d1, len_thresh=lam3[0])
        thinness3 = predictors.pred_thinness(d1, cuts=lam3)
        cluster3 = qual3 + "-" + thinness3

        print("Detected size is %s (config 1), %s (config 2), or %s (config 3)" % (qual,qual2,qual3))
        print("Object is %s (config 1), %s (config 2), or %s (config 3)" % (str(flat),str(flat2),str(flat3)))
        print("Object is %s (config 1), %s (config 2), or %s (config 3)" % (thinness,thinness2,thinness3))

        """ 7. Hybrid (area) """
        candidates = [oname for oname in self.KB.keys() if
                      len([s for s in self.KB[oname]["has_size"]
                           if (s.startswith(qual) or s.startswith(qual2) or s.startswith(qual3)) ])>0 ]
        candidates_num = [self.mapper[oname.replace(' ', '_')] for oname in candidates]


        """ 7. Hybrid (area +flat) """
        candidates_flat = [oname for oname in candidates if
                           str(flat) in str(self.KB[oname]["is_flat"])
                           or str(flat2) in str(self.KB[oname]["is_flat"])
                           or str(flat3) in str(self.KB[oname]["is_flat"])]
        candidates_num_flat = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat]

        """ 7. Hybrid (area + thin) """
        candidates_thin = [oname for oname in self.KB.keys() if
                           cluster in self.KB[oname]["has_size"]
                           or cluster2 in self.KB[oname]["has_size"]
                           or cluster3 in self.KB[oname]["has_size"]]
        candidates_num_thin = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin]

        return [candidates_num,candidates_num_flat,candidates_num_thin]

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

