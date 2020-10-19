"""Main module."""
import open3d as o3d
import json
import os
import time
import sys
import statistics
import numpy as np
import cv2
from collections import Counter
import preprocessing.utils as utl
import preprocessing.depth_img_processing as dimgproc
import preprocessing.pcl_processing as pclproc
import predict as predictors
from evalscript import eval_singlemodel, eval_KMi

class ObjectReasoner():
    def __init__(self, args):
        self.set = args.set
        self.verbose = args.verbose
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
                with open('./data/arc_obj_catalogue.json') as fin:
                    self.KB = json.load(fin) #where the ground truth knowledge is
            except FileNotFoundError:
                self.KB = utl.init_obj_catalogue(args.test_base)
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
                self.volumes[i] = v['dimensions'][0] * v['dimensions'][1] * v['dimensions'][2]  # size catalogue in meters
                self.labelset.append(v['label'])
        elif self.set == 'KMi':
            try:
                with open('./data/KMi_obj_catalogue.json') as fin,\
                    open('./data/KMi-set-2020/class_to_index.json') as cin:
                    self.KB = json.load(fin) #where the ground truth knowledge is
                    self.mapper = json.load(cin)
                    self.remapper =dict((v, k) for k, v in self.mapper.items())  # swap keys with indices
            except FileNotFoundError:
                print("No KMi catalogue or class-to-label index found - please refer to object_sizes.py for expected catalogue format")
                sys.exit(0)

    def init_txt_files(self,args):
        with open(os.path.join(args.test_base,'test-labels.txt')) as txtf, \
            open(os.path.join(args.test_base,'test-product-labels.txt')) as prf, \
            open(os.path.join(args.test_base,'test-imgs.txt')) as imgf:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
            # test samples (each of 20 classes,10 known v 10 novel, chosen at random)
            self.plabels = prf.read().splitlines()       # product img labels
            self.imglist = imgf.read().splitlines()

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
        if not os.path.isfile(('%s/test_predictions_%s.npy' % (args.preds, args.baseline))):
            if not os.path.isdir(args.preds):
                os.mkdir(args.preds)
            # # First time loading preds from raw embeddings
            self.kprod_emb, self.ktest_emb, self.nprod_emb, self.ntest_emb = utl.load_emb_space(args,fname)
            if args.baseline == 'two-stage':
                self.predictions = predictors.pred_twostage(self, args)
            else:
                self.predictions, _,_ = predictors.pred_singlemodel(self, args)
            if self.predictions is not None:
                np.save(('%s/test_predictions_%s.npy' % (args.preds, args.baseline)), self.predictions)
            else:
                print("Prediction mode not supported yet. Please choose a different one.")
                sys.exit(0)
        else: #Load from local npy file
            self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.preds, args.baseline)), allow_pickle=True)

    def run(self):
        """ Evaluate ML predictions before hybrid reasoning"""
        print("Double checking top-1 accuracies for ML baseline...")
        if self.set == 'KMi':  # class-wise report
            eval_KMi(self, depth_aligned=True)
            eval_KMi(self, depth_aligned=True, K=5)
        else:  # separate eval for known vs novel
            eval_singlemodel(self)

        print("Reasoning for correction ... ")
        """Data stats and monitoring vars for debugging"""
        start = time.time()
        non_processed_pcls = 0
        non_processed_fnames = []
        non_depth_aval = 0
        no_corrected =0
        nfallbacks = 0

        pclcluster = True
        if self.set == 'KMi':
            # segmented areas are higher quality and foreground extract is skipped
            foregroundextract = False
            all_predictions = self.predictions  # copy to store all similarity scores, not just top 5
            self.predictions = self.predictions[:, :5, :]
        elif self.set =='arc':
            foregroundextract = True

        pred_counts = Counter()
        need_corr_by_class = Counter()
        all_gt_labels = [self.remapper[l] for i,l in enumerate(self.labels) if self.dimglist[i] is not None]
        supports = Counter(all_gt_labels)
        estimated_vols = {}
        estimated_sizes = {}

        sizequal_copy = self.predictions.copy()
        flat_copy = self.predictions.copy()
        thin_copy = self.predictions.copy()
        thinAR_copy = self.predictions.copy()

        """For each depth image ... """
        for i,dimage in enumerate(self.dimglist):
            # baseline predictions as (label, distance)
            current_ranking = self.predictions[i, :]
            current_prediction = self.predictions[i, 0, 0]
            if self.set == 'KMi':
                current_label = self.remapper[current_prediction]
                gt_label = self.remapper[self.labels[i]]
            else:
                current_label = list(self.KB.keys())[current_prediction]
                gt_label = list(self.KB.keys())[int(self.labels[i]) - 1]

            if current_label != gt_label:
                no_corrected += 1
            if dimage is None:
                # no synchronised depth data found
                print("No depth data available for this RGB frame... Skipping size-based correction")
                non_depth_aval += 1
                continue

            """Uncomment to visually inspect images/debug"""
            #plt.imshow(dimage, cmap='Greys_r')
            #plt.show()
            #plt.imshow(cv2.imread(self.imglist[i]))
            #plt.title(gt_label+ " - "+self.imglist[i].split("/")[-1].split(".png")[0])
            #plt.show()

            """ 1. Depth image to pointcloud conversion
                2. OPTIONAL foreground extraction """
            obj_pcl = self.depth2PCL(dimage,foregroundextract)
            pcl_points = np.asarray(obj_pcl.points).shape[0]
            if pcl_points <= 1:
                print("Empty pcl, skipping")
                non_processed_pcls += 1
                # do not consider that obj region in eval
                non_processed_fnames.append(self.imglist[i].split('/')[-1])
                self.dimglist[i] = None
                continue
            """ 2. noise removal """
            cluster_pcl = self.PCL_3Dprocess(obj_pcl,pclcluster)
            #cluster_pcl.paint_uniform_color(np.array([0., 0., 0.]))
            #o3d.visualization.draw_geometries([orig_pcl, cluster_pcl])
            """ 3. object size estimation """
            try:
                d1, d2, depth, volume, orientedbox,aligned_box = pclproc.estimate_dims(cluster_pcl, obj_pcl)
            except TypeError:
                print("Still not enough points..skipping")
                non_processed_pcls += 1
                # do not consider that obj region in eval
                non_processed_fnames.append(self.imglist[i].split('/')[-1])
                self.dimglist[i] = None
                continue
            #o3d.visualization.draw_geometries([orig_pcl, cluster_pcl, orientedbox])
            try:
                estimated_sizes[gt_label]['d1'].append(d1)
                estimated_sizes[gt_label]['d2'].append(d2)
                estimated_sizes[gt_label]['depth'].append(depth)
            except KeyError:
                estimated_sizes[gt_label] = {}
                estimated_sizes[gt_label]['d1'] = []
                estimated_sizes[gt_label]['d2'] = []
                estimated_sizes[gt_label]['depth'] = []

                estimated_sizes[gt_label]['d1'].append(d1)
                estimated_sizes[gt_label]['d2'].append(d2)
                estimated_sizes[gt_label]['depth'].append(depth)

            """# 4. ML prediction selection module"""
            full_vision_rank = all_predictions[i, :]
            sizeValidate,read_current_rank = self.MLpred_selection(current_ranking,current_label,gt_label)

            if not sizeValidate: continue #skip correction
            else: #current_label != gt_label: #if

                """ 5. size quantization """
                qual = predictors.pred_size_qual(d1,d2)
                flat = predictors.pred_flat(d1, d2, depth)
                flat_flag = 'flat' if flat else 'non flat'
                #Aspect ratio based on crop
                aspect_ratio = predictors.pred_AR(dimage.shape,(d1,d2))
                thinness = predictors.pred_thinness(depth)

                """ 6. Hybrid (area) """
                candidates = [oname for oname in self.KB.keys() if qual in self.KB[oname]["has_size"]]
                candidates_num = [self.mapper[oname.replace(' ', '_')] for oname in candidates]
                valid_rank = full_vision_rank[[full_vision_rank[z, 0] in candidates_num for z in range(full_vision_rank.shape[0])]]
                read_rank = [(self.remapper[valid_rank[z, 0]], valid_rank[z, 1]) for z in range(valid_rank.shape[0])]

                """ 6. Hybrid (area + flat) """
                candidates_flat = [oname for oname in self.KB.keys() if
                                   (qual in self.KB[oname]["has_size"] and str(flat) in str(self.KB[oname]["is_flat"]))]
                candidates_num_flat = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat]
                valid_rank_flat = full_vision_rank[
                    [full_vision_rank[z, 0] in candidates_num_flat for z in range(full_vision_rank.shape[0])]]
                read_rank_flat = [(self.remapper[valid_rank_flat[z, 0]], valid_rank_flat[z, 1]) for z in
                                  range(valid_rank_flat.shape[0])]

                """ 6. Hybrid (area + thin) """
                candidates_thin = [oname for oname in self.KB.keys() if (qual in self.KB[oname]["has_size"]
                                                                        and thinness in str(self.KB[oname]["thinness"]))]
                candidates_num_thin = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin]
                valid_rank_thin = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thin for z in range(full_vision_rank.shape[0])]]
                read_rank_thin = [(self.remapper[valid_rank_thin[z,0]],valid_rank_thin[z,1]) for z in range(valid_rank_thin.shape[0])]

                """ 6. Hybrid (area + flat+AR) """

                candidates_flat_AR = [oname for oname in self.KB.keys() if (qual in self.KB[oname]["has_size"]
                                    and str(flat) in str(self.KB[oname]["is_flat"])
                                    and aspect_ratio in str(self.KB[oname]["aspect_ratio"]))]
                candidates_num_flatAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat_AR]
                valid_rank_flatAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flatAR for z in range(full_vision_rank.shape[0])]]
                read_rank_flatAR = [(self.remapper[valid_rank_flatAR[z,0]],valid_rank_flatAR[z,1]) for z in range(valid_rank_flatAR.shape[0])]

                """ 6. Hybrid (area + thin +AR) """
                candidates_thin_AR = [oname for oname in self.KB.keys() if (qual in self.KB[oname]["has_size"]
                            and str(thinness) in str(self.KB[oname]["thinness"]) and aspect_ratio in str(self.KB[oname]["aspect_ratio"]))]
                candidates_num_thinAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin_AR]
                valid_rank_thinAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thinAR for z in range(full_vision_rank.shape[0])]]
                read_rank_thinAR = [(self.remapper[valid_rank_thinAR[z, 0]], valid_rank_thinAR[z, 1]) for z in range(valid_rank_thinAR.shape[0])]

                self.predictions[i, :] = valid_rank_flatAR[:5, :]
                thinAR_copy[i, :] = valid_rank_thinAR[:5, :]
                thin_copy[i, :] = valid_rank_thin[:5, :]
                sizequal_copy[i, :] = valid_rank[:5, :]  # _thin[:5,:]
                flat_copy[i, :] = valid_rank_flat[:5, :]

                print("%s predicted as %s" % (gt_label,current_label))
                print("Detected size is %s" % qual)
                print("Object is %s" % flat_flag)
                print("Object is %s" % aspect_ratio)
                print("Object is %s" % thinness)
                print("Estimated dims oriented %f x %f x %f m" % (d1,d2,depth))
                print("ML based ranking")
                print(read_current_rank)
                print("Knowledge validated ranking (area)")
                print(read_rank[:5])
                print("Knowledge validated ranking (area + flat + AR)")
                print(read_rank_flatAR[:5])
                if self.verbose:
                    print("Knowledge validated ranking (area + flat)")
                    print(read_rank_flat[:5])
                    print("Knowledge validated ranking (area + thin)")
                    print(read_rank_thin[:5])
                    print("Knowledge validated ranking (area + thin + AR)")
                    print(read_rank_thinAR[:5])
                print("================================")


        print("Took % fseconds." % float(time.time() - start)) #global proc time
        print("Re-evaluating post size correction...")
        if self.set == 'KMi':
            if self.verbose:
                """# Summary stats about predicted values"""
                size_summary = estimated_sizes.copy()
                for k in list(estimated_sizes.keys()):
                    sub_dict = estimated_sizes[k]
                    for subk,v in list(sub_dict.items()):
                        try:
                            size_summary[k]['mean-%s' %subk] = statistics.mean(v)
                        except: #not enough data points
                            size_summary[k]['mean-%s' % subk] = None
                        try:
                            size_summary[k]['std-%s' %subk] = statistics.stdev(v)
                        except: #not enough data points
                            size_summary[k]['std-%s' % subk] = None
                        size_summary[k]['min-%s' %subk] = min(v)
                        size_summary[k]['max-%s' %subk] = max(v)

                with open("./data/logged_stats.json", 'w') as fout:
                    json.dump(size_summary, fout)

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
            eval_singlemodel(self)
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

    def ML_predselection(self,current_ranking,current_label,gt_label,distance_t = 0.04,n=3):
        read_current_rank = [(self.remapper[current_ranking[z, 0]], current_ranking[z, 1]) for z in
                             range(current_ranking.shape[0])]
        MLclasses = [l[0] for l in read_current_rank]
        l_, c_ = Counter(MLclasses).most_common()[0]
        dis = read_current_rank[0][1]  # distance between test embedding and prod embedding
        if dis < distance_t and c_ >= n:  # lower distance/higher conf and class appears at least three times
            # ML is confident, keep as is
            print("%s predicted as %s" % (gt_label, current_label))
            print("ML based ranking")
            print(read_current_rank)
            print("ML confident, skipping size-based validation")
            print("================================")
            return
        else:
            return True
