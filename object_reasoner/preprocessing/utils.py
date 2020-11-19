"""
Methods used to convert/handle raw input data
"""
import os
import json
import h5py
import numpy as np
import csv
import execnet


def init_obj_catalogue(path_to_data):

    obj_dict = {}
    subfols = sorted(os.listdir(os.path.join(path_to_data, 'test-item-data')))
    known_classes = sorted(os.listdir(os.path.join(path_to_data, 'train-item-data')))
    for i,class_fol in enumerate(subfols):
        cname = class_fol.lower()
        try:
            with open(os.path.join(path_to_data, 'test-item-data',class_fol, cname+'.json')) as fin:
                obj_node = json.load(fin) #start from json data given
                obj_node['known'] = True if class_fol in known_classes else False #Known v Novel?

        except FileNotFoundError:
            print("No json file found for object %s" % cname)
            print("Adding empty node")
            obj_node = {"dimensions": [0,0,0] }
            obj_node['known'] = True #'Empty' is known at training time

        obj_node['label'] = str(i + 1) # add class label in same format as gt (starts from 1)
        obj_dict[cname] = obj_node

    return obj_dict

def load_emb_space(args,fname='snapshot-test-results.h5'):
    """
    Assumes the input are the HDF5 files
    as produced by the baselines provided at
    https://github.com/andyzeng/arc-robot-vision/image-matching
    """
    if args.baseline =="imprk-net" or args.set=='KMi':

        path_to_hdf5 = os.path.join(args.test_res, args.baseline, 'snapshots-with-class',fname)
        tgt_impr = h5py.File(path_to_hdf5, 'r')
        return np.array(tgt_impr['prodFeat'], dtype='<f4'), np.array(tgt_impr['testFeat'], dtype='<f4'),\
               None, None

    else:
        path_to_hdf5 = args.test_res
        tgt_novel = os.path.join(path_to_hdf5, 'snapshots-no-class', 'results-snapshot-8000.h5') #default folder structure by Zeng et al.
        tgt_known = os.path.join(path_to_hdf5, 'snapshots-with-class', 'results-snapshot-170000.h5')

        nnetf = h5py.File(tgt_novel, 'r')
        knetf = h5py.File(tgt_known, 'r')

        return np.array(knetf['prodFeat'], dtype='<f4'), np.array(knetf['testFeat'], dtype='<f4'), \
               np.array(nnetf['prodFeat'], dtype='<f4'), np.array(nnetf['testFeat'], dtype='<f4')

def load_camera_intrinsics_txt(path_to_intr):
    """
    Expects 3x3 intrinsics matrix as tab-separated txt
    """
    intrinsics=[]
    with open(path_to_intr) as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            if row==[]: continue
            for cell in row:
                if cell=='': continue
                try:
                    intrinsics.append(float(cell.split("  ")[1]))
                except IndexError:
                    try:
                        intrinsics.append(float(cell.split(" ")[1]))
                    except IndexError:
                        intrinsics.append(float(cell))
    return intrinsics


def exclude_nodepth(Reasonerobj, basemethod):
    # eval only on those with a depth region associated
    blacklist = ['387317_6', '559158_0', '559158_3', '559158_poly3', '437317_poly10', '809309_poly2',
                 '809309_poly4', '859055_1', '859055_2', '859055_3', '859055_poly0', '246928_6', '655068_2',
                 '655068_5', '655068_6', '477148_poly7', '258729_2', '258729_3', '258729_poly4']

    keep_indices = [i for i,dimg in enumerate(Reasonerobj.dimglist) \
                        if dimg is not None
                        and '_'.join(Reasonerobj.imglist[i].split('/')[-1].split('.')[0].split('_')[-2:]) not in blacklist]
    Reasonerobj.labels = [l for i,l in enumerate(Reasonerobj.labels) if i in keep_indices]
    if Reasonerobj.set =='arc': Reasonerobj.tsamples = [l for i, l in enumerate(Reasonerobj.labels) if i in keep_indices]
    else: Reasonerobj.tsamples = None
    Reasonerobj.predictions = Reasonerobj.predictions[keep_indices]
    if basemethod=='two-stage': Reasonerobj.predictions_B[keep_indices]
    else: Reasonerobj.predictions_B = None
    Reasonerobj.dimglist = [imge for i, imge in enumerate(Reasonerobj.dimglist) if i in keep_indices]
    Reasonerobj.imglist = [imge for i, imge in enumerate(Reasonerobj.imglist) if i in keep_indices]
    return Reasonerobj

##################################################################
#
#  Creating txt ground truth files in the same format as ARC2017
#
##################################################################

def arcify(root_img_path):

    base = root_img_path.split('/')[-1]
    for root, dirs, files in os.walk(root_img_path):
            for name in dirs:
                imgpaths=[]
                imglabels=[]
                first =True
                for sroot, sdirs, sfiles in os.walk(os.path.join(root_img_path,name)):
                    if first:
                        obj_classes=sdirs
                        first = False
                        continue
                    if sfiles:
                        classname = sroot.split('/')[-1]
                        label = obj_classes.index(classname) + 1
                        imgpaths.extend([os.path.join(base,name,classname,f) for f in sfiles])
                        imglabels.extend([str(label) for f in sfiles]) # as many labels an no of files in that subfolder

                lname = name.split("-imgs")[0]
                with open(os.path.join(root_img_path,name+'.txt'), mode='w') as outf, \
                    open(os.path.join(root_img_path,lname+'-labels.txt'), mode='w') as outl:
                    outf.write('\n'.join(imgpaths))
                    outl.write('\n'.join(imglabels))

            break # skip outer for, after first folder level

def create_class_map(path_to_json):
    """
    Assuming arcify was already run locally
    """
    base= path_to_json.split("class_to_index.json")[0]
    path_train = os.path.join(base,"train-product-imgs")
    class_names = os.listdir(path_train)
    class_index={}
    try:
        with open(os.path.join(base,"train-product-imgs.txt")) as fpath, \
            open(os.path.join(base, "train-product-labels.txt")) as flabels:
            fileps= fpath.read().splitlines()
            labels = flabels.read().splitlines()
    except FileNotFoundError:
        print("Run arcify method locally before to generate reference txts")
        return 0

    for pth, label in zip(fileps, labels):
        category = pth.split("train-product-imgs/")[1].split("/")[0]
        if category not in class_index.keys():
            class_index[category] = label
    with open(path_to_json, 'w') as fout:
        json.dump(class_index, fout)
    print("Class - numeric index mapping saved locally")
    return None

def call_python_version(Version, Module, Function, ArgumentList):
    gw = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()

def list_depth_filenames(input_path):
    """
    Returns path to files containing "depth" ih their name
    Expects a macro folder with sub-folder structure divided class by class
    e.g., passing ./data will search over ./data/class1, ./data/class2 ... etc.
    """
    fnamelist = []
    for root, dirs, files in os.walk(input_path):
        for name in dirs:
            base = os.path.join(root, name)
            fnamelist.extend([os.path.join(base,f) for f in os.listdir(base) if 'depth' in f])

    if len(fnamelist)> 0 : return fnamelist
    else: return None
