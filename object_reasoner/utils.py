"""
Methods used to convert/handle raw input data
"""
import os
import json
import h5py
import numpy as np
import csv
import cv2
from PIL import Image
# import matplotlib.pyplot as plt

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

def load_emb_space(args):
    """
    Assumes the input are the HDF5 files
    as produced by the baselines provided at
    https://github.com/andyzeng/arc-robot-vision/image-matching
    """

    if args.baseline =="imprk-net":

        path_to_hdf5 = os.path.join('./data/imprintedKnet/snapshots-with-class', 'snapshot-test-results.h5')
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

def load_camera_intrinsics(path_to_intr):
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
                    intrinsics.append(float(cell.split(" ")[1]))
    return intrinsics



def BGRtoRGB(img_array):
    img = img_array.copy()
    img[:, :, 0] = img_array[:, :, 2]
    img[:, :, 2] = img_array[:, :, 0]

    return img

def img_preproc(path_to_image, transform, cropping_flag=False, array_form=False):

    if not array_form:
        path_to_image = cv2.imread(path_to_image)
    img = BGRtoRGB(path_to_image)
    try:
        x = Image.fromarray(img, mode='RGB')
        if cropping_flag:
            x = x.crop((120,30,520,430)) #as hardcoded in Zeng et al for ARC 2017

    except:
        return None
    return transform(x)

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

    return

