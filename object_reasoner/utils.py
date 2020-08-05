"""
Methods used to convert/handle raw input data
"""
import os
import json
import sys
import h5py
import numpy as np
import csv
import cv2
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import execnet
import png
import matplotlib.pyplot as plt

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

    if args.baseline =="imprk-net" or args.set=='KMi':

        path_to_hdf5 = os.path.join(args.test_res, 'snapshot-test-results.h5')
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


##################################################################
#
#  Cropping RGB test images based on annotated regions
#
##################################################################

def crop_test(path_to_imgs, path_to_annotations, path_to_out, depth_img=None, dimg_list=None):
    """
    pass depth image to crop depth images, leave set to None to crop RGB images
    """
    if depth_img is None:
        with open(os.path.join(path_to_annotations, 'test-imgs-labels1.json')) as jf, \
            open(os.path.join(path_to_out, '../class_to_index.json')) as jmap:
            jtree = json.load(jf)  #JSON file with polygonal annotations
            cmap = json.load(jmap) #JSON file with mapping from class name to number
    else:
        with open(os.path.join(path_to_annotations, 'test-imgs-labels1.json')) as jf:
            jtree = json.load(jf)
    test_imgs =[]
    test_labels =[]
    for pimg in path_to_imgs:
        fname = str(pimg.split("/")[-1])
        if depth_img is None:
            img = BGRtoRGB(cv2.imread(pimg))
            lname = fname[:-3] + 'xml'
            cropname = fname
        else:
            cropname = fname
            bname = fname[:26]
            fname = bname +'.jpg'
            lname = bname + '.xml' #already points to RGB crop and not to original fname
        try:
            tree = ET.parse(os.path.join(path_to_annotations, 'test-imgs-labels', lname))
            try:
                polyf = jtree[fname]
            except KeyError:
                polyf = None
        except:
            print("No rectangular annotations found for image %s" % fname)
            tree = None
            try:
                polyf = jtree[fname]
            except KeyError:
                print("No annotated polygons either..Skipping...")
                continue

        if tree is not None:
            """Handling rectangular bbox first"""
            root = tree.getroot()
            if depth_img is None:
                for n, object in enumerate(root.findall('object')):
                    bbox = object.find('bndbox')
                    label = object.find('name').text.replace('/', '_').replace(' ', '')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    roi = img[ymin:ymax, xmin:xmax]

                    if not os.path.isdir(os.path.join(path_to_out, label)):
                        os.mkdir(os.path.join(path_to_out, label))  # create class/category folder
                    # save roi as separate img file
                    pout = os.path.join(path_to_out, label, fname[:-4] + '_' + str(n) + '.png')
                    pil_roi = Image.fromarray(roi)
                    pil_roi.save(pout)  # cv2.imwrite(pout, roi) #cv2 gives BGR issues
                    test_imgs.append(pout)
                    try:
                        test_labels.append(cmap[label])
                    except KeyError:
                        if label == "fire_alarm_call_assembly_point_sign":
                            test_labels.append(cmap["fire_alarm_assembly_sign"])
                        else:
                            sys.exit(0)
                    # plt.imshow(roi)
                    # plt.title(label=label)
                    # plt.show()
            elif depth_img is not None and 'poly' not in cropname.split('_')[-1]:
                #select just specific roi of that crop
                img = depth_img.copy()  # copy for cropping
                n = int(cropname.split('_')[-1][:-4])
                object = root.findall('object')[n]
                bbox = object.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                roi = img[ymin:ymax, xmin:xmax]
                #dimg = Image.fromarray(roi)
                #dimg.show()
                # save as 16-bit one-channeled PNG
                with open(pimg[:-4] +'depth.png', 'wb') as f:  # 16-bit PNG img, with values in millimeters
                    writer = png.Writer(width=roi.shape[1], height=roi.shape[0], bitdepth=16)
                    # Convert array to the Python list of lists expected by the png writer.
                    gray2list = roi.tolist()
                    writer.write(f, gray2list)
                dimg_list.append(roi)
            else:
                pass #do nothing

        """Handling polygonal regions, if any"""
        if polyf is not None:
            rois = polyf["regions"]
            if depth_img is None:
                for i, (k, v) in enumerate(rois.items()):
                    label = v["region_attributes"]["label"].replace('/', '_').replace(' ', '')
                    all_x = v["shape_attributes"]["all_points_x"]
                    all_y = v["shape_attributes"]["all_points_y"]
                    cropped_img = crop_polygonal(pimg, list(zip(all_x, all_y)))
                    if not os.path.isdir(os.path.join(path_to_out, label)):
                        os.mkdir(os.path.join(path_to_out, label))
                    pout = os.path.join(path_to_out, label, fname[:-4] + '_poly' + str(i) + '.png')
                    # Save, but 3-channeled
                    cropped_img = Image.fromarray(cropped_img, "RGBA")
                    pil_roi = Image.new("RGB", cropped_img.size, (0, 0, 0))  # create black background
                    pil_roi.paste(cropped_img, mask=cropped_img.split()[3])  # paste content of alpha channel in it
                    pil_roi.save(pout)
                    # pil_roi.show()
                    test_imgs.append(pout)
                    try:
                        test_labels.append(cmap[label])
                    except KeyError:
                        if label == "fire_alarm_call_assembly_point_sign":
                            test_labels.append(cmap["fire_alarm_assembly_sign"])
                        else:
                            sys.exit(0)
            elif depth_img is not None and 'poly' in cropname.split('_')[-1]:
                # select just specific roi of that crop
                img = depth_img.copy()  # copy for cropping
                t = cropname.split('_')[-1][:-4]
                i = int(t.split('poly')[1])
                _,v = rois.items()[i]
                all_x = v["shape_attributes"]["all_points_x"]
                all_y = v["shape_attributes"]["all_points_y"]
                cropped_img = crop_polygonal(img, list(zip(all_x, all_y)), rgb=False)
                with open(pimg[:-4] + 'depth.png', 'wb') as f:  # 16-bit PNG img, with values in millimeters
                    writer = png.Writer(width=cropped_img.shape[1], height=cropped_img.shape[0], bitdepth=16)
                    # Convert array to the Python list of lists expected by the png writer.
                    gray2list = cropped_img.tolist()
                    writer.write(f, gray2list)
                dimg_list.append(cropped_img)
            else: pass #do nothing

    #Create ARC-formatted txts
    if depth_img is None:
        with open(os.path.join(path_to_out, '../test-imgs.txt'), mode='w') as outf, \
            open(os.path.join(path_to_out, '../test-labels.txt'), mode='w') as outl:
            outf.write('\n'.join(test_imgs))
            outl.write('\n'.join(test_labels))
        print("Test set ground truth annotations parsed and saved locally")
    else:
        return dimg_list

def crop_polygonal(path_image, polygon, rgb=True):
    """
    Expects path to input image file
    and list of tuples, i.e., x,y points of polygon
    returns a 4-channeled masked image if rgb=True
    masks a depth image if rgb=False
    """
    if rgb:
        # read image as RGB and add alpha (transparency)
        im = Image.open(path_image).convert("RGBA")
        imArray = np.asarray(im)
    else:
        imArray = path_image #depth img matrix passed directly

    # create mask
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)

    if rgb:
        # assemble new image (uint8: 0-255)
        newImArray = np.empty(imArray.shape, dtype='uint8')
        # colors (three first columns, RGB)
        newImArray[:, :, :3] = imArray[:, :, :3]
        # transparency (4th column)
        newImArray[:, :, 3] = mask * 255
    else: #from depth, one-channeled
        #mask binary image
        newImArray = imArray.copy()
        newImArray[mask!=1]=0.

    return newImArray



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
