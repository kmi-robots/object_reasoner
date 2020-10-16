"""
Methods to process data recorded in rosbag format
In Python 2 to use rosbag for ROS melodic
"""

import rosbag
import datetime
import os
from cv_bridge import CvBridge
import numpy as np
from rgb_img_processing import crop_test
import png
import shutil

def extract_from_bag(rgb_img_list,path_to_bags,path_to_annotations, tol_min=0.06,tol_max=0.2):
    """
    Expects a list of paths to RGB data
    and the path to the folder containing the related bags with depth data
    Method to pick the nearest depth frame and pcl in time, based
    on set time tolerance (min, max).
    In our case param values where chosen based on a 30 fps sensor
    Based on how timestamps were formatted to create img files, see also DH_IO.py
    """
    available_bags = [(datetime.datetime.strptime(objname[:-6], "%Y-%m-%d-%H-%M-%S"),objname[:-6], objname[-6:]) for objname in os.listdir(str(path_to_bags)) if objname[-4:]=='.bag']
    available_bags.sort(key=lambda x:x[0]) # order chronologically
    dimg_list =[]
    # pcls =[]
    img_index = {}
    for imgp in rgb_img_list:
        filename = str(imgp.split("/")[-1])
        out_path_rgb = imgp.split("TBA")[0] + "validated/" + filename
        if os.path.exists(out_path_rgb):
            continue  # skip
        basestamp = filename.split('_')[0]
        stampsecs = filename.split('_')[1]
        if 'jpg' or 'png' in stampsecs:
            stampsecs = stampsecs[:-4]
        basestamp = basestamp[:10]+' '+basestamp[11:13]+ ':' + basestamp[14:16]+':'+ basestamp[17:]
        timestring = basestamp+'.'+stampsecs
        rgb_time = datetime.datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S.%f")
        try:
            img_index[rgb_time]["urls"].append(imgp)
        except: # new key
            img_index[rgb_time]={}
            img_index[rgb_time]["urls"]= []
            img_index[rgb_time]["urls"].append(imgp)

    for timekey, path_dict in img_index.items():
        # array of images for that timekey
        imgps = path_dict["urls"]
        origin_path = imgps[0]
        filenam = origin_path.split("TBA")[1]
        out_path_rgb = origin_path.split("TBA")[0] + "validated/" + filenam
        out_path_depth = origin_path.split("TBA")[0] + "validated/" + filenam[:-4] + 'depth.png'
        # copy RGB file to validated files folder

        for n, (bdate, bname, ext) in enumerate(available_bags): # find bag file the img belongs to (time-wise)
            try:
                if timekey >= bdate and timekey < available_bags[n+1][0]:
                    tgt_bag = bname+ext
                    break
            except IndexError: # reached end of bag list
                if timekey >= bdate:
                    tgt_bag = bname+ext
                    break
        bag = rosbag.Bag(os.path.join(str(path_to_bags), tgt_bag))
        # iterate over bag within a certain time window before and after the rgb timestamp
        #t0_min,  t0_max = timekey - datetime.timedelta(seconds=tol_min), timekey + datetime.timedelta(seconds=tol_min)
        t1_min, t1_max = timekey - datetime.timedelta(seconds=tol_max), timekey + datetime.timedelta(seconds=tol_max)

        #d_img, pcloud = find_nearest_frame(bag, timekey, t0_min, t0_max, search_list = ['/camera/depth/image_raw','/camera/depth/points'])
        # If none found, try again with less strict tolerance
        #if d_img is None:
        d_img, pcloud = find_nearest_frame(bag, timekey, t1_min, t1_max, search_list=['/camera/depth/image_raw'])
        #if pcloud is None:
        #    d_img, pcloud = find_nearest_frame(bag, timekey, t1_min, t1_max, search_list=['/camera/depth/points'])
        if d_img is None:
            dimg_list.append(d_img)
            continue

        #Save copy of depth image locally, one copy for each crop at that timestamp
        d_img.astype(np.uint16)
        if path_to_annotations is None:
            # Just return/save full depth image
            dimg_list.append(d_img)
            # save as 16-bit one-channeled PNG

            shutil.copyfile(origin_path,out_path_rgb)
            with open(out_path_depth, 'wb') as f:  # 16-bit PNG img, with values in millimeters
                writer = png.Writer(width=d_img.shape[1], height=d_img.shape[0], bitdepth=16)
                # Convert array to the Python list of lists expected by the png writer.
                gray2list = d_img.tolist()
                writer.write(f, gray2list)

        else: #crop depth image before adding to list
            # Crop to 2D bbox or polygon
            dimg_list = crop_test(imgps, path_to_annotations, None, d_img, dimg_list)
    #return dimg_list

def find_nearest_frame(bagfile, rgb_time, lower_bound, upper_bound, search_list=[]):
    min_delta = float("inf")
    bridge = CvBridge()
    depth_img = None
    pcl = None
    for topic, msg, t in bagfile.read_messages():
        if topic not in search_list: continue
        timestamp = msg.header.stamp.to_sec()
        t = datetime.datetime.fromtimestamp(timestamp)
        delta = abs(t - rgb_time).total_seconds()
        if t >= lower_bound and t <= upper_bound and delta< min_delta:
            min_delta = delta  # find nearest frame in time
            if topic == '/camera/depth/image_raw':
                depth_img = bridge.imgmsg_to_cv2(msg, "32FC1")
            elif topic == '/camera/depth/points':
                pcl = msg

    return depth_img, pcl

def load_intrinsics(path_to_bag, path_to_out):
    """
    Derives intrinsics from camera_info messages
    of a provided bag
    """
    bag = rosbag.Bag(path_to_bag)
    for topic, msg, t in bag.read_messages():
        if topic == '/camera/depth/camera_info':
            intrinsics = msg.K
            break
    K_matrix =[]

    with open(path_to_out, 'w') as outf:
        for i,cellv in enumerate(intrinsics,1): # 3x3 matrix #start count from 1
            K_matrix.append(cellv)
            if i%3 ==0:
                outf.write(str(cellv)+"\n")
            else:
                outf.write(str(cellv)+"\t")
    return K_matrix
