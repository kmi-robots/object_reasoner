"""
In Python 2 to use rosbag
"""

import rosbag
import datetime
import os
from cv_bridge import CvBridge
import numpy as np
import png
# from PIL import Image

def extract_from_bag(rgb_img_list, path_to_bags,tol_min=0.03,tol_max=0.06):
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

    for imgp in rgb_img_list:
        filename = str(imgp.split("/")[-1])
        basestamp = filename.split('_')[0]
        stampsecs = filename.split('_')[1]
        basestamp = basestamp[:10]+' '+basestamp[11:13]+ ':' + basestamp[14:16]+':'+ basestamp[17:]
        timestring = basestamp+'.'+stampsecs
        rgb_time = datetime.datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S.%f")

        for n, (bdate, bname, ext) in enumerate(available_bags): # find bag file the img belongs to (time-wise)
            try:
                if rgb_time >= bdate and rgb_time < available_bags[n+1][0]:
                    tgt_bag = bname+ext
                    break
            except IndexError: # reached end of bag list
                if rgb_time >= bdate:
                    tgt_bag = bname+ext
                    break

        bag = rosbag.Bag(os.path.join(str(path_to_bags), tgt_bag))

        # iterate over bag within a certain time window before and after the rgb timestamp
        t0_min,  t0_max = rgb_time - datetime.timedelta(seconds=tol_min), rgb_time + datetime.timedelta(seconds=tol_min)
        d_img, pcloud = find_nearest_frame(bag, rgb_time, t0_min, t0_max, search_list = ['/camera/depth/image_raw','/camera/depth/points'])
        # If none found, try again with less strict tolerance
        t1_min, t1_max = rgb_time - datetime.timedelta(seconds=tol_max), rgb_time + datetime.timedelta(seconds=tol_max)
        if d_img is None:
            d_img, pcloud = find_nearest_frame(bag, rgb_time, t1_min, t1_max, search_list=['/camera/depth/image_raw'])
        #if pcloud is None:
        #    d_img, pcloud = find_nearest_frame(bag, rgb_time, t1_min, t1_max, search_list=['/camera/depth/points'])
        if d_img is None:
            print("No depth frame found for img %s" % imgp)
            dimg_list.append(d_img)
            continue

        #Save copy of depth image locally
        d_img.astype(np.uint16)
        # dimg = Image.fromarray(d_img)
        with open(imgp[:-4]+'depth.png', 'wb') as f: #16-bit PNG img, with values in millimeters
            writer = png.Writer(width=d_img.shape[1], height=d_img.shape[0], bitdepth=16)
            # Convert array to the Python list of lists expected by the png writer.
            gray2list = d_img.tolist()
            writer.write(f, gray2list)
        dimg_list.append(d_img)
        # pcls.append(pcloud)

    return dimg_list

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
