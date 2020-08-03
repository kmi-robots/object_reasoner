"""
In Python 2 to use rosbag
"""

import rosbag
import datetime
import os

def extract_from_bag(rgb_img_list, path_to_bags,tol_min=0.03,tol_max=0.06):
    """
    Expects a list of paths to RGB data
    and the path to the folder containing the related bags with depth data
    Method to pick the nearest depth frame and pcl in time, based
    on set time tolerance (min, max).
    In our case param values where chosen based on a 30 fps sensor
    Based on how timestamps were formatted to create img files, see also DH_IO.py
    """
    available_bags = [(datetime.datetime.strptime(objname[:-6], "%Y-%m-%d-%H-%M-%S"),objname[:-6]) for objname in os.listdir(str(path_to_bags)) if objname[-4:]=='.bag']
    available_bags.sort(key=lambda x:x[0])

    for imgp in rgb_img_list:

        filename = str(imgp.split("/")[-1])
        basestamp = filename.split('_')[0]
        stampsecs = filename.split('_')[1]
        basestamp = basestamp[:10]+' '+basestamp[11:13]+ ':' + basestamp[14:16]+':'+ basestamp[17:]
        timestring = basestamp+'.'+stampsecs
        rgb_time = datetime.datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S.%f")

        for n, (bdate, bname) in enumerate(available_bags):
            try:
                if rgb_time >= bdate and rgb_time < available_bags[n+1][0]:
                    tgt_bag = bname
                    break
            except IndexError: # reached end of bag list
                if rgb_time >= bdate:
                    tgt_bag = bname
                    break


        continue

    return None, None

