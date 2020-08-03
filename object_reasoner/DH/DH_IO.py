"""
In Python2 because it interfaces with ROS

Methods to interact with Data Hub API:
Requires key to access DH stored in a local .py file
"""
import argparse
import requests
from requests.auth import HTTPBasicAuth
import json
import datetime
import os
import time
import sys
import rosbag
from cv_bridge import CvBridge #, CvBridgeError
import cv2
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt

try:
    import DHauth as dh
except:
    print("Please create a local file named DHauth.py containing your Data Hub credentials")
    sys.exit(0)

if not os.path.isfile(dh.path_to_log):
    # create log file
    f = open(dh.path_to_log, 'w+')
    f.close()

def get_imgs():
    # print(type(DHdict['url']))
    t = datetime.datetime.fromtimestamp(time.time())
    timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    json_body= {
        "@id": "string",
        "@type": "ImageReport",
        "team": dh.teamid,
        "timestamp": timestamp,
        }

    return requests.request("GET", dh.url, data=json.dumps(json_body),
                            auth=HTTPBasicAuth(dh.teamkey, ''))

def post_img(input, tstamp, fbase, rgb=True):

    if rgb:
        _, buffer = cv2.imencode('.jpg', input)

    else: #depth img
        buffer = open(input, 'rb').read()
        fbase ="depth_" +fbase

    bstring = base64.b64encode(buffer).decode('utf-8')  # .encode('utf-8')

    """
    dstring = base64.b64decode(bstring)  # .decode('utf-8'))
    dimg = Image.open(BytesIO(dstring))
    darray = np.array(dimg)
    plt.imshow(darray, cmap='Greys_r')
    plt.show()
    # print(darray.dtype)
    """
    complete_url = os.path.join(dh.url, fbase)
    json_body ={"@id": fbase,
                 "@type": dh.nodetype,
                 "team": dh.teamid,
                 "timestamp": tstamp,
                 "x": 0,
                 "y": 0,
                 "z": 0,
                 "base64": bstring,
                 "format": "image/jpeg"
                }
    return requests.request("POST", complete_url, data=json.dumps(json_body),
                            auth=HTTPBasicAuth(dh.teamkey, ''))

def main():
    """
    Currently plays a local rosbag to export either RGB or depth data
    as JPG image
    + Sends the image to DH after converting to base64
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path_in', help="Path to input bag")
    parser.add_argument('pat'
                        'h_out', help="Where to store images locally")
    args = parser.parse_args()

    print("Loading rosbag from %s" % args.path_in)
    bag = rosbag.Bag(args.path_in)
    bridge = CvBridge()

    #create 1 dir for rgb and 1 for depth imgs
    if not os.path.isdir(os.path.join(args.path_out,'rgb')):
        os.makedirs(os.path.join(args.path_out,'rgb'))
    if not os.path.isdir(os.path.join(args.path_out,'depth')):
        os.makedirs(os.path.join(args.path_out,'depth'))

    print("Iterating through messages")
    for topic, msg, t in bag.read_messages():
        try:
            # tstamp in header more realistic than t, i.e., time img is stored in the bag
            timestamp = msg.header.stamp.to_sec()
            t = datetime.datetime.fromtimestamp(timestamp)
            timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            fname = str(t).replace('.','_').replace(' ','-').replace(':','-')

        except AttributeError: continue

        if topic == '/camera/rgb/image_raw':
            rgb_img = bridge.imgmsg_to_cv2(msg, "passthrough")
            filename = fname + ".jpg"
            if not os.path.isfile(os.path.join(args.path_out,'rgb', filename)):
                cv2.imwrite(os.path.join(args.path_out,'rgb', filename), rgb_img)
                #send img to DH
                res = post_img(rgb_img, timestamp, fname)
                print(res.content)
                if not res.ok:
                    #Log if KO
                    with open(dh.path_to_log, 'a') as logf:
                        logf.write("Problem sending image %s to DH \n" % fname)
                        logf.write("Timestamp %s \n" % timestamp)
                        logf.write("Error returned: \n")
                        logf.write(str(res.content))

        elif topic == '/camera/depth/image_raw':
            depth_data = bridge.imgmsg_to_cv2(msg, "32FC1")
            # print(depth_data.dtype)
            filename = fname + "-depth.tiff"
            dimg = Image.fromarray(depth_data)
            if not os.path.isfile(os.path.join(args.path_out, 'depth', filename)):
                dimg.save(os.path.join(args.path_out, 'depth', filename))
                #send to DH
                res = post_img(os.path.join(args.path_out, 'depth', filename),timestamp, fname, rgb=False)
                print(res.content)
                if not res.ok:
                    # Log if KO
                    with open(dh.path_to_log, 'a') as logf:
                        logf.write("Problem sending image %s to DH \n" % fname)
                        logf.write("Timestamp %s \n" % timestamp)
                        logf.write("Error returned: \n")
                        logf.write(str(res.content))

        else: continue

    print("RGB and depth images saved under %s" % args.path_out)
    return 0

if __name__ == "__main__":

    sys.exit(main())
