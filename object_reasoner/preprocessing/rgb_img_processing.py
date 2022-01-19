"""
Methods for RGB image processing
"""
import cv2
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import png
import collections
import sys
import json
import os
import numpy as np

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
#  Cropping RGB test images based on annotated regions
#
##################################################################

def crop_test(path_to_imgs, path_to_annotations, path_to_out, depth_img=None, dimg_list=None,safpx =0,safpx_rect=0):
    """
    pass depth image to crop depth images, leave set to None to crop RGB images
    safpx param reduces crop by x%edge per edge only in depth case, to increase chances of correct crop
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
                print("No annotated polygons either..Skipping img %s..." % fname)
                continue

        if tree is not None:
            """Handling rectangular bbox first"""
            root = tree.getroot()
            if depth_img is None or (depth_img is not None and 'depth' in pimg):
                for n, object in enumerate(root.findall('object')):
                    bbox = object.find('bndbox')
                    label = object.find('name').text.replace('/', '_').replace(' ', '')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    if not os.path.isdir(os.path.join(path_to_out, label)):
                        os.mkdir(os.path.join(path_to_out, label))  # create class/category folder
                    # save roi as separate img file
                    if depth_img is None: #rgb case
                        pout = os.path.join(path_to_out, label, fname[:-4] + '_' + str(n) + '.png')
                        roi = img[ymin:ymax, xmin:xmax]
                        pil_roi = Image.fromarray(roi)
                        pil_roi.save(pout)  # cv2.imwrite(pout, roi) #cv2 gives BGR issues
                        test_imgs.append(pout)
                        try:
                            test_labels.append(cmap[label])
                        except KeyError:
                            if label == "fire_alarm_call_assembly_point_sign":
                                test_labels.append(cmap["fire_alarm_assembly_sign"])
                            elif label == "pile_of_papers":
                                test_labels.append(cmap["pile_of_paper"])
                            else:
                                print("----Problem with object label formatting....exiting----")
                                sys.exit(0)
                    else: #depth case #depth img passed as list
                        # passed depth images as list
                        pout = os.path.join(path_to_out, label, fname[:-4] + 'depth_' + str(n) + '.png')
                        img = cv2.imread(pimg, cv2.IMREAD_UNCHANGED)
                        #plt.imshow(img)
                        #plt.title(label=label)
                        #plt.show()
                        roi = img[ymin:ymax, xmin:xmax]
                        #plt.imshow(roi)
                        #plt.title(label=label)
                        #plt.show()
                        if safpx_rect > 0:
                            h, w = roi.shape
                            safx, safy = int(safpx_rect* w), int(safpx_rect*h)
                            safxmax, safymax = int(w - safx), int(h - safy)
                            roi = roi[safy:safymax, safx:safxmax]

                        #plt.imshow(roi)
                        #plt.title(label=label)
                        #plt.show()
                        with open(pout, 'wb') as f:  # 16-bit PNG img, with values in millimeters
                            writer = png.Writer(width=roi.shape[1], height=roi.shape[0], bitdepth=16)
                            # Convert array to the Python list of lists expected by the png writer.
                            gray2list = roi.tolist()
                            writer.write(f, gray2list)
                    # plt.imshow(roi)
                    # plt.title(label=label)
                    # plt.show()
            elif depth_img is not None and 'poly' not in cropname.split('_')[-1]: #depth case #one crop by one instead of full list
                #select just specific roi of that crop
                img = depth_img.copy()  # copy for cropping
                n = int(cropname.split('_')[-1][:-4])
                label = object.find('name').text.replace('/', '_').replace(' ', '')
                object = root.findall('object')[n]
                bbox = object.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                roi = img[ymin:ymax, xmin:xmax]
                #dimg = Image.fromarray(roi)
                #dimg.show()
                if safpx_rect> 0:
                    h, w = roi.shape
                    safx, safy = int(safpx_rect * w), int(safpx_rect * h)
                    safxmax, safymax = int(w - safx), int(h - safy)
                    roi = roi[safy:safymax, safx:safxmax]
                #dimg = Image.fromarray(roi)
                #dimg.show()
                # save as 16-bit one-channeled PNG
                po = os.path.join(path_to_out,label,cropname)
                with open(po[:-4] +'_'+str(n)+'.png', 'wb') as f:  # 16-bit PNG img, with values in millimeters
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
            rois = collections.OrderedDict(sorted(rois.items()))
            if depth_img is None or (depth_img is not None and 'depth' in pimg):
                for i, (k, v) in enumerate(rois.items()):
                    label = v["region_attributes"]["label"].replace('/', '_').replace(' ', '')
                    all_x = v["shape_attributes"]["all_points_x"]
                    all_y = v["shape_attributes"]["all_points_y"]
                    if not os.path.isdir(os.path.join(path_to_out, label)):
                        os.mkdir(os.path.join(path_to_out, label))
                    pout = os.path.join(path_to_out, label, fname[:-4] + '_poly' + str(i) + '.png')
                    if depth_img is None:
                        polygon_coords = list(zip(all_x, all_y))
                        cropped_img, bounding_rect = crop_polygonal(pimg, polygon_coords)
                        # Save, but 3-channeled
                        cropped_img = Image.fromarray(cropped_img, "RGBA")
                        # create white background region
                        pil_roi = Image.new("RGB", cropped_img.size, (255, 255, 255))
                        pil_roi.paste(cropped_img, mask=cropped_img.split()[3])  # paste content of alpha channel in it
                        # crop to rectangle bounding the polygonal mask
                        #pil_roi.show()
                        region =pil_roi.crop(bounding_rect)
                        #region.show()
                        region.save(pout)
                        test_imgs.append(pout)

                        try:
                            test_labels.append(cmap[label])
                        except KeyError:
                            if label == "fire_alarm_call_assembly_point_sign":
                                test_labels.append(cmap["fire_alarm_assembly_sign"])
                            elif label == "pile_of_papers":
                                test_labels.append(cmap["pile_of_paper"])
                            else:
                                print("----Problem with object label formatting....exiting----")
                                sys.exit(0)
                    else: #save depth img
                        img = cv2.imread(pimg, cv2.IMREAD_UNCHANGED)
                        #plt.imshow(img, cmap='Greys_r')
                        #plt.show()
                        cropped_img, bounding_rect = crop_polygonal(img, list(zip(all_x, all_y)), rgb=False)
                        #plt.imshow(cropped_img, cmap='Greys_r')
                        #plt.show()
                        if bounding_rect:
                            y,ymax,x,xmax = bounding_rect
                            #plt.imshow(cropped_img, cmap='Greys_r')
                            #plt.show()
                            pre_crop = cropped_img.copy()
                            cropped_img = cropped_img[y:ymax,x:xmax]
                            pre_scale = cropped_img.copy()
                            #plt.imshow(cropped_img, cmap='Greys_r')
                            #plt.show()
                            if safpx > 0:
                                h,w = cropped_img.shape
                                safx, safy = int(safpx*w), int(safpx*h)
                                safxmax, safymax = int(w-safx),int(h-safy)
                                cropped_img = cropped_img[safy:safymax, safx:safxmax]
                            #plt.imshow(cropped_img, cmap='Greys_r')
                            #plt.show()
                        po = os.path.join(path_to_out,label, cropname)
                        #continue
                        with open(po[:-4] + '_poly'+str(i)+'.png', 'wb') as f:  # 16-bit PNG img, with values in millimeters
                            try:
                                writer = png.Writer(width=cropped_img.shape[1], height=cropped_img.shape[0], bitdepth=16)
                                # Convert array to the Python list of lists expected by the png writer.
                                gray2list = cropped_img.tolist()
                            except png.ProtocolError: #too few points are non zero
                                #reverting back to image pre-crop
                                writer = png.Writer(width=pre_crop.shape[1], height=pre_crop.shape[0],bitdepth=16)
                                # Convert array to the Python list of lists expected by the png writer.
                                gray2list = pre_crop.tolist()
                            writer.write(f, gray2list)

            elif depth_img is not None and 'poly' in cropname.split('_')[-1]:
                # Imgs cropped individually, select just specific roi of that crop
                img = depth_img.copy()  # copy for cropping
                t = cropname.split('_')[-1][:-4]
                i = int(t.split('poly')[1])
                _,v = rois.items()[i]
                all_x = v["shape_attributes"]["all_points_x"]
                all_y = v["shape_attributes"]["all_points_y"]
                cropped_img, bounding_rect = crop_polygonal(img, list(zip(all_x, all_y)), rgb=False)
                if bounding_rect:
                    y,ymax,x,xmax = bounding_rect
                    pre_crop = cropped_img.copy()
                    cropped_img = cropped_img[y:ymax, x:xmax]
                    if safpx > 0:
                        h, w = cropped_img.shape
                        safx, safy = int(safpx * w), int(safpx * h)
                        safxmax, safymax = int(w - safx), int(h - safy)
                        cropped_img = cropped_img[safy:safymax, safx:safxmax]
                with open(pimg[:-4] + 'depth.png', 'wb') as f:  # 16-bit PNG img, with values in millimeters
                    try:
                        writer = png.Writer(width=cropped_img.shape[1], height=cropped_img.shape[0], bitdepth=16)
                        # Convert array to the Python list of lists expected by the png writer.
                        gray2list = cropped_img.tolist()
                    except png.ProtocolError:  # too few points are non zero
                        # reverting back to image pre-crop
                        writer = png.Writer(width=pre_crop.shape[1], height=pre_crop.shape[0], bitdepth=16)
                        # Convert array to the Python list of lists expected by the png writer.
                        gray2list = pre_crop.tolist()
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
        bin_mask = newImArray.copy()[:, :, 3]
        cropped_img_bin = Image.fromarray(bin_mask)
        cropBox = cropped_img_bin.getbbox()
    else: #from depth, one-channeled
        #mask binary image
        newImArray = imArray.copy()
        newImArray[mask!=1]=0.
        bin_mask = newImArray
        non_empty_columns = np.where(bin_mask.max(axis=0) > 0)[0]
        non_empty_rows = np.where(bin_mask.max(axis=1) > 0)[0]
        try:
            cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        except ValueError: # mask is empty, no depth data
            cropBox = None
    #cropped_img_bin.show()
    return newImArray, cropBox
