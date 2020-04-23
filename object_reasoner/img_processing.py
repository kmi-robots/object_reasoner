import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import yaml

def extract_foreground(dm):
    #flatten matrix first
    dmat = dm.copy()
    fmatrix = np.reshape(dmat,(dm.shape[0]*dm.shape[1], dm.shape[-1]))
    clt = KMeans(n_clusters=3)  # background, foreground, no-value
    clt.fit(fmatrix)
    #Find cluster index for closest points to camera (foreground)
    tgt_idx = np.argmin(clt.cluster_centers_, axis=0)[0]
    # cmatrix = np.uint8(fmatrix.copy())
    cbin = np.uint8(fmatrix.copy()) #binarized version
    for i in range(clt.labels_.shape[0]):
        #color pixels based on cluster label
        if clt.labels_[i]==tgt_idx:   #foreground
            # cmatrix[i,:] = np.array([0, 255, 0])
            cbin[i, :] = np.array([0, 0, 0])
        else: #either background or no value, treated the same
            # cmatrix[i, :] = np.array([0, 0, 255])
            cbin[i, :] = np.array([255, 255, 255])

    # cmatrix = np.reshape(cmatrix,dm.shape)
    return np.reshape(cbin, dm.shape)

def detect_contours(dmatrix, cbin):
    """
    Expects clustered version of depth img
    with only black and white colors
    """
    plt.imshow(cbin, cmap='Greys_r')
    plt.show()
    gray = cv2.cvtColor(cbin.astype(np.float32), cv2.COLOR_RGB2GRAY)
    #_, binary = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(gray,cmap='Greys_r')
    plt.show()
    # plt.imshow(binary,cmap='Greys_r')
    # plt.show()
    contours, hierarchy = cv2.findContours(gray.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image = cv2.drawContours(cbin, contours, -1, (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()
    #bounding rotated rectangle of largest contour
    c = max(contours, key=cv2.contourArea)
    out_mask = np.zeros_like(dmatrix)
    cv2.drawContours(out_mask, [c], -1, (255), cv2.FILLED, 1)
    out = dmatrix.copy()
    out[out_mask == 0] = 0.
    # minRect = cv2.minAreaRect(c)
    # box = np.intp(cv2.boxPoints(minRect))
    # image = cv2.drawContours(cbin, [c], 0, (0,0,255),thickness=4)
    """
    plt.imshow(dmatrix)  # ,cmap='Greys_r')
    plt.show()
    """
    # print(out[out != 0])
    plt.imshow(out_mask)
    plt.show()
    plt.imshow(out)#,cmap='Greys_r')
    plt.show()

    return out

