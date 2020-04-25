import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import yaml

def extract_foreground_2D(dm):
    #flatten matrix first
    dmat = dm.copy()
    # plt.imshow(dmat, cmap='Greys_r')
    # plt.show()
    fmatrix = np.reshape(dmat,(dm.shape[0]*dm.shape[1], 1))
    clt = KMeans(n_clusters=3)  # background, foreground, no-value
    clt.fit(fmatrix)
    #Find cluster index for closest points to camera (foreground)
    #Second smallest to avoid zero value cluster
    tgt_idx = np.where(clt.cluster_centers_ == sorted(clt.cluster_centers_)[1][0])[0][0]
    cbin = np.uint8(fmatrix.copy()) #binarized version
    for i in range(clt.labels_.shape[0]):
        #color pixels based on cluster label
        if clt.labels_[i]==tgt_idx:   #foreground
            cbin[i, :] = 255
        else: #either background or no value, treated the same
            cbin[i, :] = 0
    # plt.imshow(np.reshape(cbin, dm.shape), cmap='Greys_r')
    # plt.show()
    return np.reshape(cbin, dm.shape)


def detect_contours(dmatrix, cbin):
    """
    Expects clustered version of depth img
    with only black and white colors
    """
    # plt.imshow(cbin, cmap='Greys_r')
    # plt.show()
    contours, hierarchy = cv2.findContours(cbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # select largest contour
    c = max(contours, key=cv2.contourArea)
    # Create binary mask from largest contour
    out_mask = np.zeros_like(dmatrix)
    cv2.drawContours(out_mask, [c], -1, (255), cv2.FILLED, 1)
    # mask original depth image
    out = dmatrix.copy()
    out[out_mask == 0] = 0.
    # print(out[out != 0])
    """
    plt.imshow(out_mask,cmap='Greys_r')
    plt.show()
    plt.imshow(out,cmap='Greys_r')
    plt.show()
    """
    return out



