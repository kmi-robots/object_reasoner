import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from collections import Counter

def cluster_3D(pcl, eps=0.1, minpoints=10, vsize=0.05):
    """
    DBSCAN clustering for pointclouds
    vsize: voxel size for downsampling
    """
    downpcl= pcl.voxel_down_sample(voxel_size=vsize)
    # o3d.visualization.draw_geometries([downpcl])
    # numpoints = np.asarray(downpcl.points).shape[0]
    # minpoints = int(round(numpoints / ratio))
    print("Clustering point cloud")
    start = time.time()
    labels = np.array(downpcl.cluster_dbscan(eps=eps, min_points=minpoints, print_progress=True))
    print("Took % fseconds." % float(time.time() - start))
    max_label = labels.max()
    print("Pcl has %d clusters" % (max_label + 1))

    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcl.colors = o3d.utility.Vector3dVector(colors[:, :3])

    #Select largest cluster
    # Skipping -1, where -1 indicates noise as in open3d docs
    win_label = [key for key, val in Counter(labels).most_common() if key != -1][0]
    indices = [i for i, x in enumerate(labels) if x == win_label]

    new_pcl=downpcl.select_down_sample(indices)
    """
    print("Subsampling point cloud")
    start = time.time()
    new_pcl, intlist = pcl.remove_radius_outlier(nb_points=minpoints, radius=eps)
    print("Took % fseconds." % float(time.time() - start))
    """
    # o3d.visualization.draw_geometries([new_pcl])
    return new_pcl

def MatToPCL(imgMat, camera_intrinsics):
    """
    Convert depth matrix
    to pointcloud object
    """
    # plt.imshow(imgMat)
    # plt.show()
    dimg = o3d.geometry.Image(imgMat)
    o3d.io.write_image("./temp_depth.png", dimg)
    depth_raw = o3d.io.read_image("./temp_depth.png")
    os.remove("./temp_depth.png")
    # plt.imshow(depth_raw,cmap='Greys_r')
    # plt.show()
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, camera_intrinsics,depth_scale=10000.0, depth_trunc=10000.0)
    # plt.imshow(depth_raw)
    # plt.show()
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])

    return pcd

def PathToPCL(imgpath, camera_intrinsics):
    """
    Convert depth image as filepath
    to pointcloud object
    """
    depth_raw = o3d.io.read_image(imgpath)
    # plt.imshow(depth_raw,cmap='Greys_r')
    # plt.show()
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, camera_intrinsics,depth_scale=10000.0, depth_trunc=10000.0)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def estimate_dims(pcd,original_pcd):
    """
    Finds bounding solid and
    estimates obj dimensions from vertices
    """
    # threedbox = pcd.get_axis_aligned_bounding_box()
    orthreedbox = pcd.get_oriented_bounding_box()
    # orthreedbox = pcd.get_axis_aligned_bounding_box()
    # print(orthreedbox.dimension())
    box_points = np.asarray(orthreedbox.get_box_points())
    box_center = np.asarray(orthreedbox.get_center())

    # o3d.visualization.draw_geometries([original_pcd, orthreedbox])

    #cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin= box_points[0])
    #cf2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=box_points[1])
    #cf3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=box_points[2])
    #cf4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=box_points[3])
    # o3d.visualization.draw_geometries([cf, cf2, cf3, cf4, orthreedbox, pcd])

    #Compute width, height and depth from bbox points
    #Point order not documented. But First 3 vertices look like they lie always on same surface
    # and the 4th one perpendicular to the first one
    d1 = np.linalg.norm(box_points[0] - box_points[1])
    d2 = np.linalg.norm(box_points[0] - box_points[2])
    d3 = np.linalg.norm(box_points[0] - box_points[3])

    """
    Hard to know a priori what is the w and what is the h
    But we can assume the depth will be always the min due to how data are captured
    """
    return d1,d2,min(d1,d2,d3)
