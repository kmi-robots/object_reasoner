""""
Sorts objects automatically on a 2D plane based on raw dimension data
Returns qualitative object properties for a given JSON catalogue
as well as the logarithmic area and depth thresholds used to divide objects in bins
"""

import json
import sys
import argparse
import numpy as np
import statistics
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def object_sorting(KB,args):
    areas1, areas2, areas3 = [], [], []
    depths1, depths2, depths3 = [], [], []
    if args.set =='arc':
        keyword = 'dimensions'
        for k in KB.keys():
            d1, d2, d3 = KB[k][keyword][0], KB[k][keyword][1], KB[k][keyword][2]
            area1, area2, area3 = float(d1 * d2), float(d1 * d3), float(d2 * d3)
            areas1.append((k, area1))
            areas2.append((k, area2))
            areas3.append((k, area3))
            depths1.append((k, d3))
            depths2.append((k, d2))
            depths3.append((k, d1))
    elif args.set =='KMi':
        keyword = 'dims_cm'
        for k in KB.keys():
            measurements = KB[k][keyword] #array of measurements
            if str(True) in str(KB[k]["is_flat"]) and not str(False) in str(KB[k]["is_flat"]):
                #if object marked as striclty flat, we can assume depth is the minimum
                # i.e., only one configuration
                all_depths,all_areas=[],[]
                for dims in measurements:
                    sdims = [float(d / 100) for d in dims]  # convert cm to meters,i.e., as in arc case
                    all_depths.append(min(sdims))
                    sdims.remove(min(sdims))
                    all_areas.append(np.prod(sdims))
                #add mean to class-wise aggregated list
                depths1.append((k,statistics.mean(all_depths)))
                areas1.append((k,statistics.mean(all_areas)))

            else:  # we cannot univoquely map any of the dimensions to w,h,d
                #exploring 3 configurations as in the ARC case
                d1s,d2s,d3s,a1s,a2s,a3s= [],[],[],[],[],[]
                for d1,d2,d3 in measurements:
                    d1 = float(d1/100)
                    d2 = float(d2/100)
                    d3 = float(d3/100)
                    d1s.append(d1) # d1 is the depth
                    d2s.append(d2) #d2 is the depth, etc.
                    d3s.append(d3)
                    a1s.append(d2*d3)
                    a2s.append(d1*d3)
                    a3s.append(d1*d2)

                depths1.append((k,statistics.mean(d1s)))
                depths2.append((k,statistics.mean(d2s)))
                depths3.append((k,statistics.mean(d3s)))
                areas1.append((k, statistics.mean(a1s)))
                areas2.append((k, statistics.mean(a2s)))
                areas3.append((k, statistics.mean(a3s)))

    else:
        print("Dataset not supported yet")
        sys.exit(0)
    #Sort (ascending)
    areas1 = list(sorted(areas1, key=lambda x: x[1]))
    areas2 = list(sorted(areas2, key=lambda x: x[1]))
    areas3 = list(sorted(areas3, key=lambda x: x[1]))
    depths1 = list(sorted(depths1, key=lambda x: x[1]))
    depths2 = list(sorted(depths2, key=lambda x: x[1]))
    depths3 = list(sorted(depths3, key=lambda x: x[1]))

    #OPTIONAL: create output txt to visually inspect sorted results
    with open(('./data/%s_sorting.txt') % args.set, 'w') as fout:
        fout.write("Sorted by surface area - ascending (config1)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in areas1))
        fout.write("========== \n")
        fout.write("Sorted by surface area - ascending (config2)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in areas2))
        fout.write("========== \n")
        fout.write("Sorted by surface area - ascending (config3)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in areas3))
        fout.write("========== \n")
        fout.write("Sorted by depth value - ascending (config1)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in depths1))
        fout.write("========== \n")
        fout.write("Sorted by depth value - ascending (config2)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in depths2))
        fout.write("========== \n")
        fout.write("Sorted by depth value - ascending (config3)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in depths3))
        fout.write("========== \n")

    return [areas1,areas2,areas3,depths1,depths2,depths3]

def bin_creation(obj_dict,input_sorts,n_areabins=5,n_depthbins=4, C=3):
    """
    Create object groups/quadrants based on area surface v depth sorting
    Splits up groups automatically to keep histogram bins equidistributed
    Expects dims of a N x M (e.g.,5x4) grid to be given, depending on how many bins one wants to create
    """
    Xlabels = ['XS','small','medium','large','XL']
    Ylabels = ['flat','thin','thick','bulky']
    area_thresholds, depth_thresholds = [],[]
    for config in range(C): # 3 configurations
        areas, depths = input_sorts[config], input_sorts[config+3]
        areas_ar, depths_ar = np.asarray(list(zip(*areas))[1]), np.asarray(list(zip(*depths))[1])
        areas_ar, depths_ar = np.log(areas_ar),np.log(depths_ar)
        H, xedges, yedges = np.histogram2d(areas_ar,depths_ar,bins=[n_areabins,n_depthbins])
        H = H.T # Let each row list bins with common y range.
        # Plot 2D histogram
        """fig = plt.figure(figsize=(7, 3))
        ax = fig.add_subplot(132, title=('Objects histogram - config %s') % str(config+1),aspect = 'equal')
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, H)
        plt.show()"""
        #Keep track of thresholds used for bin creation
        xedges_log,yedges_log = xedges.tolist().copy(),yedges.tolist().copy()
        area_thresholds.append(xedges_log[1:len(xedges)-1])
        depth_thresholds.append(yedges_log[1:len(yedges)-1])
        #Area bin memberships
        xindices = np.digitize(areas_ar,xedges)
        yindices = np.digitize(depths_ar,yedges)
        for j,(obj1, areav) in enumerate(areas):
            idepth = [i for i,(o,v) in enumerate(depths) if o == obj1][0]
            areabin,depthbin = xindices[j],yindices[idepth]
            #adjust exceeding values
            if areabin > n_areabins: areabin = n_areabins
            if depthbin > n_depthbins: depthbin = n_depthbins
            #print("%s is %s and %s" % (obj1,Xlabels[areabin-1],Ylabels[depthbin-1]))
            #update KB
            try:
                obj_dict[obj1]["has_size"].append('-'.join((Xlabels[areabin-1],Ylabels[depthbin-1])))
            except KeyError:
                obj_dict[obj1]["has_size"] = []
                obj_dict[obj1]["has_size"].append('-'.join((Xlabels[areabin - 1], Ylabels[depthbin - 1])))

    #After all configurations are considered, remove area-depth combination duplicates
    for k in obj_dict.keys():
        size_list = obj_dict[k]["has_size"]
        obj_dict[k]["has_size"] = list(set(size_list))

    return obj_dict,area_thresholds,depth_thresholds

def remove_outliers(obj_dict):
    clf = LocalOutlierFactor(n_neighbors=2,metric='euclidean',n_jobs=-1)
    for k in obj_dict.keys():
        measurements = obj_dict[k]["dims_cm"]  # array of measurements
        try:
            is_inlier = clf.fit_predict(measurements)
            cleaned_measures = [dims for i,dims in enumerate(measurements) if is_inlier[i]==1]
            obj_dict[k]["dims_cm"] = cleaned_measures
        except ValueError:
            #too few examples for outlier removal based on NN
            continue #skip
    return obj_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('catalogue', help="Path to input object catalogue, in JSON format")
    parser.add_argument('--set',choices=['arc','KMi'],default='arc',help="name of dataset to sort", required=False)
    args = parser.parse_args()
    with open(args.catalogue) as fin:
        KB = json.load(fin)

    if args.set=='KMi': KB = remove_outliers(KB)
    sorted_res = object_sorting(KB,args)
    KB, aTs, dTs = bin_creation(KB,sorted_res)

    print("Saving object catalogue under ./data ...")
    with open('./data/KMi_obj_catalogue_autom.json', 'w') as fout:
        json.dump(KB, fout)
    print("File saved as KMi_object_catalogue_autom.json")

    print("The logarithmic area thresholds used for bin creation were")
    print(("Config 1 %s") % str(aTs[0]))
    print(("Config 2 %s") % str(aTs[1]))
    print(("Config 3 %s") % str(aTs[2]))
    print("The logarithmic depth thresholds used for bin creation were")
    print(("Config 1 %s") % str(dTs[0]))
    print(("Config 2 %s") % str(dTs[1]))
    print(("Config 3 %s") % str(dTs[2]))

if __name__ == "__main__":
    sys.exit(main())
