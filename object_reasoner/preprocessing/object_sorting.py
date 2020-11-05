import json
import sys
import argparse
import numpy as np
import statistics
from sklearn.neighbors import LocalOutlierFactor

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
    depths = list(sorted(depths1, key=lambda x: x[1]))
    depths2 = list(sorted(depths2, key=lambda x: x[1]))
    depths3 = list(sorted(depths3, key=lambda x: x[1]))

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
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in depths))
        fout.write("========== \n")
        fout.write("Sorted by depth value - ascending (config2)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in depths2))
        fout.write("========== \n")
        fout.write("Sorted by depth value - ascending (config3)\n")
        fout.write('\n'.join('{} {}'.format(x[0], x[1]) for x in depths3))
        fout.write("========== \n")

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
    object_sorting(KB,args)

if __name__ == "__main__":
    sys.exit(main())
