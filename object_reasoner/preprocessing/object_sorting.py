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
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
from mpl_toolkits.axes_grid1 import make_axes_locatable

Xlabels = ['XS','small','medium','large','XL']
Ylabels = ['flat','thin','thick','bulky']
#Xlabels = ['XS','small','medium','MtL','large','LtX','XL'] #['XS','small','StM','medium','MtL','large','LtX','XL']
#Ylabels = ['flat','thin','THNtTHK','thick','THKtoB','bulky']
all_bins = []
for al in Xlabels:
    for dl in Ylabels:
        all_bins.append(al+"-"+dl)
body_bins = ['large-thin','large-thick','large-bulky','XL-thin','XL-thick',"XL-bulky"]
appearance_wise =['black_fashion_gloves','cherokee_easy_tee_shirt','measuring_spoons',"white_facecloth"]

def object_sorting(KB,args):
    areas1, areas2, areas3 = [], [], []
    depths1, depths2, depths3 = [], [], []
    if args.set =='arc':
        keyword = 'dimensions'
        for k in KB.keys():
            d1, d2, d3 = KB[k][keyword][0], KB[k][keyword][1], KB[k][keyword][2]
            # avoid near-zero values
            if d1 < 0.01:  d1 = 0.01
            if d2 < 0.01:  d2 = 0.01
            if d3 < 0.01:  d3 = 0.01

            area1, area2, area3 = float(d1 * d2), float(d1 * d3), float(d2 * d3)
            areas1.append((k, area1))
            areas2.append((k, area2))
            areas3.append((k, area3))
            depths1.append((k, d3))
            depths2.append((k, d2))
            depths3.append((k, d1))

            if k!='empty' and KB[k]['type'] == 'book' or k in appearance_wise:
                dims = [d1,d2,d3]
                md = max(dims) * 2.
                dims.remove(max(dims))
                dims.append(md)
                d1,d2,d3 = dims
                area1, area2, area3 = float(d1 * d2), float(d1 * d3), float(d2 * d3)
                areas1.append((k, area1))
                areas2.append((k, area2))
                areas3.append((k, area3))
                depths1.append((k, d3))
                depths2.append((k, d2))
                depths3.append((k, d1))

    elif args.set =='lab':
        keyword = 'dims_cm'
        for k in KB.keys():
            measurements = KB[k][keyword] #array of measurements
            if str(True) in str(KB[k]["is_flat"]) and not str(False) in str(KB[k]["is_flat"]):
                #if object marked as striclty flat, we can assume depth is the minimum
                # i.e., only one configuration
                all_depths,all_areas=[],[]
                for dims in measurements:
                    # avoid near-zero values
                    for j,d in enumerate(dims):
                        if d < 1.:  # in cm
                            dims[j] = 1.
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
                for dims in measurements:
                    # avoid near-zero values
                    for j, d in enumerate(dims):
                        if d < 1.:  # in cm
                            dims[j] = 1.
                    d1, d2, d3 = dims
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


def bin_creation(obj_dict,input_sorts,n_areabins=len(Xlabels),n_depthbins=len(Ylabels), C=3):
    """
    Create object groups/quadrants based on area surface v depth sorting
    Splits up groups automatically to keep histogram bins equidistributed
    Expects dims of a N x M (e.g.,5x4) grid to be given, depending on how many bins one wants to create
    """
    area_thresholds, depth_thresholds = [],[]
    #fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(7,3))#, constrained_layout=True)
    for config in range(C): # 3 configurations
        areas, depths = input_sorts[config], input_sorts[config+3]
        areas_ar, depths_ar = np.asarray(list(zip(*areas))[1]), np.asarray(list(zip(*depths))[1])
        areas_ar, depths_ar = np.log(areas_ar),np.log(depths_ar)
        H, xedges, yedges = np.histogram2d(areas_ar,depths_ar,bins=[n_areabins,n_depthbins])
        H = H.T # Let each row list bins with common y range.

        # Plot 2D histogram
        """ax = axes[config]
        #im = ax.imshow(H, interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        #fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        #ax.set_aspect('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        if config ==1: xttl,yttl ='Mean area (d1*d2)', 'Mean thickness (d3)'
        elif config ==2: xttl,yttl ='Mean area (d1*d3)', 'Mean thickness (d2)'
        else: xttl,yttl ='Mean area (d2*d3)', 'Mean thickness (d1)'
        ax.set_xlabel(xttl)
        ax.set_ylabel(yttl)
        X, Y = np.meshgrid(xedges, yedges)
        pcm = ax.pcolormesh(X,Y,H)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.05)
        #cb.set_label('counts in bin')
        #set aspect based on axis limits to obtained three scaled subplots
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)
        """
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

    #plt.tight_layout()
    #plt.show()

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
            outls = [dims for i,dims in enumerate(measurements) if is_inlier[i]==-1]
            cleaned_measures = [dims for i,dims in enumerate(measurements) if is_inlier[i]==1]
            obj_dict[k]["dims_cm"] = cleaned_measures
        except ValueError:
            #too few examples for outlier removal based on NN
            continue #skip
    return obj_dict

def refresh(obj_dict):
    """Remove prior annotations, if any"""
    for k in obj_dict:
        obj_dict[k]['has_size'] = []
        obj_dict[k]['is_flat'] = []
    return obj_dict

def check_flat(obj_dict):
    """
    ARC set: flat info is acquired automatically rather than manually
    (it is not part of the benchmark set)
    """
    for k in obj_dict:
        #ARC set: derive "is_flat" property from thinness annotations
        if "flat" in str(obj_dict[k]['has_size']):
            obj_dict[k]['is_flat'].append(True)
        noflats = [l for l in Ylabels[1:] if l in str(obj_dict[k]['has_size'])]
        if len(noflats)>0:
            obj_dict[k]['is_flat'].append(False)
        obj_dict[k]['is_flat'] = list(set(obj_dict[k]['is_flat']))
    return obj_dict


def rule_adjust(obj_dict):
    """
    Fll in gaps for area annotations. e.g., if annotated as medium and XL also add large
    """
    # all cases: fill gaps on X axis (e.g., object marked as both medium and XL but not large
    for k in obj_dict:
        as_ = list(set([c.split('-')[0] for c in obj_dict[k]['has_size']]))
        if len(as_) > 1:
            ar_indices = [Xlabels.index(a) for a in as_]
            mina, maxa = min(ar_indices), max(ar_indices)
            all_indices = list(range(mina, maxa + 1))
            missing_indices = [ind for ind in all_indices if ind not in ar_indices]
            if len(missing_indices) > 0:
                tgt_thick = [c.split('-')[1] for c in obj_dict[k]['has_size'] if c.split('-')[0] == Xlabels[mina]][
                    0]  # thickness of lowest area bin
                obj_dict[k]['has_size'].extend([Xlabels[ind] + '-' + tgt_thick for ind in all_indices])
    # After all configurations are considered, remove area-depth combination duplicates
    for k in obj_dict.keys():
        size_list = obj_dict[k]["has_size"]
        obj_dict[k]["has_size"] = list(set(size_list))

    return obj_dict

def valid_adjustments(obj_dict):
    """
    Validate the automatically-generated bins with manually collected
    knowledge of flat/non-flat objects
    """
    for k in obj_dict.keys():
        flat_only = False
        if k =='box' or k=='power cord':
            # extreme cases, all bin combinations are possible
            obj_dict[k]['has_size']= all_bins
            continue
        elif k =='person':
            #not enough measures to model people sizes, understimates
            obj_dict[k]['has_size'] = body_bins
            continue
        if str(True) in str(obj_dict[k]["is_flat"]) and not str(False) in str(obj_dict[k]["is_flat"]):
            # if object marked as striclty flat, validate autom generated thinness
            bins_ = [tuple(t.split('-')) for t in obj_dict[k]['has_size']]
            nbins = list(set([ a+'-'+'flat' for a,d in bins_]))
            obj_dict[k]['has_size'] = nbins
            flat_only = True
        elif str(True) in str(obj_dict[k]["is_flat"]) and str(False) in str(obj_dict[k]["is_flat"])\
            and "flat" not in str(obj_dict[k]["has_size"]):
            # can be flat or not, but it was not annotated as flat automatically
            nbins = list(set([s.split("-")[0] for s in obj_dict[k]['has_size']]))
            obj_dict[k]['has_size'].extend([s+"-flat" for s in nbins])
        elif not str(True) in str(obj_dict[k]["is_flat"]) and str(False) in str(obj_dict[k]["is_flat"])\
            and "flat" in str(obj_dict[k]["has_size"]):
            #conversely, object is strictly non-flat, but was marked as flat automatically
            # replace "flat" cases with "thin" instead
            nbins = []
            for s in obj_dict[k]['has_size']:
                if "flat" in s:
                    nbins.append(s.split("-")[0]+"-thin")
                else: nbins.append(s)
            obj_dict[k]['has_size'] = list(set(nbins))

        #all cases: fill gaps on X axis (e.g., object marked as both medium and XL but not large
        as_ = list(set([c.split('-')[0] for c in obj_dict[k]['has_size']]))
        if len(as_)>1:
            ar_indices = [Xlabels.index(a) for a in as_]
            mina,maxa = min(ar_indices),max(ar_indices)
            all_indices = list(range(mina,maxa+1))
            missing_indices = [ind for ind in all_indices if ind not in ar_indices]
            if len(missing_indices)> 0:
                tgt_thick = [c.split('-')[1] for c in obj_dict[k]['has_size'] if c.split('-')[0]==Xlabels[mina]][0] #thickness of lowest area bin
                obj_dict[k]['has_size'].extend([Xlabels[ind]+'-'+tgt_thick for ind in all_indices])

        if not flat_only:
            # fill gaps in between flat and max thinness at a certain area value
            as_ = list(set([c.split('-')[0] for c in obj_dict[k]['has_size']]))
            for a in as_:
                same_clus = [c.split('-')[1] for c in obj_dict[k]['has_size'] if c.split('-')[0]==a]
                if len(same_clus)> 1: # there is at least another bin of same qual area
                    #check if there are gaps to fill in terms of thickness
                    thick_indices = [Ylabels.index(t) for t in same_clus]
                    mint, maxt = min(thick_indices), max(thick_indices)
                    obj_dict[k]['has_size'].extend([a+'-'+Ylabels[k] for k in range(mint+1,maxt)])
            nb = obj_dict[k]['has_size'] #finally, remove dups if any
            obj_dict[k]['has_size'] = list(set(nb))
            continue

    return obj_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('catalogue', help="Path to input object catalogue, in JSON format")
    parser.add_argument('--set',choices=['arc','lab'],default='arc',help="name of dataset to sort", required=False)
    args = parser.parse_args()
    with open(args.catalogue) as fin:
        KB = json.load(fin)

    if args.set=='lab': KB = remove_outliers(KB)
    elif args.set=='arc': KB = refresh(KB)
    sorted_res = object_sorting(KB,args)
    KB, aTs, dTs = bin_creation(KB,sorted_res)
    if args.set=='lab': KB= valid_adjustments(KB) #autom annotation adjusted based on flat/no-flat collected manually
    elif args.set=='arc':
        KB = rule_adjust(KB)
        KB = check_flat(KB) #autom annotation is used to automatically annotate objects as flat-no-flat

    print("The logarithmic area thresholds used for bin creation were")
    print(("Config 1 %s") % str(aTs[0]))
    print(("Config 2 %s") % str(aTs[1]))
    print(("Config 3 %s") % str(aTs[2]))
    print("The logarithmic depth thresholds used for bin creation were")
    print(("Config 1 %s") % str(dTs[0]))  #print(("Config 1 %s") % (str(dthresh_config1)+str(dTs[0][1:]))) ###
    print(("Config 2 %s") % str(dTs[1]))
    print(("Config 3 %s") % str(dTs[2]))

    if args.set =="lab":
        print("Saving object catalogue under ./data ...")
        with open('./data/lab_obj_catalogue_autom.json', 'w') as fout:
            json.dump(KB, fout)
        print("File saved as KMi_object_catalogue_autom.json")
    elif args.set =="arc":
        print("Saving object catalogue under ./data ...")
        with open('./data/arc_obj_catalogue_autom.json', 'w') as fout:
            json.dump(KB, fout)
        print("File saved as arc_object_catalogue_autom.json")

if __name__ == "__main__":
    sys.exit(main())
