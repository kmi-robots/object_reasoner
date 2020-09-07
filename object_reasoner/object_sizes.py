"""
Methods to extract size distributions from various raw data
Creates a python dictionary representing the target object catalogue
"""

import os
import csv
import argparse
import sys
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import re
import time

# Distributions to check
DISTRIBUTIONS = [
        stats.alpha,stats.anglit,stats.arcsine,stats.beta,stats.betaprime,stats.bradford,stats.burr,stats.cauchy,stats.chi,stats.chi2,stats.cosine,
        stats.dgamma,stats.dweibull,stats.erlang,stats.expon,stats.exponnorm,stats.exponweib,stats.exponpow,stats.f,stats.fatiguelife,stats.fisk,
        stats.foldcauchy,stats.foldnorm,stats.frechet_r,stats.frechet_l,stats.genlogistic,stats.genpareto,stats.gennorm,stats.genexpon,
        stats.genextreme,stats.gausshyper,stats.gamma,stats.gengamma,stats.genhalflogistic,stats.gilbrat,stats.gompertz,stats.gumbel_r,
        stats.gumbel_l,stats.halfcauchy,stats.halflogistic,stats.halfnorm,stats.halfgennorm,stats.hypsecant,stats.invgamma,stats.invgauss,
        stats.invweibull,stats.johnsonsb,stats.johnsonsu,stats.ksone,stats.kstwobign,stats.laplace,stats.levy,stats.levy_l,stats.levy_stable,
        stats.logistic,stats.loggamma,stats.loglaplace,stats.lognorm,stats.lomax,stats.maxwell,stats.mielke,stats.nakagami,stats.ncx2,stats.ncf,
        stats.nct,stats.norm,stats.pareto,stats.pearson3,stats.powerlaw,stats.powerlognorm,stats.powernorm,stats.rdist,stats.reciprocal,
        stats.rayleigh,stats.rice,stats.recipinvgauss,stats.semicircular,stats.t,stats.triang,stats.truncexpon,stats.truncnorm,stats.tukeylambda,
        stats.vonmises,stats.vonmises_line,stats.wald,stats.weibull_min,stats.weibull_max,stats.wrapcauchy
    ]

def get_csv_data(filepath, source='DoQ'):
    """
    Yield a large csv row by row to avoid memory overload
    """
    if source =='DoQ':
        with open(filepath, "rt") as csvfile:
            datareader = csv.reader(csvfile, delimiter='\t')
            for row in datareader:
                if row[2] == 'LENGTH' or row[2] == 'VOLUME': #filter only by length or volume measures
                    yield row
    else:
        with open(filepath, "rt") as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader: yield row

def dict_from_csv(csv_gen, classes, base, source='ShapeNet'):
    if source =='ShapeNet':
        for h,row in enumerate(csv_gen):
            if h==0: continue #skip header
            obj_name = row[3]
            super_class = row[1]
            # find all class names matching row keywords
            tgts = []
            for cat in classes:
                if cat == obj_name or cat in obj_name: #exact or partial match
                    if (not "piano" in obj_name) \
                        and (not "lamppost" in obj_name) \
                        and (not (cat == 'chair' and "armchair" in obj_name)):
                        tgts.append((cat,obj_name))
                #elif cat in super_class.lower():
                #    tgts.append((cat, super_class.lower()))
                elif (cat == 'sofa' and "armchair" in obj_name) \
                    or (cat == 'cupboard' and "cabinet" in obj_name) \
                    or (cat == 'big screen' and "tv" in obj_name) \
                    or (cat == 'plant vase' and "vase" in obj_name) \
                    or (cat == 'rubbish bin' and ("trash" in obj_name or "waste" in obj_name) ) \
                    or (cat == 'headphones' and "earphone" in obj_name) \
                    or (cat == 'desk' and "table" in obj_name):
                    tgts.append((cat, obj_name))

                elif (cat == 'sofa' and "couch" in super_class.lower()) \
                    or (cat =='whiteboard' and super_class=='Whiteboard')\
                    or (cat == 'wallpaper' and "WallArt" in super_class) \
                    or ('food' in cat and "FoodItem" in super_class):
                    tgts.append((cat, super_class.lower()))

            if len(tgts)>0:
                if len(tgts)==1: #only one match available
                    cat = tgts[0][0]
                else: #more than one match available
                    #prefer exact match first
                    exacts = [match for cl,match in tgts if cl==match]
                    if len(exacts)>0:
                        cat = exacts[0]
                    else:
                        # partial match
                        # assume it is a compound word - pick last token
                        #print("Compound word in %s" % str(tgts))
                        cat = tgts[0][1].split(' ')[-1]
                try:
                    base[cat]['dims_cm'].append([float(dim) for dim in row[7].split('\,')])
                except: #first time object is added to the dictionary
                    base[cat] = {}
                    base[cat]['dims_cm'] = []
                    base[cat]['volume_cm3'] = []
                    base[cat]['volume_m3'] = []
                    base[cat]['dims_cm'].append([float(dim) for dim in row[7].split('\,')])

                vol = reduce(operator.mul, [float(dim) for dim in row[7].split('\,')], 1)
                base[cat]['volume_cm3'].append(vol)
                base[cat]['volume_m3'].append(float(vol / 10 ** 6))
    else: #DoQ
        for row in csv_gen:
            if row[0] in classes:
                try:
                    base[row[0]]["DoQ_" + row[2]].append(row[3:])
                except KeyError:
                    try:
                        base[row[0]]["DoQ_" + row[2]] = []
                    except KeyError:
                        base[row[0]] = {}
                        base[row[0]]["DoQ_" + row[2]] = []
                    base[row[0]]["DoQ_" + row[2]].append(row[3:])

    return base

def derive_distr(data_dict,N):
    for key in data_dict.keys():
        start = time.time()
        volumes = np.array(data_dict[key]['volume_m3']).astype('float')
        try:
            if len(volumes)>N:
                _, bins, _ = plt.hist(volumes, bins='auto', density=True)
                y, x = np.histogram(volumes, bins='auto', density=True)
                x = (x + np.roll(x, -1))[:-1] / 2.0
                #if key not in clothing:
                best_sse = np.inf
                for distribution in DISTRIBUTIONS:
                    try:
                        # fit dist to data
                        params = distribution.fit(volumes)
                    except: continue
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

                #plot best distribution found
                print("Found best fitting distribution for %s" % key)
                print("Took %f seconds" % float(time.time()-start))
                #plot_pdf(plt,key,x,best_distribution,best_params)
                data_dict[key]['distribution'] = best_distribution.name
                data_dict[key]['params'] = best_params
            else:
                # Otherwise, uniform distribution between min and max
                data_dict[key]['distribution'] = stats.uniform.name
                data_dict[key]['params'] = [volumes.min(), volumes.max()]
        except TypeError:
            # blacklisted object with None value
            data_dict[key]['distribution'] = None
            data_dict[key]['params'] = None
            continue

    return data_dict


def log_normalise(obj_dict,N,clothing,tolerance= 0.05):
    """
    Only find theoretical lognormal when more than x points available
    """
    #probs=[]
    for key in obj_dict.keys():
        try:
            volumes = np.array(obj_dict[key]['volume_m3']).astype('float')
            try:
                dims = np.array(obj_dict[key]['dims_cm']).astype('float')
                mean_dims = np.mean(dims, axis=0).tolist()
            except KeyError: #anthropometric data, mean dims not needed anyways (enough data points)
                mean_dims=[]
            #if key != 'wallpaper' and not key in clothing:  # avoid memory leak for big lists
            #    plot_hist(volumes, title=key, mean_cm=mean_dims)
            if len(volumes)>=N:
                dist = stats.lognorm
                p = dist.fit(volumes)
                #if key != 'wallpaper' and not key in clothing:  # avoid memory leak for big lists
                #    plot_hist(volumes, title=key, mean_cm=mean_dims)
                obj_dict[key]['lognorm-params'] = stats.lognorm.fit(volumes)
                obj_dict[key]['uniform-params'] = None
            elif len(volumes)==2: #was already an hardcoded min max # object with uniform distr between min and max
                dist = stats.uniform
                p = dist.fit(volumes)
                #if key!='wallpaper' and not key in clothing: #avoid memory leak for big lists
                #    plot_hist(volumes, title=key, mean_cm=mean_dims, uniform=True)
                obj_dict[key]['lognorm-params'] = None
                obj_dict[key]['uniform-params'] = p
            else: #object with not enough data points retrieved, create uniform from average value
                # to lower incidence of potential outliers
                #fit uniform instead
                meandims_min = [(d - tolerance * d) for d in mean_dims]  # min-max range of dims
                meandims_max = [(d + tolerance * d) for d in mean_dims]
                meanvol_min, meanvol_max = reduce(operator.mul, meandims_min, 1), reduce(operator.mul, meandims_max, 1)
                vols_uni = [float(meanvol_min / 10 ** 6), float(meanvol_max / 10 ** 6)]
                #if key!='wallpaper' and not key in clothing: #avoid memory leak for big lists
                #    plot_hist(vols_uni, title=key, mean_cm=mean_dims, uniform=True)
                dist = stats.uniform
                p = dist.fit(vols_uni)
                obj_dict[key]['lognorm-params'] = None
                obj_dict[key]['uniform-params'] = p

            #probs.append((key,(dist.cdf((0.05 + 0.0001), *p) - \
            #       dist.cdf((0.05 - 0.0001), *p))))

        except: #DoQ only object or blacklisted
            obj_dict[key]['lognorm-params'] = None
            obj_dict[key]['uniform-params'] = None
    #print(sorted(probs, key=lambda x: x[1],reverse=True))
    return obj_dict

def plot_pdf(plt, obj_name, x,distribution,params):
    pdf = distribution.pdf(x, *params)
    plt.plot(x, pdf, label=distribution.name)
    plt.title(obj_name + " - Distribution of object sizes")
    plt.xlabel("Volume [m3]")
    plt.ylabel("Density [normalised bin counts]")
    plt.legend(loc='best')
    plt.show()


def plot_hist(data_list,title='', mean_cm=None,uniform=False):
    """Plots histogram of list of float values"""
    n, bins, _ = plt.hist(data_list, bins='auto', density=True, label='histogram')
    if not uniform:
        y = stats.lognorm.pdf(bins,*stats.lognorm.fit(data_list))
    else:
        y = stats.uniform.pdf(bins,*stats.uniform.fit(data_list))
    plt.plot(bins, y, label='fit')
    if mean_cm is not None:
        plt.title(title+" (mean: {:.1f} x {:.1f} x {:.1f} cm)".format(mean_cm[0],mean_cm[1],mean_cm[2]))
    else:
        plt.title(title)
    plt.legend()
    plt.show()


def compute_size_proba(obj_name, obj_dict, estimated_size, tolerance = 0.000001): # 1 cm3 tolerance
    try:
        if obj_dict[obj_name]['distribution'] is not None:
            dist = getattr(stats, obj_dict[obj_name]['distribution'])
            p = obj_dict[obj_name]['params']
            prob = dist.cdf((estimated_size+tolerance), *p) -\
                   dist.cdf((estimated_size-tolerance), *p)
        else: prob = None # if no best fitting distribution available for that object
        return prob
    except:
        print("Please provide a valid object name / reference catalogue")
        sys.exit(0)

def add_hardcoded(obj_dict, bespoke_list, ref_csv, tolerance= 0.05): #10% of obj dim
    """
    # Add hardcoded entries
    # overwrites ShapeNet if class present in both (more accurate info)
    # if full==False hardcodes only objects in bespoke_list, otherwise hardcodes all entries found in ref_csv
    returns: object dictionary with hardcoded values + list of objects for which no measures could be defined
    """
    blacklisted =[]
    for i,row in enumerate(ref_csv):
        if i ==0: continue
        obj_name = row[0].replace("_", " ")
        obj_dict[obj_name] = {}

        if row[1] != '':
            dims_min = [float(v) for v in row[1:4]]
            dims_max = [float(v) for v in row[4:]]

        elif row[1]=='' and row[4] !='':
            dims = [float(v) for v in row[4:]]
            dims_min = [(d - tolerance * d) for d in dims]  # min-max range of dims
            dims_max = [(d + tolerance * d) for d in dims]
        else:
            blacklisted.append(obj_name) #blacklisted object
            obj_dict[obj_name]['dims_cm'] = None
            obj_dict[obj_name]['volume_cm3'] = None
            obj_dict[obj_name]['volume_m3'] = None
            continue
        if obj_name in bespoke_list: # update dictionary only for give set of objects
            obj_dict[obj_name]['dims_cm'] = [dims_min, dims_max]
            vol_min, vol_max = reduce(operator.mul, dims_min, 1), reduce(operator.mul, dims_max, 1)
            obj_dict[obj_name]['volume_cm3'] = [vol_min, vol_max]
            obj_dict[obj_name]['volume_m3'] = [float(vol_min / 10 ** 6), float(vol_max / 10 ** 6)]
    return obj_dict, blacklisted


def parse_anthro(csv_gen, unit ='mm'):
    """Expects csv generator and converts measures to cm
    Returns list of measures
    """
    return [float(row[1])/10. for i,row in enumerate(csv_gen) if i>0]

def handle_clothing(obj_dict, path_to_anthropometrics, clothing):
    """
    Derive fitted clothes measurements from human anthropometrics
    Based on ANSUR II (2012) public data retrieved from openlab.psu.edu
    Combined M+F gender
    """
    for p in os.listdir(path_to_anthropometrics):
        if p[-3:] == 'csv':
            if p[:-4] == 'footbreadth':
                footbreadth = get_csv_data(os.path.join(path_to_anthropometrics, p),source='anthro')
                footbreadths = parse_anthro(footbreadth)
            elif p[:-4] == 'footlength':
                footlength = get_csv_data(os.path.join(path_to_anthropometrics, p),source='anthro')
                footlengths = parse_anthro(footlength)
            elif p[:-4] == 'lateralmallheight':
                footheight = get_csv_data(os.path.join(path_to_anthropometrics, p),source='anthro')
                footheights = parse_anthro(footheight)
            elif p[:-4] == 'shoulderwidth':
                shoulwidth = get_csv_data(os.path.join(path_to_anthropometrics, p),source='anthro')
                shoulwidths = parse_anthro(shoulwidth)
            elif p[:-4] == 'sleevelengthspinewrists':
                sleevelth = get_csv_data(os.path.join(path_to_anthropometrics, p),source='anthro')
                sleevelths = parse_anthro(sleevelth)
    #Load anthropometric data
    d = 2 #indicative depth of sweater hanging [cm] #not using chest measures because not wore by person
    for obj_name in clothing:
        obj_dict[obj_name] = {}
        if obj_name =='shoes':
            volumes = [w*l*h for w,l,h in list(zip(footbreadths,footlengths,footheights))]
        elif obj_name =='hanged coat/sweater':
            volumes = [w*l*d for w,l in list(zip(shoulwidths,sleevelths))]

        obj_dict[obj_name]['volume_cm3'] = volumes
        obj_dict[obj_name]['volume_m3'] = [float(vol / 10 ** 6) for vol in volumes]
    return obj_dict

def integrate_scraped(obj_dict, path_to_csvs,remainder_list,blacklist):
    weirdos = []
    for csvp in path_to_csvs:
        scrap_gen = get_csv_data(csvp, source='scraper')
        for i,row in enumerate(scrap_gen):
            if i == 0: continue #skip header
            obj_name = row[0].replace('_', ' ')
            base = row[1].replace(' ','')
            unit = base[-2:]
            if unit == 'cm' or unit =='mm':
                dimensions = base[:-2]
                if 'x' in dimensions:
                    num_dims = len([d for d in dimensions.split('x')])
                    # if it contains any other letters, remove them
                    dim_list = [float(''.join([c if c.isdigit() else '' for c in d])) \
                                    if re.search('[a-zA-Z]', d) is not None else float(d) \
                                for d in dimensions.split('x')]

                elif ',' in dimensions:
                    num_dims = len([d for d in dimensions.split(',')])
                    dim_list = [float(''.join([c if c.isdigit() else '' for c in d])) \
                                    if re.search('[a-zA-Z]', d) is not None else float(d) \
                                for d in dimensions.split(',')]

                if num_dims < 3: #here happens for signs of paper with non-significant depth
                    dim_list.append(0.25)#add depth without compromising volume too much
                if unit=='mm': #convert to cm first
                    dim_list= [float(d/ 10) for d in dim_list]
                if obj_name in remainder_list and obj_name not in blacklist:
                    try:
                        obj_dict[obj_name]['dims_cm'].append(dim_list)
                    except:
                        obj_dict[obj_name] = {}
                        obj_dict[obj_name]['dims_cm'] = []
                        obj_dict[obj_name]['volume_cm3'] = []
                        obj_dict[obj_name]['volume_m3'] = []
                        obj_dict[obj_name]['dims_cm'].append(dim_list)

                    vol = reduce(operator.mul, dim_list, 1)
                    obj_dict[obj_name]['volume_cm3'].append(vol)
                    obj_dict[obj_name]['volume_m3'].append(float(vol / 10 ** 6))

            # empty field or field without unit of measurement
            elif unit=='': continue
            else:
                weirdos.append(row)
                continue
    return obj_dict

def remove_outliers(obj_dict,remainder_list,clothing):
    for key in remainder_list:
        try:
            volumes = np.array(obj_dict[key]['volume_m3']).astype('float')
            """
            if key != 'wallpaper' and key !='toy' and key not in clothing:
                if len(volumes) > N:
                    plot_hist(volumes, title=key)
                else: plot_hist(volumes, title=key, uniform=True)
            """
            dims = np.array(obj_dict[key]['dims_cm']).astype('float')
            #density set to false here to return bin count
            n, bins = np.histogram(volumes, bins='auto') #, density=False)
            nindices_out = np.where(n==1.)[0].tolist() #indices of outliers, i.e., bin frequency ==1
            if len(nindices_out)>0:
                print("removing outliers from class %s" % key)
                edgesindices_out = [(bins[int(i)], bins[int(i+1)]) for i in nindices_out] #equivalent in bin edges
                new_volumes = volumes.tolist()
                new_dims = dims.tolist()
                for vol in sorted(new_volumes, reverse=True): #iterate backwards to keep indices after del
                    for vmin, vmax in edgesindices_out:
                        if vol >= vmin and vol <= vmax:
                            i = new_volumes.index(vol)
                            del new_volumes[i]
                            del new_dims[i]
                            break
                """
                if key != 'wallpaper' and key not in clothing:
                    if len(new_volumes) > N:
                        plot_hist(new_volumes, title=key)
                    else:
                        plot_hist(new_volumes, title=key, uniform=True)
                """
                #update dict with new values
                obj_dict[key]['volume_m3'] = new_volumes
                obj_dict[key]['dims_cm'] = new_dims
            # else no outliers found, do nothing
        except: continue #blacklisted or DoQ only object
    return obj_dict

def select_thresholded(obj_dict):
    """
    Subsample data known to be "bimodal" based on a given threshold
    values were hardcoded here after visual inspection of original histograms
    """
    bis = [('rubbish bin',0.4,'min'), ('whiteboard',0.2,'min'), \
           ('toy',0.3,'min'), ('lamp',0.9,'min'), ('foosball table',0.2,'maj')]

    for name,th,flag in bis:
        try:
            volumes = np.array(obj_dict[name]['volume_m3']).astype('float')
            dims = obj_dict[name]['dims_cm']
        except:  # empty point or object not in catalogue, skip
            continue
        if flag =='min':
            idxs = np.where(volumes >= th)[0].tolist() #indices to remove in dims
            volumes = volumes[volumes < th]
        elif flag=='maj':
            idxs = np.where(volumes <= th)[0].tolist() #indices to remove in dims
            volumes = volumes[volumes > th]
        else:
            print("invalid data flag")
            sys.exit(0)
        #remove equivalent indices from dims
        for idx in sorted(idxs, reverse=True): del dims[idx]
        # and update object dictionary
        obj_dict[name]['volume_m3'] = volumes.tolist()
        obj_dict[name]['dims_cm'] = dims

    return obj_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('shp', help="Path to ShapeNetSem csv file")
    parser.add_argument('scrap', help="Path to Web-scraped csv files")
    parser.add_argument('anthropo', help="Path to anthropometric data")
    parser.add_argument('--classes', help="Path to class index, json file",default='./data/KMi-set-2020/class_to_index.json',required=False)
    parser.add_argument('--doq', help="Path to DoQ csv file", default=None, required=False)
    parser.add_argument('--customc', nargs='+',
                        default=['projector', 'electric heater', 'podium', 'welcome pod', 'printer',
                                 'recording sign','power cord','robot','person', 'fire extinguisher', 'door', 'window'],
                        help="List of classes with custom dimensions", required=False)
    parser.add_argument('--cloth', nargs='+',
                        default=['shoes', 'hanged coat sweater'],
                        help="List of clothing classes to be modelled with anthropometrics", required=False)
    parser.add_argument('--fullfit', action = 'store_true',
                        help="If True, finds best fitting distribution for each object with enough measurements", required=False)
    parser.add_argument('--fullmanual', action = 'store_true',
                        help="If True, only hardcoded measurements are used, otherwise external sources are integrated for specific objects",
                        required=False)
    parser.add_argument('--pmanual', default='./data/KMi_obj_catalogue_manual.csv',
                        help="Path to csv with manually-defined/hardcoded measurements",
                        required=False)
    parser.add_argument('--N', default=40,
                        help="Num of data points below which uniform distribution is used.",
                        required=False)

    args = parser.parse_args()
    try:
        hcsv_gen = get_csv_data(args.pmanual, source='hardcoded')
        shp_gen = get_csv_data(args.shp, source='ShapeNet')
        scrap_csvs = [os.path.join(args.scrap,fname) for fname in os.listdir(args.scrap)\
                      if fname[-3:]=='csv']
        with open(args.classes) as clfile:
            CLASSES = [cl.replace("_", " ") for cl in json.load(clfile).keys()]
            #[cl.split("\n")[0].replace("_", " ") for cl in clfile.readlines()]
    except Exception as e:
        print(str(e))
        print("Please provide valid input paths as specified in the helper")
        return 0

    if os.path.exists('./data/KMi_obj_catalogue.json'):
        print("Retrieve existing catalogue for update...")
        with open('./data/KMi_obj_catalogue.json', 'r') as fin:
            matches = json.load(fin)
        print("Starting lognormal fitting")
        matches = log_normalise(matches,args.N,args.cloth)
        print("Lognormal fitting complete")
    else:

        # Init catalogue with hardcoded objects
        matches = {}
        if args.fullmanual:
            args.customc = CLASSES
            matches,blacklisted = add_hardcoded(matches,args.customc,hcsv_gen)
        else:
            customc = args.customc + ['shoes', 'fire extinguisher sign', 'hanged coat sweater', 'microphone'\
                                     , 'foosball table', 'guitar', 'fire alarm assembly sign', 'coat stand', 'office signs'\
                                     , 'toilet sign', 'radiator', 'mug', 'emergency exit sign', 'drink can',\
                                     'pigeon holes', 'handbag', 'pile of paper', 'desk phone']
            matches, blacklisted = add_hardcoded(matches,customc,hcsv_gen)
            print("Adding more data from external sources")
            # integrating extra measures only for those classes which are nor hardcoded nor marked as blacklisted
            remainder = list(set(CLASSES)-set(customc)-set(blacklisted))

            matches = dict_from_csv(shp_gen,remainder,matches,source='ShapeNet')
            matches = remove_outliers(matches,remainder,args.cloth)
            #matches = handle_clothing(matches, args.anthropo, args.cloth)
            matches = integrate_scraped(matches,scrap_csvs,CLASSES,blacklisted)
            matches = remove_outliers(matches,remainder,args.cloth)

            matches = select_thresholded(matches)
            if args.doq is not None:  # add Google's DoQ set
                # default DoQ data header
                # HEADER = ['object', 'head', 'dim', 'mean', 'perc5', 'perc25', 'median', 'perc75', 'perc95', 'std']
                try:
                    doq_gen = get_csv_data(args.doq)
                except:
                    print("Please provide valid input paths as specified in the helper")
                    return 0
                matches = dict_from_csv(doq_gen,remainder, matches, source='DoQ')
        if args.fullfit:
            print("Starting distribution fit from raw data...This may take a while to complete")
            matches = derive_distr(matches,args.N)
        else:
            print("Starting lognormal fitting")
            matches = log_normalise(matches,args.N,args.cloth)

    #In both cases, save result locally
    print("Saving object catalogue under ./data ...")
    with open('./data/KMi_obj_catalogue.json', 'w') as fout:
        json.dump(matches, fout)
    print("File saved as KMi_object_catalogue.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())


