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

blacklisted = ['power cable', 'person']
clothing = ['shoes', 'hanged coat/sweater']

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

def dict_from_csv(csv_gen, classes, base=None):
    if base is None: #ShapeNet
        base = {}
        for row in csv_gen:
            obj_name = row[3]
            super_class = row[1]
            tgts = [cat for cat in classes if cat in obj_name or cat in super_class.lower() \
                    and "piano" not in obj_name \
                    or (cat == 'big screen' and "tv" in obj_name) \
                    or (cat == 'wallpaper' and "WallArt" in super_class)\
                    or (cat == 'plant vase' and "vase" in obj_name)\
                    or (cat == 'rubbish bin' and "can" in obj_name)\
                    or ('food' in cat and "FoodItem" in super_class)]  #only keyboards, not piano keyboards

            if len(tgts)>0:
                if len(tgts)==1: #row[3] in classes:
                    cat = tgts[0]
                elif len(tgts)>1: # go for longest matching substring (e.g., bookcase instead of just book)
                    tgts.sort(key=len,reverse=True) #sort by descending length
                    cat = tgts[0]
                try:
                    base[cat]['dims_cm'].append([float(dim) for dim in row[7].split('\,')])
                except:
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

def derive_distr(data_dict):
    for key in data_dict.keys():
        key ='shoes'
        volumes = np.array(data_dict[key]['volume_m3']).astype('float')
        try:
            if len(volumes)>=20:
                _, bins, _ = plt.hist(volumes, bins=100, density=True)
                y, x = np.histogram(volumes, bins=100, density=True)
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

                #else: #follows human anthropometrics
                #    best_distribution = stats.norm
                #    best_params = stats.norm.fit(volumes)

                data_dict[key]['distribution'] = best_distribution.name
                data_dict[key]['params'] = best_params

                best_pdf = best_distribution.pdf(x, *best_params)
                plot_pdf(plt, key, x, volumes, best_pdf)
                #best_fit_logn = stats.lognorm.pdf(bins, *stats.lognorm.fit(volumes))
                #plot_pdf(plt,key, bins,volumes, best_fit_logn)
                continue
            else:
                # Otherwise, uniform distribution between min and max
                data_dict[key]['distribution'] = stats.uniform
                data_dict[key]['params'] = [volumes.min(), volumes.max()]
        except TypeError:
            # blacklisted object with None value
            data_dict[key]['distribution'] = None
            data_dict[key]['params'] = None
            continue

    return data_dict

def plot_pdf(plt, obj_name, bins, data, pdf):

    plt.plot(bins, pdf, label='log-normal')
    plt.title(obj_name + " - Distribution of object sizes")
    plt.xlabel("Volume [m3]")
    plt.ylabel("Density [normalised bin counts]")
    plt.legend(loc='best')
    plt.show()

def load_obj_catalogue(path_to_json):
    with open('./data/KMi_obj_catalogue.json', 'r') as fin:
        matches = json.load(fin)
    return matches

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

def add_hardcoded(obj_dict, bespoke_list, tolerance= 0.10): #10% of obj dim
    # Add hardcoded entries
    # overwrites ShapeNet if class present in both (more accurate info)
    """
    #wire standards
    lengths_cm2 = np.expand_dims(np.arange(start=10., stop=200.),axis=1)
    cross_sections_mm2 = [0.5, 0.75, 1, 1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150,
                              185, 240, 300, 400, 500, 630, 800, 1000]
    cross_sections_cm2 = np.expand_dims(np.array([float(cs / 100.) for cs in cross_sections_mm2]), axis=1)
    vols_cm3 = np.dot(lengths_cm2,cross_sections_cm2.T)
    """
    for obj_name in bespoke_list:
        obj_dict[obj_name] = {}
        if obj_name == 'projector': dims = [11.8, 41.1, 26.8]
        elif obj_name == 'electric heater': dims = [12.5, 24.5, 25.]
        elif obj_name == 'podium': dims = [98.04, 46., 200.]
        elif obj_name == 'welcome pod': dims = [86.36, 48.26, 24.64]
        elif obj_name == 'printer': dims = [96.5, 119.4, 65.4]
        elif obj_name == 'recording sign': dims = [19.5, 22., 135.]
        elif obj_name == 'robot': dims = [38., 35., 31.]
        elif obj_name == 'door':
            dims_min = [61.0,203.2,3.5]
            dims_max = [121.9,300.,4.4]
        elif obj_name == 'window':
            dims_min = [50.8, 61.0, 3.5]
            dims_max = [200., 200., 4.5]
        elif obj_name in blacklisted:
            obj_dict[obj_name]['dims_cm'] = None #size not relevant for wires
            obj_dict[obj_name]['volume_cm3'] = None
            obj_dict[obj_name]['volume_m3'] = None
            continue # skip remainder
        if obj_name not in ['door', 'window', 'person']:
            dims_min = [(d - tolerance*d) for d in dims] #min-max range of dims
            dims_max = [(d + tolerance * d) for d in dims]
        obj_dict[obj_name]['dims_cm'] = [dims_min, dims_max]
        vol_min, vol_max = reduce(operator.mul,dims_min, 1),reduce(operator.mul,dims_max, 1)
        obj_dict[obj_name]['volume_cm3'] = [vol_min, vol_max]
        obj_dict[obj_name]['volume_m3'] = [float(vol_min / 10 ** 6), float(vol_max / 10 ** 6)]
    return obj_dict

def parse_anthro(csv_gen, unit ='mm'):
    """Expects csv generator and converts measures to cm
    Returns list of measures
    """
    return [float(row[1])/10. for i,row in enumerate(csv_gen) if i>0]

def handle_clothing(obj_dict, path_to_anthropometrics):
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

def integrate_scraped(obj_dict, path_to_csvs):
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
                try:
                    obj_dict[obj_name]['dims_cm'].append(dim_list)
                    vol = reduce(operator.mul, dim_list, 1)
                    obj_dict[obj_name]['volume_cm3'].append(vol)
                    obj_dict[obj_name]['volume_m3'].append(float(vol / 10 ** 6))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doq', help="Path to DoQ csv file")
    parser.add_argument('shp', help="Path to ShapeNetSem csv file")
    parser.add_argument('scrap', help="Path to Web-scraped csv files")
    parser.add_argument('anthropo', help="Path to anthropometric data")
    parser.add_argument('classes', help="Path to txt file listing the target object classes")
    parser.add_argument('--customc', nargs='+',
                        default=['projector', 'electric heater', 'podium', 'welcome pod', 'printer',
                                 'recording sign','power cable','robot', 'person', 'door'],
                        help="List of classes with custom dimensions", required=False)
    args = parser.parse_args()
    #default DoQ data header
    #HEADER = ['object', 'head', 'dim', 'mean', 'perc5', 'perc25', 'median', 'perc75', 'perc95', 'std']
    try:
        doq_gen = get_csv_data(args.doq)
        shp_gen = get_csv_data(args.shp, source='ShapeNet')
        scrap_csvs = [os.path.join(args.scrap,fname) for fname in os.listdir(args.scrap)\
                      if fname[-3:]=='csv']
        with open(args.classes) as clfile:
            CLASSES = [cl.split("\n")[0].replace("_", " ") for cl in clfile.readlines()]
    except Exception as e:
        print(str(e))
        print("Please provide valid input paths as specified in the helper")
        return 0

    matches = dict_from_csv(shp_gen, CLASSES)
    matches = add_hardcoded(matches, args.customc)
    matches = handle_clothing(matches, args.anthropo)
    matches = integrate_scraped(matches, scrap_csvs)
    print("Starting distribution fit from raw data...This may take a while to complete")
    matches = derive_distr(matches)
    matches = dict_from_csv(doq_gen, CLASSES, base=matches)
    print("Saving object catalogue under ./data ...")
    with open('./data/KMi_obj_catalogue.json', 'w') as fout:
        json.dump(matches, fout)
    print("File saved as KMi_object_catalogue.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())


