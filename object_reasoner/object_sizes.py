"""
Methods to extract size distributions from raw data
"""

import csv
import argparse
import sys
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json

# Distributions to check
DISTRIBUTIONS = [stats.uniform, stats.norm, stats.lognorm, stats.foldcauchy
    ]
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
        stats.uniform,stats.vonmises,stats.vonmises_line,stats.wald,stats.weibull_min,stats.weibull_max,stats.wrapcauchy
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

def dict_from_csv(csv_gen, classes, base=None):

    if base is None:
        base = {}
        for row in csv_gen:
            obj_name = row[3]
            tgts = [cat for cat in classes if cat in obj_name]
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
    else:
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
        volumes = np.array(data_dict[key]['volume_m3']).astype('float')
        if len(volumes)>=50:
            _, bins, _ = plt.hist(volumes, bins=100, density=True)
            y, x = np.histogram(volumes, bins=100, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0
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


            data_dict[key]['distribution'] = best_distribution.name
            data_dict[key]['params'] = best_params
            """
            #best_pdf = best_distribution.pdf(x, *best_params)
            #plot_pdf(plt, key, x, volumes, best_pdf)
            best_fit_logn = stats.lognorm.pdf(bins, *stats.lognorm.fit(volumes))
            plot_pdf(plt,key, bins,volumes, best_fit_logn)
            """
        else:
            data_dict[key]['distribution'] = None
            data_dict[key]['params'] = None
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doq', help="Path to DoQ csv data")
    parser.add_argument('shp', help="Path to ShapeNetSem csv data")
    parser.add_argument('classes', help="Path to txt file listing the target object classes")
    args = parser.parse_args()
    #default DoQ data header
    #HEADER = ['object', 'head', 'dim', 'mean', 'perc5', 'perc25', 'median', 'perc75', 'perc95', 'std']
    try:
        doq_gen = get_csv_data(args.doq)
        shp_gen = get_csv_data(args.shp, source='ShapeNet')
        with open(args.classes) as clfile:
            CLASSES = [cl.split("\n")[0].replace("_", " ") for cl in clfile.readlines()]
    except:
        print("Please provide valid input paths as specified in the helper")
        return 0

    matches = dict_from_csv(shp_gen, CLASSES)
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


