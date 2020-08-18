"""
Methods to extract size distributions from raw data
"""

import csv
import argparse
import sys
from functools import reduce
import operator

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doq', help="Path to DoQ csv data")
    parser.add_argument('shp', help="Path to ShapeNetSem csv data")
    parser.add_argument('classes', help="Path to txt file listing the target object classes")
    args = parser.parse_args()
    #default DoQ data header
    HEADER = ['object', 'head', 'dim', 'mean', 'perc5', 'perc25', 'median', 'perc75', 'perc95', 'std']
    try:
        doq_gen = get_csv_data(args.doq)
        shp_gen = get_csv_data(args.shp, source='ShapeNet')
        with open(args.classes) as clfile:
            CLASSES = [cl.split("\n")[0].replace("_", " ") for cl in clfile.readlines()]
    except:
        print("Please provide valid input paths as specified in the helper")
        return 0

    shp_matches = {}
    for row in shp_gen:
        if row[3] in CLASSES:
            try:
                shp_matches[row[3]]['dims_cm'].append([float(dim) for dim in row[7].split('\,')])
            except:
                shp_matches[row[3]] = {}
                shp_matches[row[3]]['dims_cm'] = []
                shp_matches[row[3]]['volume_cm3'] = []
                shp_matches[row[3]]['volume_m3'] = []
                shp_matches[row[3]]['dims_cm'].append([float(dim) for dim in row[7].split('\,')])

            vol = reduce(operator.mul,[float(dim) for dim in row[7].split('\,')],1)
            shp_matches[row[3]]['volume_cm3'].append(vol)
            shp_matches[row[3]]['volume_m3'].append(float(vol/ 10**6))
            continue

    doq_matches = shp_matches #{}
    for row in doq_gen:
        if row[0] in CLASSES:
            try:
                doq_matches[row[0]]["DoQ_"+row[2]].append(row[3:])

            except KeyError:
                try:
                    doq_matches[row[0]]["DoQ_"+row[2]] = []
                except KeyError:
                    doq_matches[row[0]] = {}
                    doq_matches[row[0]]["DoQ_"+row[2]] = []
                doq_matches[row[0]]["DoQ_"+row[2]].append(row[3:])

    return 0

if __name__ == "__main__":
    sys.exit(main())


