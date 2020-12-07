"""
Methods to read qualitative size data in the same format as ./data/lab_obj_catalogue_manual.csv
and create a JSON file in the format of ./data/lab_obj_catalogue.json
"""

import argparse
import sys
from object_sizes import get_csv_data,dict_from_csv,integrate_scraped, add_hardcoded
import json
import os
import itertools


def add_qual_hardcoded(obj_dict, ref_csv,flat=True, thinness=True, proportion=True): #10% of obj dim
    """
    # Add hardcoded qualitative sizes as small, medium, etc
    returns: object dictionary with hardcoded values
    """
    for i,row in enumerate(ref_csv):
        if i ==0: continue
        obj_name = row[0].replace("_", " ")
        obj_dict[obj_name] = {}
        quals = row[7]
        if "-" in quals:
            #object belongs to more than one bin
            quals = quals.split("-")
        obj_dict[obj_name]['has_size'] = quals
        if thinness: #on a scale from very thin to bulky
            flat = row[8]
            rates = flat.split('/')
            if len(rates)>1: obj_dict[obj_name]['thinness'] = rates
            else: obj_dict[obj_name]['thinness'] = rates[0]
        if flat: # flat v non-flat format
            flat = row[10]
            if "-" in flat: obj_dict[obj_name]['is_flat'] =[True,False]
            elif '0' in flat: obj_dict[obj_name]['is_flat'] = False
            elif '1' in flat:obj_dict[obj_name]['is_flat'] = True
        if proportion:
            prop = row[9]
            rates = prop.split('/')
            if len(rates)>1: obj_dict[obj_name]['aspect_ratio'] = rates
            else: obj_dict[obj_name]['aspect_ratio'] = rates[0]

    return obj_dict

def add_ARflat_hardcoded(obj_dict, ref_csv):
    for i,row in enumerate(ref_csv):
        if i ==0: continue
        obj_name = row[0].replace("_", " ")
        prop = row[9]
        rates = prop.split('/')
        if len(rates) > 1:
            obj_dict[obj_name]['aspect_ratio'] = rates
        else:
            obj_dict[obj_name]['aspect_ratio'] = rates[0]
        flat = row[10]
        if "-" in flat:obj_dict[obj_name]['is_flat'] = [True, False]
        elif '0' in flat:obj_dict[obj_name]['is_flat'] = False
        elif '1' in flat: obj_dict[obj_name]['is_flat'] = True
    return obj_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('shp', help="Path to ShapeNetSem csv file")
    parser.add_argument('--scrap_path', help="Path to scraped csv data,if any is used", required=False)
    parser.add_argument('--classes', help="Path to class index, json file",
                        default='./data/Lab-set/class_to_index.json', required=False)
    parser.add_argument('--pmanual', default='./data/lab_obj_catalogue_manual.csv',
                        help="Path to csv with manually-defined/hardcoded measurements",
                        required=False)
    args = parser.parse_args()
    try:
        shp_gen = get_csv_data(args.shp, source='ShapeNet')
        hcsv_gen = get_csv_data(args.pmanual, source='hardcoded')
        with open(args.classes) as clfile:
            CLASSES = [cl.replace("_", " ") for cl in json.load(clfile).keys()]
    except Exception as e:
        print(str(e))
        print("Please provide valid input paths as specified in the helper")
        return 0

    KB = {} #init empty catalogue
    #KB = add_qual_hardcoded(KB, hcsv_gen)
    KB = dict_from_csv(shp_gen,CLASSES,KB,source='ShapeNet') #populate with ShapeNet data
    remainder = [c for c in CLASSES if c not in KB]
    scraped_files = [os.path.join(args.scrap_path,fname) for fname in os.listdir(args.scrap_path) if fname.endswith('csv')]
    KB = integrate_scraped(KB,scraped_files,remainder,[])
    remainder = [c for c in CLASSES if c not in KB] # or len(KB[c]["dims_cm"])<=10]
    remainder.append("printer") #only home printers found via Web
    #Save KB locally as JSON file
    hcsv_gen, ar_gen = itertools.tee(hcsv_gen)
    KB= add_hardcoded(KB,remainder,hcsv_gen)
    KB = add_ARflat_hardcoded(KB, ar_gen)

    try: #load flat/no-flat from existing json, if any is found
        with open('./data/lab_obj_catalogue_autom_raw.json', 'r') as fin:
            backup = json.load(fin)
            for k in KB:
                KB[k]['is_flat'] = backup[k]['is_flat']
    except: pass
    #save updated KB locally
    print("Saving object catalogue under ./data ...")
    with open('./data/lab_obj_catalogue_autom_raw.json', 'w') as fout:
        json.dump(KB, fout)
    print("File saved as KMi_object_catalogue_autom_raw.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
