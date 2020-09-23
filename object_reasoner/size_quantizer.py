import argparse
import sys
from object_sizes import get_csv_data
from object_sizes import dict_from_csv
import json


def add_qual_hardcoded(obj_dict, ref_csv): #10% of obj dim
    """
    # Add hardcoded qualitative sizes as small, medium, etc
    returns: object dictionary with hardcoded values
    """
    for i,row in enumerate(ref_csv):
        if i ==0: continue
        obj_name = row[0].replace("_", " ")
        obj_dict[obj_name] = {}
        quals = row[7]
        flat = row[8]
        if "-" in quals:
            #object belongs to more than one bin
            quals = quals.split("-")
        obj_dict[obj_name]['has_size'] = quals
        if "-" in flat: obj_dict[obj_name]['is_flat'] =[True,False]
        elif '0' in flat: obj_dict[obj_name]['is_flat'] = False
        elif '1' in flat:obj_dict[obj_name]['is_flat'] = True
    return obj_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('shp', help="Path to ShapeNetSem csv file")
    parser.add_argument('--classes', help="Path to class index, json file",
                        default='./data/KMi-set-2020/class_to_index.json', required=False)
    parser.add_argument('--pmanual', default='./data/KMi_obj_catalogue_manual.csv',
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
    KB = add_qual_hardcoded(KB, hcsv_gen)
    #KB = dict_from_csv(shp_gen,CLASSES,KB,source='ShapeNet') #populate with ShapeNet data

    #Save KB locally as JSON file
    print("Saving object catalogue under ./data ...")
    with open('./data/KMi_obj_catalogue.json', 'w') as fout:
        json.dump(KB, fout)
    print("File saved as KMi_object_catalogue.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
