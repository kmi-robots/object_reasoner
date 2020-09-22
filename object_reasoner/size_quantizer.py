import argparse
import sys
from object_sizes import get_csv_data
from object_sizes import dict_from_csv
import json


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('shp', help="Path to ShapeNetSem csv file")
    parser.add_argument('--classes', help="Path to class index, json file",
                        default='./data/KMi-set-2020/class_to_index.json', required=False)
    args = parser.parse_args()
    try:
        shp_gen = get_csv_data(args.shp, source='ShapeNet')
        with open(args.classes) as clfile:
            CLASSES = [cl.replace("_", " ") for cl in json.load(clfile).keys()]
    except Exception as e:
        print(str(e))
        print("Please provide valid input paths as specified in the helper")
        return 0

    KB = {} #init empty catalogue
    KB = dict_from_csv(shp_gen,CLASSES,KB,source='ShapeNet') #populate with ShapeNet data
    #TODO: split data in volume bins
    # identify a different manifestation for each bin, if bin frequency above threshold (exclude outliers or non-significant measures)

    #Save KB locally as JSON file
    print("Saving object catalogue under ./data ...")
    with open('./data/KMi_obj_catalogue.json', 'w') as fout:
        json.dump(KB, fout)
    print("File saved as KMi_object_catalogue.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
