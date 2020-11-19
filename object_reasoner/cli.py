"""Console script for object_reasoner."""
import argparse
import sys
from object_reasoner import ObjectReasoner
from MLonly.param_tuner import subsample
from preprocessing import utils as utls
from sklearn.model_selection import StratifiedKFold
import statistics
import os,json

def main():
    """Console script for object_reasoner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('test_base', help="Base path to test imgs and product data")
    parser.add_argument('test_res', help="Base path to dataset")
    parser.add_argument('--set', default='KMi', choices=['arc', 'KMi'], help="Chosen dataset")
    parser.add_argument('--baseline', nargs='?', choices=['k-net', 'n-net', 'two-stage','imprk-net', 'triplet', 'baselineNN'],
                        default="k-net",
                        help="Baseline method to retrieve predictions from."
                        "Imprk-net adds weight imprinting based on our pytorch implementation of k-net")
    parser.add_argument('--scenario', nargs='?',
                        choices=['best', 'worst','selected'],
                        default="selected",
                        help="Hybrid correction scenario."
                             "best: correct all which need correction, based on ground truth"
                             "worst: correct all ML predictions indiscriminately"
                             "selected: apply selection based on ML confidence")
    parser.add_argument('--mode', nargs='?', choices=['size'],
                        default="size",
                        help="Reasoning mode. Only reasoning on obj sizes is currently supported.")
    parser.add_argument('--bags', default=None,
                        help='path to bag/bags where depth data are logged')
    parser.add_argument('--regions', default=None,
                        help='path to region annotations: i.e., annotated manually or through automated segmentation')
    parser.add_argument('--origin', default=None,
                        help='path to full size images pre-cropping')
    parser.add_argument('--preds', default='./data/logged-predictions',
                        help='path to logged ML-based predictions as output by ML-only/main.py')
    parser.add_argument('--verbose', type=str2bool, nargs='?',const=True, default=True,
                        help='prints more stats on console, if True')

    args = parser.parse_args()
    if args.set == 'KMi' and args.baseline == 'two-stage':
        print("Known vs Novel leverage only supported for arc set")
        return 0
    overall_res= {m:{} for m in ['MLonly','area','area+flat','area+thin','area+flat+AR','area+thin+AR']}
    reasoner = ObjectReasoner(args)
    reasoner = utls.exclude_nodepth(reasoner,args.baseline)
    # overall_res = reasoner.run(overall_res)
    # Nfold stratified cross-validation for test results
    # subsample test set to devote a small portion to param tuning
    nsplits = 7
    skf = StratifiedKFold(n_splits=nsplits)
    for test1_index, test2_index in skf.split(reasoner.predictions, reasoner.labels):
        reasoner = subsample(reasoner, test1_index, test2_index, args.baseline)
        overall_res = reasoner.run(overall_res)

    mean_res ={}
    for method, subdict in overall_res:
        print("---Cross-fold eval results for method %s----" % method)
        mean_res[method]={}
        for metric_name, metric_array in subdict:
            meanm = statistics.mean(metric_array)
            print("Mean %s: %f" % (metric_name,meanm))
            mean_res[method][metric_name]= meanm
    with open(os.path.join(args.preds,'eval_results_%s_%s'% (args.baseline,args.set)),'w') as jout:
        json.dump(mean_res,jout)
    return 0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
