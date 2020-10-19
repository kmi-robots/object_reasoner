"""Console script for object_reasoner."""
import argparse
import sys
from object_reasoner import ObjectReasoner


def main():
    """Console script for object_reasoner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('test_base', help="Base path to test imgs and product data")
    parser.add_argument('test_res', help="Base path to test predictions")
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
    reasoner = ObjectReasoner(args)
    reasoner.run()
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
