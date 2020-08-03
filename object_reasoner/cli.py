"""Console script for object_reasoner."""
import argparse
import sys
from object_reasoner import ObjectReasoner


def main():
    """Console script for object_reasoner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('test_base', help="Base path to test imgs and product data")
    parser.add_argument('test_res', help="Base path to test predictions")
    parser.add_argument('--set', default='arc', choices=['arc', 'KMi'], help="Chosen dataset")
    parser.add_argument('--baseline', nargs='?', choices=['k-net', 'n-net', 'two-stage','imprk-net', 'triplet'],
                        default="k-net",
                        help="Baseline method to retrieve predictions from."
                        "Imprk-net adds weight imprinting based on our pytorch re-implementation of k-net")
    parser.add_argument('--mode', nargs='?', choices=['size'],
                        default="size",
                        help="Reasoning mode. Only reasoning on obj sizes is currently supported.")
    parser.add_argument('--bags', default=None,
                        help='path to bag/bags where depth data are logged')

    args = parser.parse_args()
    reasoner = ObjectReasoner(args)
    reasoner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
