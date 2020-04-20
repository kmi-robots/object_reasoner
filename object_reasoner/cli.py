"""Console script for object_reasoner."""
import argparse
import sys
from object_reasoner import ObjectReasoner


def main():
    """Console script for object_reasoner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('test_base', help="Base path to test imgs and product data")
    parser.add_argument('test_res', help="Base path to test predictions")
    parser.add_argument('--baseline', nargs='?', choices=['k-net', 'n-net', 'two-stage'],
                        default="k-net",
                        help="Baseline method to retrieve predictions from")
    parser.add_argument('--mode', nargs='?', choices=['size'],
                        default="size",
                        help="Reasoning mode. Only reasoning on obj sizes is currently supported.")

    args = parser.parse_args()
    reasoner = ObjectReasoner(args)
    reasoner.run(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
