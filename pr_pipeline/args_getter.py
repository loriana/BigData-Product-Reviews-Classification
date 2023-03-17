import argparse


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(
        description="Arguments to use gauss_anomaly_detection"
    )

    parser.add_argument(
        "--path-to-data",
        required=True,
        help="Local absolute path to the data",
    )

    parser.add_argument(
        "--preprocess",
        required=False,
        action="store_true",
        help="If specified, preprocess the data",
    )

    parser.add_argument(
        "--missing-val-report",
        required=False,
        action="store_true",
        help="Shows a report with the num of nan/null values per column in the train/test/val datasets",
    )

    parser.add_argument(
        "--verbosity",
        required=False,
        action="store_true",
        help="Whether to print more status updates throughout the different steps",
    )

    args = parser.parse_args()
    return args
