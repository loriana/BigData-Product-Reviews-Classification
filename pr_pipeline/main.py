from .args_getter import get_args
from .preprocessing.data_cleaning import clean_data


def main():
    # Import args
    args = get_args()

    if args.preprocess:
        clean_data(args.path_to_data)


if __name__ == "__main__":
    main()
