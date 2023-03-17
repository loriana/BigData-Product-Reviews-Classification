from .args_getter import get_args
from .preprocessing.data_cleaning import clean_data, show_missing_vals


def main():
    # Import args
    args = get_args()

    if args.missing_val_report:
        show_missing_vals(args.path_to_data)

    if args.preprocess:
        clean_data(args.path_to_data)



if __name__ == "__main__":
    main()
