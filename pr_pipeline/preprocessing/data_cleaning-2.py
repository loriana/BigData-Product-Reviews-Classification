from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import col

from utils import multi_clean_text, get_most_freq_val_for_group, show_missing_values_report, convert_empty_str_to_null, \
    drop_na_for_col, impute_placeholder_when_null_or_empty
import os
from pyspark.sql.functions import col, isnan, when, count, udf
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("merging and cleaning").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


def read_in_data(path_to_data: str):
    """
    Iterates through the files in the data directory, and converts them to spark data structures
    ------------------------------------------------------
    Returns a dict of dicts {{}}, where the top level keys are:
    - 'data' (files for train, test, val)
    - 'aux': auxiliary data to be joined with the train, test val data
    Each of 'data' and 'aux' are dictionaries as well, where:
    - the keys are the file names for 'aux', and 'train'/'test'/'validation' for 'data'
    - the values are the data, as a spark structure
    """
    data_spark_aux = {}
    data_spark = {}

    train_paths = []
    test_paths = []
    validation_paths = []
    for filename in os.listdir(path_to_data):
        full_path = os.path.join(path_to_data, filename)
        file_name = filename.split(".")[0].replace("-", "_")

        if "train" in file_name:
            train_paths.append(
                full_path
            )  # we collect the paths here since we need to stack them all
        elif "test" in file_name:
            test_paths.append(full_path)
        elif "validation" in file_name:
            validation_paths.append(full_path)
        elif "category" in file_name:
            data_spark_aux[file_name] = read_category(
                full_path
            )  # category has a different json structure than marketplace
        elif "marketplace" in file_name:
            data_spark_aux[file_name] = read_marketplace(full_path)

    data_spark["train"] = stack_csvs(train_paths).drop("c0")
    data_spark["test"] = stack_csvs(test_paths)
    data_spark["validation"] = stack_csvs(validation_paths)

    return {"data": data_spark, "aux": data_spark_aux}


def read_category(full_path: str):
    """Converts the categories to a pyspark df"""
    df = spark.read.option("multiline", "false").json(full_path)
    return df.withColumnRenamed('name', 'product_category')


def read_marketplace(full_path: str):
    """Converts the marketplaces to a pyspark df"""
    # this I will give a df with 2 cols: id and name, BUT with all the values stores in a 1-row struct :(
    df = spark.read.json(full_path)

    # we want to be able to use explode on the structs, so we'll make them arrays explode takes lists stored in rows,
    # and places each list item on a new row in the same column that used to host the list
    df = df.select(
        F.array(F.expr("id.*")).alias("id"), F.array(F.expr("name.*")).alias("name")
    )

    # we use zip to zip the row-level pairs of id and name before using explode
    # the zip, under this new fake column, will generate a list of pairs like {0, null}, {1, UK}, ...
    # we then explode this list of pairs, so that we make 2 columns, one for each pair item
    df = df.withColumn("id_name", F.explode(F.arrays_zip("id", "name"))).select(
        "id_name.id", "id_name.name"
    )
    return df.withColumnRenamed('name', 'marketplace')


def stack_csvs(paths: list):
    """Concatenates (join) multiple csvs together"""
    stacked_df = (
        spark.read.option("quote", '"')
        .option("escape", '"')
        .csv(paths[0], header=True, inferSchema=True)
    )
    for t in paths[:-1]:
        df_pyspark = (
            spark.read.option("quote", '"')
            .option("escape", '"')
            .csv(t, header=True, inferSchema=True)
        )
        stacked_df = stacked_df.union(df_pyspark)
    return stacked_df


def clean_reviews(spark_df, columns: list):
    """Cleans the title and head of the reviews"""

    # Custom UDF with select()
    return multi_clean_text(columns)(spark_df)


# COLUMNS CLEAN/IMPUTE TASK SPLIT: O-product_id: we drop the whole column regardless, cause it's not so informative
# and if it has missing values, there's no good way to impute O-product parent: get the parent of a product with the
# same id (from another review of this product), if none is found, then depending on the case, drop row or use a
# filler like "no parent" O-product title: get the title of a product with the same id (from another review of this
# product), if none is found, then depending on the case, drop row or use a filler like "no title" O-vine: get most
# frequent value for that product id O-verified_purchase: most frequent value for that product id


# L-review_headline: if missing set to 'no headline' or drop, depending on case review_body: Julio's code handles
# that L-review date: get most frequent timestamp for reviews on this product id, or drop if no products with the
# same id have a date L-marketplace_id: set it to the most frequent marketplace for this product id's reviews,
# but would be cool to also decide based on the review body's language L-product_category_id: set it to the prod cat
# id of other reviews for this product id, else either drop or use a filler value like "no category id" depending on
# case

def join(data, aux, left_on, right_on):
    joined_df = data.join(aux, col(left_on) == col(right_on))
    joined_df = joined_df.drop(right_on)  # dropping the joining col cause we have it under another name
    return joined_df


def show_missing_vals(path_to_data: str):
    all_data_map = read_in_data(path_to_data)
    show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'],
                               all_data_map['data']['validation'])


def clean_data(path_to_data: str):
    """Loads and cleans the data"""
    all_data_map = read_in_data(path_to_data)

    # clean text columns
    for label in all_data_map['data'].keys():
        all_data_map['data'][label] = clean_reviews(
            all_data_map["data"][label],
            ["product_title", "review_headline", "review_body"],
        )

    # convert all empty strings to null
    for label in all_data_map['data'].keys():
        all_data_map['data'][label] = convert_empty_str_to_null(all_data_map['data'][label])

    # show initial report show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'],
    # all_data_map['data']['validation'])

    # product_title has very few missing values, so we'll just drop them across train, test, and validation sets
    # same for the few instances of missing review_body and review_date
    for label in all_data_map['data'].keys():
        all_data_map['data'][label] = drop_na_for_col(all_data_map['data'][label], 'product_title')
        all_data_map['data'][label] = drop_na_for_col(all_data_map['data'][label], 'review_body')
        all_data_map['data'][label] = drop_na_for_col(all_data_map['data'][label], 'review_date')

    # impute missing review headline
    for label in all_data_map['data'].keys():
        all_data_map['data'][label] = impute_placeholder_when_null_or_empty(all_data_map['data'][label],
                                                                            'review_headline', '')

    # --> put this last: drop product id column entirely print('&&&&&&&&&&&&&&& AFTER CLEANING &&&&&&&&&&&&&&&&&&')
    # show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'], all_data_map['data'][
    # 'validation'])

    # join aux data
    for label in all_data_map['data'].keys():
        # product category aux data
        all_data_map['data'][label] = join(all_data_map['data'][label],
                                           all_data_map['aux']['category'],
                                           'product_category_id', 'id')
        # marketplace aux data
        all_data_map['data'][label] = join(all_data_map['data'][label],
                                           all_data_map['aux']['marketplace'],
                                           'marketplace_id', 'id')

    # impute missing marketplace
    for label in all_data_map['data'].keys():
        all_data_map['data'][label] = impute_placeholder_when_null_or_empty(all_data_map['data'][label], 'marketplace',
                                                                            'UNDEFINED')

    # convert Y/N values in verified_purchase and vine columns to 0/1
    for label in all_data_map['data'].keys():
        all_data_map['data'][label]['verified_purchase'] = all_data_map['data'][label]['verified_purchase'].apply(
            lambda x: 1 if x == 'Y' else 0)
        all_data_map['data'][label]['vine'] = all_data_map['data'][label]['vine'].apply(lambda x: 1 if x == 'Y' else 0)

    # extract day of the week from review_date
    for label in all_data_map['data'].keys():
        all_data_map['data'][label]['review_dayofweek'] = pd.to_datetime(
            all_data_map['data'][label]['review_date']).dt.dayofweek

    # map length of review headline and review body to [0,1], with max value as 95
    for label in all_data_map['data'].keys():
        # review headline length
        all_data_map['data'][label]['review_headline_len'] = all_data_map['data'][label]['review_headline'].str.len()
    max_headline_len = all_data_map['data'][label]['review_headline_len'].quantile(0.95)
    all_data_map['data'][label]['review_headline_len'] = all_data_map['data'][label]['review_headline_len'].clip(
        upper=max_headline_len)
    all_data_map['data'][label]['review_headline_len'] = all_data_map['data'][label][
                                                             'review_headline_len'] / max_headline_len
    # review body length
    all_data_map['data'][label]['review_body_len'] = all_data_map['data'][label]['review_body'].str.len()
    max_body_len = all_data_map['data'][label]['review_body_len'].quantile(0.95)
    all_data_map['data'][label]['review_body_len'] = all_data_map['data'][label]['review_body_len'].clip(
        upper=max_body_len)
    all_data_map['data'][label]['review_body_len'] = all_data_map['data'][label]['review_body_len'] / max_body_len


# convert Y/N values of verified_purchase and vine to 0/1
all_data_map['data'][label]['verified_purchase'] = all_data_map['data'][label]['verified_purchase'].map(
    {"Y": 1, "N": 0})
all_data_map['data'][label]['vine'] = all_data_map['data'][label]['vine'].map({"Y": 1, "N": 0})

# check if there are any missing values left
for label in all_data_map['data'].keys():
    print(f"Missing values in {label}:\n{all_data_map['data'][label].isnull().sum()}\n")

if __name__ == "__main__":
    clean_data(path_to_data="data")
