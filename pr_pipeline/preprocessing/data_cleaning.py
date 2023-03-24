from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from .utils import (
    multi_clean_text,
    get_most_freq_val_for_group,
    show_missing_values_report,
    convert_empty_str_to_null,
    drop_na_for_col,
    impute_placeholder_when_null_or_empty,
    convert_day_of_week,
    convert_yes_no,
    process_review,
    impute_common_value_when_null_or_empty,
    get_gpt_score_col
)
import os
from pyspark.sql.functions import col, isnan, when, count, udf, mean
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
    return df.withColumnRenamed("name", "product_category")


def read_marketplace(full_path: str):
    """Converts the marketplaces to a pyspark df"""
    # this'll give a df with 2 cols: id and name, BUT with all the values stores in a 1-row struct :(
    df = spark.read.json(full_path)

    # we want to be able to use explode on the structs, so we'll make them arrays
    # explode takes lists stored in rows, and places each list item on a new row in the same column that used to host the list
    df = df.select(
        F.array(F.expr("id.*")).alias("id"), F.array(F.expr("name.*")).alias("name")
    )

    # we use zip to zip the row-level pairs of id and name before using explode
    # the zip, under this new fake column, will generate a list of pairs like {0, null}, {1, UK}, ...
    # we then explode this list of pairs, so that we make 2 columns, one for each pair item
    df = df.withColumn("id_name", F.explode(F.arrays_zip("id", "name"))).select(
        "id_name.id", "id_name.name"
    )
    return df.withColumnRenamed("name", "marketplace")


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



def join(data, aux, left_on, right_on):
    joined_df = data.join(aux, col(left_on) == col(right_on))
    joined_df = joined_df.drop(
        right_on
    )  # dropping the joining col cause we have it under another name
    return joined_df


def show_missing_vals(path_to_data: str):
    all_data_map = read_in_data(path_to_data)
    show_missing_values_report(
        all_data_map["data"]["train"],
        all_data_map["data"]["test"],
        all_data_map["data"]["validation"],
    )



def clean_data(path_to_data: str):
    """Loads and cleans the data"""
    all_data_map = read_in_data(path_to_data)

    # clean text columns
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = clean_reviews(
            all_data_map["data"][label],
            ["product_title", "review_headline", "review_body"],
        )

    # convert all empty strings to null
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = convert_empty_str_to_null(
            all_data_map["data"][label]
        )

    # show initial report
    # show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'], all_data_map['data']['validation'])

    # product_title has very few missing values, so we'll just drop them across train, and input empty string for test and validation sets
    # same for the few instances of missing review_body and review_date
    for label in all_data_map["data"].keys():
        if label in ["test", "validation"]:
            all_data_map["data"][label] = impute_common_value_when_null_or_empty(
                all_data_map["data"][label], "marketplace_id"
            )
            all_data_map["data"][label] = impute_common_value_when_null_or_empty(
                all_data_map["data"][label], "product_category_id"
            )
            all_data_map["data"][label] = impute_common_value_when_null_or_empty(
                all_data_map["data"][label], "verified_purchase"
            )
            all_data_map["data"][label] = impute_common_value_when_null_or_empty(
                all_data_map["data"][label], "vine"
            )
        else:
            all_data_map["data"][label] = drop_na_for_col(
                all_data_map["data"][label], "product_title"
            )
            all_data_map["data"][label] = drop_na_for_col(
                all_data_map["data"][label], "review_body"
            )
            all_data_map["data"][label] = drop_na_for_col(
                all_data_map["data"][label], "review_date"
            )

    # impute missing review headline
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = impute_placeholder_when_null_or_empty(
            all_data_map["data"][label], "review_headline", ""
        )
        all_data_map["data"][label] = impute_placeholder_when_null_or_empty(
            all_data_map["data"][label], "review_body", ""
        )
        all_data_map["data"][label] = impute_placeholder_when_null_or_empty(
            all_data_map["data"][label], "product_title", ""
        )

    # --> put this last: drop product id column entirely
    # print('&&&&&&&&&&&&&&& AFTER CLEANING &&&&&&&&&&&&&&&&&&')
    # show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'], all_data_map['data']['validation'])

    # join aux data
    for label in all_data_map["data"].keys():
        # product category aux data
        all_data_map["data"][label] = join(
            all_data_map["data"][label],
            all_data_map["aux"]["category"],
            "product_category_id",
            "id",
        )
        # marketplace aux data
        all_data_map["data"][label] = join(
            all_data_map["data"][label],
            all_data_map["aux"]["marketplace"],
            "marketplace_id",
            "id",
        )

    # print('&&&&&&&&&&&&&&& AFTER JOINING &&&&&&&&&&&&&&&&&&')
    # show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'], all_data_map['data']['validation'])

    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = impute_placeholder_when_null_or_empty(
            all_data_map["data"][label], "marketplace", "UNDEFINED"
        )

    # print('&&&&&&&&&&&&&&& AFTER IMPUTING MARKETPLACE &&&&&&&&&&&&&&&&&&')
    # show_missing_values_report(all_data_map['data']['train'], all_data_map['data']['test'], all_data_map['data']['validation'])

    # convert Y/N values in verified_purchase and vine columns to 0/1
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = convert_yes_no(
            all_data_map["data"][label], "verified_purchase"
        )
        all_data_map["data"][label] = convert_yes_no(
            all_data_map["data"][label], "vine"
        )

    # get day of week from review_date and input missing
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = convert_day_of_week(
            all_data_map["data"][label], "review_date"
        )
        all_data_map["data"][label] = impute_common_value_when_null_or_empty(
            all_data_map["data"][label], "review_date_day_of_week"
        )

    # Get length of text fields
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = process_review(
            all_data_map["data"][label], "review_headline", 0.95
        )
        all_data_map["data"][label] = process_review(
            all_data_map["data"][label], "review_body", 0.95
        )

    # get openAI score
    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = get_gpt_score_col(all_data_map["data"][label])


    for label in all_data_map["data"].keys():
        all_data_map["data"][label] = all_data_map["data"][label].select(
            col("_c0"),
            col("product_id"),
            col("product_title"),
            col("review_headline"),
            col("review_body"),
            col("marketplace_id"),
            col("product_category_id"),
            col("verified_purchase"),
            col("vine"),
            col("review_date_day_of_week"),
            col("review_headline_length"),
            col("review_body_length"),
            col("GPT_score")
        )


    return all_data_map


if __name__ == "__main__":
    clean_data(path_to_data="data")
