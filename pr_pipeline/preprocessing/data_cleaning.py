from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from .utils import multi_clean_text
import os

data_dir_path = "data"
spark = SparkSession.builder.appName("merging and cleaning").getOrCreate()


def read_in_data():
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
    for filename in os.listdir(data_dir_path):
        full_path = os.path.join(data_dir_path, filename)
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

    data_spark["train"] = stack_csvs(train_paths)
    data_spark["test"] = stack_csvs(test_paths)
    data_spark["validation"] = stack_csvs(validation_paths)

    return {"data": data_spark, "aux": data_spark_aux}


def read_category(full_path):
    """Converts the categories to a pyspark df"""
    return spark.read.option("multiline", "false").json(full_path)


def read_marketplace(full_path):
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
    return df.withColumn("id_name", F.explode(F.arrays_zip("id", "name"))).select(
        "id_name.id", "id_name.name"
    )


def stack_csvs(paths):
    """Concatenates (join) multiple csvs together"""
    stacked_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(paths[0], header=True, inferSchema=True)
    )
    for t in paths[:-1]:
        df_pyspark = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(paths[0], header=True, inferSchema=True)
        )
        stacked_df = stacked_df.union(df_pyspark)
    return stacked_df


def clean_reviews(spark_df, columns: list):
    """Cleans the title and head of the reviews"""

    # Custom UDF with select()
    return multi_clean_text(columns)(spark_df)


def clean_data(path_to_data: str):
    """Loads and cleans the data"""
    all_data_map = read_in_data()

    # TODO next @lori: check out the aux data formats and merge the aux data with the train/val/test data
    all_data_map["data"]["train"] = clean_reviews(
        all_data_map["data"]["train"],
        ["product_title", "review_headline", "review_body"],
    )
    print(all_data_map["data"]["train"].show(20))


if __name__ == "__main__":
    clean_data(path_to_data="data")
