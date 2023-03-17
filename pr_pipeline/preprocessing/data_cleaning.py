from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from .utils import multi_clean_text, get_most_freq_val_for_group
import os
from pyspark.sql.functions import col, isnan, when, count


spark = SparkSession.builder.appName("merging and cleaning").getOrCreate()


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


def read_category(full_path):
    """Converts the categories to a pyspark df"""
    return spark.read.option("multiline", "false").json(full_path)


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
    return df.withColumn("id_name", F.explode(F.arrays_zip("id", "name"))).select(
        "id_name.id", "id_name.name"
    )


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


# COLUMNS CLEAN/IMPUTE TASK SPLIT:
# O-product_id: we drop the whole column regardless, cause it's not so informative and if it has missing values, there's no good way to impute
# O-product parent: get the parent of a product with the same id (from another review of this product), if none is found, then depending on the case, drop row or use a filler like "no parent"
# O-product title: get the title of a product with the same id (from another review of this product), if none is found, then depending on the case, drop row or use a filler like "no title"
# O-vine: get most frequent value for that product id
# O-verified_purchase: most frequent value for that product id

# L-review_headline: if missing set to 'no headline' or drop, depending on case
# review_body: Julio's code handles that
# L-review date: get most frequent timestamp for reviews on this product id, or drop if no products with the same id have a date
# L-marketplace_id: set it to the most frequent marketplace for this product id's reviews, but would be cool to also decide based on the review body's language
# L-product_category_id: set it to the prod cat id of other reviews for this product id, else either drop or use a filler value like "no category id" depending on case


def clean_data(path_to_data: str):
    """Loads and cleans the data"""
    all_data_map = read_in_data(path_to_data)

    all_data_map["data"]["train"] = clean_reviews(
        all_data_map["data"]["train"],
        ["product_title", "review_headline", "review_body"],
    )
    print(all_data_map["data"]["train"].show())

    # convert all these string cols that should be bool to bool: string, string

    """
    print(all_data_map["data"]["train"].columns)
    df = all_data_map["data"]["train"]

    # this is just to test sth, and i'll remove it soon (pushed so that Oumayma can see how to use the function)
    df.groupBy("product_id").count().where("count > 1").drop("count").show()
    print('******')
    df.select('product_parent').where(df.product_id == "B0000251VP").show()
    print('******')
    most_freq_parent = get_most_freq_val_for_group(df, 'product_id', 'product_parent', 'B0000251VP')
    print(most_freq_parent)
    """


if __name__ == "__main__":
    clean_data(path_to_data="data")
