import re
import string

# Convert function to udf
from pyspark.sql.functions import (
    col,
    udf,
    row_number,
    isnan,
    when,
    count,
    length,
    expr,
    dayofweek,
    desc,
    mean,
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window


def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_nonspeech(text):
    clean_text = re.sub(r"[^a-zA-Z0-9\'\s\w\b\.,;\(\)]+", "", text, re.UNICODE)
    return "".join(clean_text)


# Create custom function
def clean_text(str):
    if not str:
        return str
    text = remove_html(str)
    text = remove_emoji(text)
    text = remove_nonspeech(text)
    return text


cleantextUDF = udf(lambda x: clean_text(x), StringType())


def multi_clean_text(col_names):
    def inner(df):
        for col_name in col_names:
            df = df.withColumn(col_name, cleantextUDF(col(col_name)))
        return df

    return inner


def get_most_freq_val_for_group(
    data_df, grouping_col: str, col_to_impute: str, filter_value
):
    """
    This can be used for imputing: if a review for product X is missing a column Y value, there might be another review for X
    that has thos Y value --> we take it from there.

    Usage example: most_freq_parent = get_most_freq_val_for_group(df, 'product_id', 'product_parent', 'B0000251VP')
    ------
    Returns most frequent value for 'col_to_impute' within a grouping by 'grouping_col',
    where the 'grouping_col' has the value `filter_value`.
    """

    grouped = (
        data_df.groupBy(grouping_col, col_to_impute)
        .count()
        .where((col(grouping_col) == filter_value) & col(col_to_impute).isNotNull())
    )
    window = Window.partitionBy(grouping_col).orderBy(col("count").desc())
    return (
        grouped.withColumn("order", row_number().over(window))
        .where(col("order") == 1)
        .first()[f"{col_to_impute}"]
    )


def impute_placeholder_when_null_or_empty(df, col_to_impute, placeholder):
    """Replacs missing or empty string with placeholder value"""
    df = df.withColumn(
        col_to_impute,
        when(
            col(col_to_impute).isNull() | (df[col_to_impute] == ""), placeholder
        ).otherwise(df[col_to_impute]),
    )
    return df


def impute_common_value_when_null_or_empty(df, col_to_impute):
    """Replacs missing or empty string with placeholder value"""
    # First get type
    column_type = str(df.schema[col_to_impute].dataType)

    if column_type in ["StringType()", "IntegerType()"]:
        common_value = (
            df.groupby(col_to_impute).count().orderBy(desc("count")).take(1)[0][0]
        )
        df = df.withColumn(
            col_to_impute,
            when(col(col_to_impute).isNull(), common_value).otherwise(
                df[col_to_impute]
            ),
        )
    elif column_type == "DoubleType()":
        mean_value = df.select(mean("review_body_length")).collect()[0][0]
        df = df.withColumn(
            col_to_impute,
            when(col(col_to_impute).isNull(), mean_value).otherwise(df[col_to_impute]),
        )
    return df


def convert_empty_str_to_null(df):
    """
    Convert empty strings in all string columns to null
    """
    string_cols = [c[0] for c in df.dtypes if c[1] == "string"]
    for col in string_cols:
        df = df.withColumn(col, when(df[col] == "", None).otherwise(df[col]))
    return df


def drop_na_for_col(df, column_name):
    """Drops rows that have the value null at a specific column"""
    df = df.dropna(subset=[column_name], how="all")
    return df


def missing_vals_report_numeric(df):
    """Shows num of missing values / col for all number cols"""
    included_types = [
        "short",
        "int",
        "float",
        "long",
        "byte",
        "double",
        "decimal",
        "numeric",
    ]
    return df.select(
        [
            count(when(isnan(c) | col(c).isNull(), c)).alias(c)
            for c, t in df.dtypes
            if t in included_types
        ]
    )


def missing_vals_report_categorical(df):
    """Shows num of missing values / col for all categorical cols"""
    included_types = ["boolean", "string", "date", "timestamp"]
    return df.select(
        [
            count(when(col(c).isNull(), c)).alias(c)
            for c, t in df.dtypes
            if t in included_types
        ]
    )


def show_missing_values_report(train_df, test_df, val_df):
    """Displays a missing values report for each col of all three datasets, in the form of tables"""

    print("******************* TRAIN *********************")
    print("----- NUMERIC -----")
    missing_vals_report_numeric(train_df).show()
    print("----- CATEGORICAL -----")
    missing_vals_report_categorical(train_df).show()

    print("******************* TEST *********************")
    print("----- NUMERIC -----")
    missing_vals_report_numeric(test_df).show()
    print("----- CATEGORICAL -----")
    missing_vals_report_categorical(test_df).show()

    print("******************* VAL *********************")
    print("----- NUMERIC -----")
    missing_vals_report_numeric(val_df).show()
    print("----- CATEGORICAL -----")
    missing_vals_report_categorical(val_df).show()


def convert_yes_no(df, column):
    return (
        df.withColumn(
            column + "_exp", when(col(column).eqNullSafe("Y"), 1).otherwise(0)
        )
        .drop(column)
        .withColumnRenamed(column + "_exp", column)
    )


def convert_day_of_week(df, column):
    return (
        df.withColumn(column + "_day_of_week_num", dayofweek(column))
        .withColumn(column + "_day_of_week", col(column + "_day_of_week_num") - 1)
        .drop(column + "_day_of_week_num")
    )


def process_review(df, column, percentile):
    tdf = df.withColumn(column + "_length", length(column))
    tdf = tdf.withColumn(
        column + "_percentile",
        expr(f"percentile({column}_length, {percentile})").over(Window.partitionBy()),
    )
    return tdf.withColumn(
        f"{column}_length",
        when(col(f"{column}_length") > col(f"{column}_percentile"), 1).otherwise(
            col(f"{column}_length") / col(f"{column}_percentile")
        ),
    ).drop(column + "_percentile")


# impute_same_ID_val_UDF = udf(lambda x: get_most_freq_val_for_group(x.df, x.grouping_col, x.col_to_impute, x.grouping_col_filter), StringType())


# def impute_by_id_group(df, col_to_impute, placeholder):
#     df.select(['product_id', 'product_title']).where(col('product_title').isNull() | (df['product_title'] == '')).show()
#     prod_ids_missing_title = df.select(['product_id', 'product_title']).where(col('product_title').isNull() | (df['product_title'] == ''))
#     prod_ids_list = [r.product_id for r in prod_ids_missing_title.collect()]

#     df = df.withColumn('product_id', when(col('product_title').isNull(), impute_same_ID_val_UDF(df, 'product_id',
#                                                                                                 'product_title',
#                                                                                                 df['product_id'])).otherwise(df['product_title']))

#     df.select(['product_id', 'product_title']).where(df['product_id'].isin(prod_ids_list)).show()
#     return df
