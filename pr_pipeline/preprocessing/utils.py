import re
import string

# Convert function to udf
from pyspark.sql.functions import col, udf, row_number, isnan, when, count
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

def get_most_freq_val_for_group(data_df, grouping_col: str, col_to_impute: str, filter_value):
    '''
    This can be used for imputing: if a review for product X is missing a column Y value, there might be another review for X
    that has thos Y value --> we take it from there.

    Usage example: most_freq_parent = get_most_freq_val_for_group(df, 'product_id', 'product_parent', 'B0000251VP')
    ------
    Returns most frequent value for 'col_to_impute' within a grouping by 'grouping_col',
    where the 'grouping_col' has the value `filter_value`.
    '''
    grouped = data_df.groupBy(grouping_col, col_to_impute).count().where((col(grouping_col) == filter_value) & col(col_to_impute).isNotNull())
    window = Window.partitionBy(grouping_col).orderBy(col('count').desc())
    return grouped\
        .withColumn('order', row_number().over(window))\
        .where(col('order') == 1)\
        .first()[f'{col_to_impute}']


def impute_placeholder_when_null_or_empty(df, col_to_impute, placeholder):
    '''Replacs missing or empty string with placeholder value'''
    df = df.withColumn(col_to_impute, when(col(col_to_impute).isNull() | (df[col_to_impute] == ''), placeholder).otherwise(df[col_to_impute]))
    return df


def convert_empty_str_to_null(df):
    '''
    Convert empty strings in all string columns to null
    '''
    string_cols = [c[0] for c in df.dtypes if c[1] == 'string']
    for col in string_cols:
        df = df.withColumn(col, when(df[col] == '', None).otherwise(df[col]))
    return df


def drop_na_for_col(df, column_name):
    '''Drops rows that have the value null at a specific column'''
    df = df.dropna(subset=[column_name], how="all")
    return df


def missing_vals_report_numeric(df):
    '''Shows num of missing values / col for all number cols'''
    included_types =  ['short', 'int', 'float', 'long', 'byte', 'double', 'decimal', 'numeric']
    return df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c, t in df.dtypes if t in included_types])


def missing_vals_report_categorical(df):
    '''Shows num of missing values / col for all categorical cols'''
    included_types = ['boolean', 'string', 'date', 'timestamp']
    return df.select([count(when(col(c).isNull() | (df[c] == ''), c)).alias(c) for c, t in df.dtypes if t in included_types])


def show_missing_values_report(train_df, test_df, val_df):
    '''Displays a missing values report for each col of all three datasets, in the form of tables'''

    print('******************* TRAIN *********************')
    print('----- NUMERIC -----')
    missing_vals_report_numeric(train_df).show()
    print('----- CATEGORICAL -----')
    missing_vals_report_categorical(train_df).show()

    print('******************* TEST *********************')
    print('----- NUMERIC -----')
    missing_vals_report_numeric(test_df).show()
    print('----- CATEGORICAL -----')
    missing_vals_report_categorical(test_df).show()

    print('******************* VAL *********************')
    print('----- NUMERIC -----')
    missing_vals_report_numeric(val_df).show()
    print('----- CATEGORICAL -----')
    missing_vals_report_categorical(val_df).show()
      


