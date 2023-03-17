import re
import string

# Convert function to udf
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


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
