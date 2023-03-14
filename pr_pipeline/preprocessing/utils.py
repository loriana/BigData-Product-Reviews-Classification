import re
import string

# Convert function to udf
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


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
