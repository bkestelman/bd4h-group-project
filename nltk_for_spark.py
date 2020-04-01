import nltk
from pyspark.sql.functions import col, when, udf

import nltk_tokenizer

tokenize_words_udf = udf(nltk_tokenizer.tokenize_words)

def tokenize_words(df, col_to_tokenize):
    df = df.withColumn(col_to_tokenize + '_TOKENIZED',
        tokenize_words_udf(col(col_to_tokenize))
        )
    return df
