import nltk
from pyspark.sql.functions import col, when, udf

import nltk_tokenizer

tokenize_words_udf = udf(nltk_tokenizer.tokenize_words)

def tokenize_words(df, col_to_tokenize):
    df = df.withColumn(col_to_tokenize + '_TOKENIZED',
        tokenize_words_udf(col(col_to_tokenize))
        )
    return df

def bag_of_words(words):
    bag = {}
    for word in words:
        if word in bag.keys():
            bag[word] += 1
        else:
            bag[word] = 1
    return bag

bag_of_words_udf = udf(bag_of_words)

def add_bag_of_words(df, words_col):
    df = df.withColumn('BAG_OF_WORDS',
        bag_of_words_udf(col(words_col))
        )
    return df
