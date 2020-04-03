from pyspark.sql.functions import col, when, udf
from pyspark.ml.feature import CountVectorizer, Tokenizer, RegexTokenizer, StopWordsRemover

def tokenize(df, text_col):
    tokenizer = RegexTokenizer(inputCol=text_col, outputCol='RAW_TOKENS', pattern='\\W')
    result = tokenizer.transform(df)
    remover = StopWordsRemover(inputCol='RAW_TOKENS', outputCol=text_col+'_TOKENIZED')
    return remover.transform(result)

def add_bag_of_words(df, words_col):
    cv = CountVectorizer(inputCol=words_col, outputCol='FEATURES')
    model = cv.fit(df)
    result = model.transform(df)
    return result
