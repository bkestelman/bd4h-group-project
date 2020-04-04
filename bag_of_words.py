from pyspark.ml import Pipeline
from nlp_preprocessing_tools import NoPuncTokenizer, StopWordsRemover 

def BagOfWords(inputCol, outputCol):
    tokenizer = NoPuncTokenizer(inputCol=text_col, outputCol='RAW_TOKENS')
    stopWordsRemover = StopWordsRemover(inputCol='RAW_TOKENS', outputCol='TOKENS')
    countVectorizer = CountVectorizer(inputCol='TOKENS', outputCol=outputCol)
    pipe = Pipeline(stages=[
        tokenizer,
        stopWordsRemover,
        countVectorizer,
        ])
    return pipe
