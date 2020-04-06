from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from nlp_preprocessing_tools import NoPuncTokenizer, StopWordsRemover 
from utils import timeit

@timeit
def BagOfWords(inputCol, outputCol):
    tokenizer = NoPuncTokenizer(inputCol=inputCol, outputCol='RAW_TOKENS')
    stopWordsRemover = StopWordsRemover(inputCol='RAW_TOKENS', outputCol='TOKENS')
    countVectorizer = CountVectorizer(inputCol='TOKENS', outputCol=outputCol)
    pipe = Pipeline(stages=[
        tokenizer,
        stopWordsRemover,
        countVectorizer,
        ])
    return pipe
