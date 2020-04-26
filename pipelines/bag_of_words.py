from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from nlp_preprocessing_tools import NoPuncTokenizer, StopWordsRemover 
from conf.hyperparameters import bag_of_words_params

def BagOfWords(inputCol, outputCol):
#    tokenizer = NoPuncTokenizer(inputCol=inputCol, outputCol='RAW_TOKENS')
#    stopWordsRemover = StopWordsRemover(inputCol='RAW_TOKENS', outputCol='TOKENS')
    countVectorizer = CountVectorizer(inputCol=inputCol, outputCol=outputCol, **bag_of_words_params)
    pipe = Pipeline(stages=[
#        tokenizer,
#        stopWordsRemover,
        countVectorizer,
        ])
    return pipe
