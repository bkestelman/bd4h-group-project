from pyspark.ml import Pipeline
from pyspark.ml.feature import (IDF, CountVectorizer, HashingTF, Tokenizer, StopWordsRemover)
from sparknlp.annotator import (Tokenizer, Normalizer, Stemmer)
from sparknlp.base import DocumentAssembler, Finisher
from conf.hyperparameters import tf_idf_params

def TfIdf(inputCol, outputCol):

    # https://codelabs.developers.google.com/codelabs/spark-nlp/#7
    # https://spark.apache.org/docs/latest/ml-features#tf-idf

#    stopword_remover = StopWordsRemover(inputCol='to_spark', outputCol='filtered')

    tf = CountVectorizer(inputCol=inputCol, outputCol='raw_features', **tf_idf_params)
    idf = IDF(inputCol='raw_features', outputCol=outputCol)

    pipe = Pipeline(stages=[
#        stopword_remover,
        tf,
        idf
        ])
    return pipe
