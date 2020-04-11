from pyspark.ml import Pipeline
from pyspark.ml.feature import (IDF, CountVectorizer, HashingTF, Tokenizer, StopWordsRemover)
from sparknlp.annotator import (Tokenizer, Normalizer, Stemmer)
from sparknlp.base import DocumentAssembler, Finisher
from hyperparameters import tf_idf_params

def TfIdf(inputCol, outputCol):

    # https://codelabs.developers.google.com/codelabs/spark-nlp/#7
    # https://spark.apache.org/docs/latest/ml-features#tf-idf

    document_assembler = DocumentAssembler().setInputCol(inputCol).setOutputCol('document')
    tokenizer = Tokenizer().setInputCols(['document']).setOutputCol('token')
    normalizer = Normalizer().setInputCols(['token']).setOutputCol('normalizer')
    # stemmer = Stemmer().setInputCols(['normalizer']).setOutputCol('stem')
    # finisher = Finisher().setInputCols(['stem']).setOutputCols(['to_spark']).setValueSplitSymbol(' ')
    finisher = Finisher().setInputCols(['normalizer']).setOutputCols(['to_spark'])
    stopword_remover = StopWordsRemover(inputCol='to_spark', outputCol='filtered')
    tf = CountVectorizer(inputCol='filtered', outputCol='raw_features', **tf_idf_params)
    idf = IDF(inputCol='raw_features', outputCol=outputCol)

    pipe = Pipeline(stages=[
        document_assembler,
        tokenizer,
        normalizer,
        # stemmer,
        finisher,
        stopword_remover,
        tf,
        idf
        ])
    return pipe
