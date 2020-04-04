### Unified interface to access NLP preprocessing tools from various libraries
# Tokenizers, Stemmers, Lemmatizers
# (i.e. tools for cleaning text before transforming it to vectors or counts)
import pyspark.ml.feature
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, RegexTokenizer
from pyspark.sql.functions import col, when, udf

import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Stemmer
from sparknlp.pretrained import LemmatizerModel, BertEmbeddings, ElmoEmbeddings

lang = 'en'

def RawTokenizer(inputCol, outputCol):
    """Tokenizes words and punctuations with no frills"""
    # sparknlp's Tokenizer requires a Document type as input (created by DocumentAssembler) 
    doc_assembler = DocumentAssembler().setInputCol(inputCol).setOutputCol('_document')
    tokenizer = Tokenizer().setInputCols(['_document']).setOutputCol(outputCol)
    tokenizer_pipe = Pipeline(stages=[doc_assembler, tokenizer])
    return tokenizer_pipe

def NoPuncTokenizer(inputCol, outputCol):
    """Tokenizes words and removes punctuation"""
    tokenizer = RegexTokenizer(inputCol=inputCol, outputCol=outputCol, pattern='\\W')
    return tokenizer

def StopWordsRemover(inputCol, outputCol):
    return pyspark.ml.feature.StopWordsRemover(inputCol=inputCol, outputCol=outputCol)

def Stemmer(inputCol, outputCol):
    stemmer = Stemmer().setInputCols([inputCol]).setOutputCol(outputCol)
    return stemmer

def Lemmatizer(inputCol, outputCol):
    lemmatizer_name = 'lemma_antbnc' # John Snow Labs' pretrained English lemmatizer
    lemmatizer = (LemmatizerModel.pretrained(lemmatizer_name, lang=lang)
            .setInputCols([inputCol])
            .setOutputCol(outputCol)
            )
    return lemmatizer

def Finisher(inputCol, outputCol):
    """
    Required at the end of sparknlp pipelines to return from sparknlp format to 
    python list
    """
    finisher = Finisher().setInputCols([inputCol])
    return finisher.withColumnRenamed('finished_' + inputCol, outputCol)

