### Unified interface to access NLP preprocessing tools from various libraries
# Tokenizers, Stemmers, Lemmatizers
# (i.e. tools for cleaning text before transforming it to vectors or counts)
import pyspark.ml.feature
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, RegexTokenizer, SQLTransformer
from pyspark.sql.functions import col, when, udf

import sparknlp
import sparknlp.base
from sparknlp.base import RecursivePipeline
from sparknlp.annotator import Tokenizer, Stemmer, SentenceEmbeddings
from sparknlp.pretrained import LemmatizerModel, WordEmbeddingsModel, BertEmbeddings, ElmoEmbeddings, NorvigSweetingModel

lang = 'en'
document_col = '_document' # column name to use as output for sparknlp's DocumentAssembler

def DocAssembler(inputCol, outputCol):
    doc_assembler = (sparknlp.base.DocumentAssembler()
        .setInputCol(inputCol)
        .setOutputCol(outputCol)
        .setCleanupMode('shrink')
        )
    return doc_assembler

def RawTokenizer(inputCol, outputCol):
    """Tokenizes words and punctuations with no frills"""
    # sparknlp's Tokenizer requires a Document type as input (created by DocumentAssembler) 
    doc_assembler = (sparknlp.base.DocumentAssembler()
        .setInputCol(inputCol)
        .setOutputCol(document_col)
        .setCleanupMode('shrink')
        )
    tokenizer = Tokenizer().setInputCols([document_col]).setOutputCol(outputCol)
    tokenizer_pipe = RecursivePipeline(stages=[doc_assembler, tokenizer])
    return tokenizer_pipe

def NoPuncTokenizer(inputCol, outputCol):
    """Tokenizes words and removes punctuation"""
    tokenizer = RegexTokenizer(inputCol=inputCol, outputCol=outputCol, 
        pattern='\\W|[0-9]',
        toLowercase=True)
    return tokenizer

#def SeparatePuncTokenizer(inputCol, outputCol):
#    """
#    Separates words from punctuation (but does not remove punctuation)
#    Useful for Word2Vec - the point is that we want to create vectors for punctuation because
#    it may include useful information, but we don't want to create vectors for specific 
#    combinations of words with punctuation. 
#    Example: "Weight:130" -> ['weight', ':', '130']
#    """
#    tokenizer = RegexTokenizer(inputCol=inputCol, outputCol=outputCol, 
#        gaps=False, # match pattern in text, not gaps
#        pattern='[.,/?!$%()+="\'<>:;]|[^0-9\W]+', # match words and most types of punctuation separately
#        toLowercase=True
#    )
#    return tokenizer
def SeparatePuncTokenizer(inputCol, outputCol):
    """
    Separates words from punctuation (but does not remove punctuation)
    Useful for Word2Vec - the point is that we want to create vectors for punctuation because
    it may include useful information, but we don't want to create vectors for specific 
    combinations of words with punctuation. 
    Example: "Weight:130" -> ['weight', ':', '130']
    """
    tokenizer = (Tokenizer()
        .setInputCols([inputCol])
        .setOutputCol(outputCol)
        #.setContextChars(['(', ')', '?', '!', ':', ';', '/', '+'])
        .setTargetPattern('\w')
        .setSplitPattern('\W')
    )
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

def SpellChecker(inputCol, outputCol):
    #doc_assembler = (sparknlp.base.DocumentAssembler()
    #    .setInputCol(inputCol)
    #    .setOutputCol(document_col)
    #    .setCleanupMode('shrink')
    #    )
    spell_checker = NorvigSweetingModel.pretrained('spellcheck_norvig').setInputCols([inputCol]).setOutputCol(outputCol)
    return spell_checker

def GloveWordEmbeddings(inputCol, outputCol):
    """
    @param inputCol : tokens
    """
    word_embeddings = (WordEmbeddingsModel.pretrained()
        .setInputCols([document_col, inputCol])
        .setOutputCol('EMBEDDINGS')
        )
    document_embeddings = (SentenceEmbeddings()
        .setInputCols([document_col, 'EMBEDDINGS'])
        .setOutputCol('DOC_EMBEDDINGS')
        .setPoolingStrategy('AVERAGE')
        )
    unwrap = SQLTransformer(statement = '''
        SELECT *, EXPLODE(DOC_EMBEDDINGS.embeddings) AS UNWRAPPED_EMBEDDINGS
        FROM __THIS__''')
    list_to_vector = SQLTransformer(statement = '''
        SELECT *, list_to_vector_udf(UNWRAPPED_EMBEDDINGS) AS {outputCol} 
        FROM __THIS__'''.format(outputCol=outputCol))
    embeddings_pipe = RecursivePipeline(stages=[
        word_embeddings, 
        document_embeddings, 
        unwrap,
        list_to_vector,
        ])
    return embeddings_pipe 

def Finisher(inputCol, outputCol):
    """
    Required at the end of sparknlp pipelines to return from sparknlp format to 
    python list
    """
    finisher = sparknlp.base.Finisher().setInputCols([inputCol]).setOutputCols([outputCol])
    return finisher

