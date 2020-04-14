### Includes word2vec and other word embeddings
from pyspark import SparkContext
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType

import nlp_preprocessing_tools as nlp
from nlp_preprocessing_tools import RawTokenizer, Lemmatizer, Finisher
from hyperparameters import word2vec_params

def BasicWord2Vec(inputCol, outputCol):
    tokenizer = RawTokenizer(inputCol=inputCol, outputCol='TOKENS') 
    finisher = Finisher(inputCol='TOKENS', outputCol='FINISHED_TOKENS') 
    word2vec = Word2Vec(inputCol='FINISHED_TOKENS', outputCol=outputCol, minCount=0, **word2vec_params) 
    pipe = Pipeline(stages=[
        tokenizer,
        finisher,
        word2vec,
        ])
    return pipe

def GloveWordEmbeddings(inputCol, outputCol):
    tokenizer = RawTokenizer(inputCol=inputCol, outputCol='TOKENS') 
    word_embeddings = nlp.GloveWordEmbeddings(inputCol='TOKENS', outputCol=outputCol)
    pipe = Pipeline(stages=[
        tokenizer,
        word_embeddings,
        ])
    return pipe

class Words2MatrixTransformer(Transformer):
    """
    Uses a trained Word2VecModel to transform tokenized text to a matrix composed of word
    vectors
    """
    def __init__(self, inputCol, outputCol, word2vecModel):
        super(Words2MatrixTransformer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.word2vecModel = word2vecModel

    def transform(self, df):
        word_vectors = self.word2vecModel.getVectors()
        sc = SparkContext.getOrCreate()
        broadcast_vectors = sc.broadcast(word_vectors.rdd.collectAsMap()) 
        def words2matrix(words):
            matrix = []
            for word in words:
                vec = broadcast_vectors.value.get(word)
                if vec is not None:
                    matrix.append(broadcast_vectors.value[word])
            return matrix
        words2matrix_udf = udf(words2matrix, ArrayType(VectorUDT()))
        return df.withColumn(self.outputCol, words2matrix_udf(self.inputCol))

def Words2Matrix(inputCol, outputCol, word2vecModel):
    tokenizer = RawTokenizer(inputCol=inputCol, outputCol='TOKENS') 
    finisher = Finisher(inputCol='TOKENS', outputCol='FINISHED_TOKENS') 
    pipe = Pipeline(stages=[
        tokenizer,
        finisher,
        Words2MatrixTransformer('FINISHED_TOKENS', outputCol, word2vecModel),
        ])
    return pipe

# BERT is a state of the art pretrained word embedding model
# Couldn't test locally - ran out of memory. May be worthwhile testing on the cluster
#word_embeddings = (BertEmbeddings.pretrained('bert_base_cased', lang=lang)
#    .setInputCols(['lemmas'])
#    .setOutputCols('embeddings')
#    )
#word_embeddings = (ElmoEmbeddings.pretrained('elmo', lang=lang)
#        .setInputCols(['lemmas'])
#        .setOutputCols('embeddings')
#    )

