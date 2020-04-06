from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec
from utils import timeit

from nlp_preprocessing_tools import RawTokenizer, Lemmatizer, Finisher
from hyperparameters import word2vec_params

@timeit
def BasicWord2Vec(inputCol, outputCol):
    tokenizer = RawTokenizer(inputCol=inputCol, outputCol='TOKENS') 
    finisher = Finisher(inputCol='TOKENS', outputCol='FINISHED_TOKENS') 
    word2vec = Word2Vec(inputCol='FINISHED_TOKENS', outputCol=outputCol, minCount=0, vectorSize=word2vec_params['vectorSize']) #TODO: move vectorSize to config
    pipe = Pipeline(stages=[
        tokenizer,
        finisher,
        word2vec,
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

