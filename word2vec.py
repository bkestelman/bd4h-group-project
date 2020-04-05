from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec

from nlp_preprocessing_tools import RawTokenizer, Lemmatizer, Finisher

def BasicWord2Vec(inputCol, outputCol, **kwargs):
    tokenizer = RawTokenizer(inputCol=inputCol, outputCol='tokens') 
    finisher = Finisher(inputCol='tokens', outputCol='finished_tokens') 
    word2vec = Word2Vec(inputCol='finished_tokens', outputCol=outputCol, **kwargs)
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

