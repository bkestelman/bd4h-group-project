### Includes word2vec and other word embeddings
from pyspark.ml import Pipeline
from pyspark.ml.feature import Word2Vec

import nlp_preprocessing_tools as nlp
from nlp_preprocessing_tools import DocAssembler, RawTokenizer, SeparatePuncTokenizer, Lemmatizer, Finisher, SpellChecker
from conf.hyperparameters import word2vec_params

def BasicWord2Vec(inputCol, outputCol):
    word2vec = Word2Vec(inputCol=inputCol, outputCol=outputCol, minCount=0, **word2vec_params) 
    pipe = Pipeline(stages=[word2vec])
    return pipe 

#def GloveWordEmbeddings(inputCol, outputCol):
#    """
#    DEPRECATED
#    There are some complications making GloveWordEmbeddings work with the new code structure
#    where we clean and tokenize data as part of preprocessing before passing it to ML models.
#    Since we test glove embeddings in the pytorch code anyway, this function is not so 
#    important. Thus, at this point it makes sense to just not use it.
#    """
##    tokenizer = RawTokenizer(inputCol=inputCol, outputCol='TOKENS') 
#    word_embeddings = nlp.GloveWordEmbeddings(inputCol=inputCol, outputCol=outputCol)
#    pipe = Pipeline(stages=[
##        tokenizer,
#        word_embeddings,
#        ])
#    return pipe

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

