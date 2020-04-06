from utils import timeit

from bag_of_words import BagOfWords
from word2vec import BasicWord2Vec
import config

#    bagOfWords = BagOfWords(inputCol='TEXT', outputCol='FEATURES').fit(sample_noteevents)
#    bagOfWordsResults = bagOfWords.transform(sample_noteevents)
#    print('***Bag of Words***')
#    bagOfWordsResults.show()
#
#    vectorSize=50
#    word2vec = BasicWord2Vec(inputCol='TEXT', outputCol='FEATURES', minCount=0, vectorSize=vectorSize)
#    word2vecModel = word2vec.fit(sample_noteevents)
#    word2vecResults = word2vecModel.transform(sample_noteevents)
#    print('***Word2Vec***')
#    word2vecResults.show()
#    print('***Word Vectors***')
#    word2vecModel.stages[2].getVectors().show(truncate=False)

@timeit
def add_features(dataset, features_builder):
    """Adds a features column to dataset
    @param dataset : the dataset to transform
    @param features_builder : a Spark Pipeline containing the stages required to build the features
    @return dataset with added column 'FEATURES'
    """

    pipelineModel = features_builder(inputCol='TEXT', outputCol='FEATURES').fit(dataset)
    dataset_w_features = pipelineModel.transform(dataset)

    if config.debug_print:
        print('tokenizer')
        text_tokenized.select('SUBJECT_ID', 'TEXT', 'TOKENS').show()

        print('features')
        dataset_w_features.select('SUBJECT_ID', 'TEXT', 'TOKENS', 'FEATURES').show()

    return dataset_w_features
