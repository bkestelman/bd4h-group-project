from utils import timeit

from bag_of_words import BagOfWords
from word2vec import BasicWord2Vec
from hyperparameters import fit_limits 
import config

@timeit
def add_features(dataset, features_builder):
    """Adds a features column to dataset
    @param dataset : the dataset to transform
    @param features_builder : a Spark Pipeline containing the stages required to build the features
    @return dataset with added column 'FEATURES'
    """
    print('Adding features using', features_builder.__name__)

    if features_builder.__name__ == 'BasicWord2Vec':
        fit_dataset = dataset.limit(fit_limits['word2vec']) # Word2Vec gets much slower as the dataset grows, so we can only use part of it to create the word vectors
    else:
        fit_dataset = dataset

    pipelineModel = features_builder(inputCol='TEXT', outputCol='FEATURES').fit(fit_dataset)
    dataset_w_features = pipelineModel.transform(dataset)

    if config.debug_print:
        print('tokenizer')
        text_tokenized.select('SUBJECT_ID', 'TEXT', 'TOKENS').show()

        print('features')
        dataset_w_features.select('SUBJECT_ID', 'TEXT', 'TOKENS', 'FEATURES').show()

    return dataset_w_features
