from utils import timeit
from pyspark.ml import PipelineModel

from bag_of_words import BagOfWords
from word2vec import BasicWord2Vec
from hyperparameters import fit_limits 
import config
import hdfs_utils

@timeit
def add_features(dataset, features_builder, save_path=None):
    """Adds a features column to dataset
    @param dataset : the dataset to transform
    @param features_builder : a Spark Pipeline containing the stages required to build the features
    @param save_path : save the fitted PipelineModel. Load from this path if it already exists
    @return dataset with added column 'FEATURES'
    """
    print('Adding features using', features_builder.__name__)

    if features_builder.__name__ == 'BasicWord2Vec': #TODO: ideally, model-specific functionality should not be in this function, but not sure where to put this
        fit_dataset = dataset.limit(fit_limits['word2vec']) # Word2Vec gets much slower as the dataset grows, so we can only use part of it to create the word vectors
    else:
        fit_dataset = dataset

    if save_path is not None and hdfs_utils.file_exists(save_path):
        print('Loading saved model from', save_path)
        pipelineModel = PipelineModel.load(save_path)
    else:
        pipelineModel = features_builder(inputCol='TEXT', outputCol='FEATURES').fit(fit_dataset)

    if save_path is not None and not hdfs_utils.file_exists(save_path):
        print('Saving model to', save_path)
        pipelineModel.save(save_path)

    dataset_w_features = pipelineModel.transform(dataset)
    #print(dataset_w_features.select('FEATURES').first())

    if config.debug_print:
        print('tokenizer')
        text_tokenized.select('SUBJECT_ID', 'TEXT', 'TOKENS').show()

        print('features')
        dataset_w_features.select('SUBJECT_ID', 'TEXT', 'TOKENS', 'FEATURES').show()

    return dataset_w_features