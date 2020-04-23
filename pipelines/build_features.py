from pyspark.ml import PipelineModel

from .bag_of_words import BagOfWords
from .word2vec import BasicWord2Vec
from utils.utils import timeit
from conf.hyperparameters import fit_limits 
from export_data import write_vectors_csv
import conf.config as config
import utils.hdfs_utils as hdfs_utils

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
        if features_builder.__name__ == 'BasicWord2Vec': #TODO: ideally, model-specific functionality should not be in this function, but not sure where to put this
            write_vectors_csv(pipelineModel.stages[-1].getVectors(), save_path + '_vectors') # besides saving the full pipeline model, we also want the word vectors as their own csv for pytorch
            print('Wrote vectors to ', save_path + '_vectors')

    dataset_w_features = pipelineModel.transform(dataset)
    #print(dataset_w_features.select('FEATURES').first())


    return dataset_w_features
