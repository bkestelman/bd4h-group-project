import time

import conf.config as config
from .bag_of_words import BagOfWords 
from .word2vec import BasicWord2Vec
from .tf_idf import TfIdf
from .build_features import add_features
from models import Model

def run_spark_pipelines(labeled_dataset, train_ids, test_ids):
    features_builders = [
        TfIdf,
        BagOfWords,
        BasicWord2Vec,
        ]

    algorithms = [
        'LinearSVM',
        'LogisticRegression'
    ]

    for features_builder in features_builders:
        f_start = time.time()
        print('-'*50)
        print('running feature builder: {}'.format(features_builder.__name__))
        save_model_path = config.save_model_paths.get(features_builder.__name__)
        dataset_w_features = add_features(labeled_dataset, features_builder, save_model_path)\
            .select('HADM_ID', 'FEATURES', 'LABEL')

        print('feature run completed in {:.2f} minutes'.format((time.time()-f_start)/60.))
        dataset_w_features.cache()
        #dataset_w_features.persist(StorageLevel.MEMORY_AND_DISK)

        train = train_ids.join(dataset_w_features, train_ids['HADM_ID_SPLIT'] == dataset_w_features['HADM_ID'])
        test = test_ids.join(dataset_w_features, test_ids['HADM_ID_SPLIT'] == dataset_w_features['HADM_ID'])

        for algorithm in algorithms:
            print('starting algorithm: {}'.format(algorithm))
            train_start = time.time()
            ml_model = Model(algorithm=algorithm, train=train, test=test, features_col='FEATURES',
                             cv=False, label_col='LABEL')

            ml_model.train_model()
            ml_model.evaluate(save_fig=False)
            print('algorithm: {} completed in {:.2f} minutes'.format(algorithm, (time.time()-train_start)/60.))

