import os, shutil
from pyspark.sql.functions import col

import conf.config as config

def dump_dataset(dataset_w_features, features_col='FEATURES'):
    # for debugging, in the event we want see what's fed to the ML algorithms
    if config.dump_dataset:
        csv_dump_dir = 'dataset_input'
        if os.path.exists(csv_dump_dir) and os.path.isdir(csv_dump_dir):  # delete csv dir if exists
            shutil.rmtree(csv_dump_dir)

        csv_df = dataset_w_features.withColumn('FEATURES_AS_STRING', col('FEATURES').cast('string'))
        csv_df = csv_df.drop('FEATURES').drop('TEXT_TOKENIZED_STRING').drop('RAW_TOKENS').drop('TEXT')
        csv_df = csv_df['HADM_ID', 'LABEL',  'FEATURES_AS_STRING']
        csv_df.printSchema()
        csv_df.repartition(1).write.csv(config.dump_dataset, header=True)

