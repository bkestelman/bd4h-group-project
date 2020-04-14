"""Loads ADMISSIONS and NOTEEVENTS from S3"""
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql.functions import col, when, lag, lead, datediff, concat_ws, collect_list, count, rand
from pyspark.sql.window import Window
from utils import timeit
import os, time, shutil
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
# import matplotlib.pyplot as plt
# TODO: does not make it to AWS pyspark, even though installed via requirements.txt. pip vs pip3 problem?

from bag_of_words import BagOfWords 
from word2vec import BasicWord2Vec, GloveWordEmbeddings, Words2Matrix, Words2MatrixTransformer
from tf_idf import TfIdf
from build_features import add_features, prepare_features_builder

import config
import helper_udfs
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from models import Model


SEED = 40**3

#from nlp_preprocessing_tools import NoPuncTokenizer, StopWordsRemover
#from pyspark.ml.feature import CountVectorizer, RegexTokenizer

sc = SparkContext(master=config.spark_master, appName=config.spark_app_name)

spark = SparkSession.builder.appName(config.spark_app_name).getOrCreate()
sc.setLogLevel('WARN')

# I think the driver memory will only take effect when set from spark-submit
# spark = SparkSession.builder.appName(config.spark_app_name).config('spark.driver.memory', '12g').getOrCreate()
# print(sc.getConf().getAll())

# Add local modules here
for sc_py_file in config.sc_py_files:
    sc.addPyFile(sc_py_file)

# Register UDF's
spark.udf.register('list_to_vector_udf', helper_udfs.list_to_vector, VectorUDT())

@timeit
def load_data():

    admissions = (spark.read.format('csv')
        .option('header', 'true')
        .load(config.path_to_mimic + 'ADMISSIONS.csv.gz')
    )
    admissions = admissions.withColumn('ADMITTIME', col('ADMITTIME').cast('date')) # cast ADMITTIME from string to date

    noteevents = (spark.read.format('csv')
        .option('header', 'true')
        .option('multiline', 'true') # The text for the discharge summaries spans multiple lines. This option prevents Spark from treating each line as a new row
        .option('escape', '"') # The discharge summaries use a double quote as the escape character (i.e. quotes within a discharge summary are escaped as "" instead of \")
        .load(config.path_to_mimic + 'NOTEEVENTS.csv.gz')
    ).repartition(20) # Spark reads .gz file as single partition, so we need to split to more partitions before we do any processing on this dataframe. Still playing around to find the optimal number, but 10 at least works (without repartition, was hitting memory errors during text tokenization).
    return admissions, noteevents

@timeit
def preprocess_data(admissions, noteevents):

    # add_next_admission date and days between admissions for each admission record
    next_admit = add_next_admission(admissions)

    # filter noteevents to only look for discharge summary
    noteevents_discharge = noteevents.where(col('CATEGORY') == 'Discharge summary')

    # concatenate discharge summaries, if multiple found, for a given SUBJECT_ID, HADM_ID
    # noteevents_discharge cols: SUBJECT_ID, HADM_ID, TEXT
    # https://stackoverflow.com/questions/41788919/concatenating-string-by-rows-in-pyspark
    noteevents_discharge = noteevents_discharge.\
        groupBy('SUBJECT_ID', 'HADM_ID').\
        agg(concat_ws(',', collect_list(noteevents['TEXT'])).alias('TEXT'))

    # merge df: next_admit & noteevents_discharge (inner join)
    # https://stackoverflow.com/questions/33745964/how-to-join-on-multiple-columns-in-pyspark
    dataset = next_admit.join(noteevents_discharge, ['SUBJECT_ID', 'HADM_ID'])

    # keep what we need
    dataset = dataset['SUBJECT_ID', 'HADM_ID', 'NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE', 'NEXT_DAYS_ADMIT', 'TEXT']

    # label dataset
    dataset_labeled = label_readmissions(dataset, days=30)
    # dataset_labeled.where(~col('NEXT_ADMITTIME').isNull()).show()

    if config.debug_print:
        # ***admissions total records:  58976
        # ***next_admit total records:  58976
        # ***noteevents total records:  2083180
        # ***noteevents TEXT column null records:  0
        # ***noteevents discharge summary records:  52726
        # ***noteevents category counts
        # +-----------------+------+
        # |         CATEGORY|counts|
        # +-----------------+------+
        # |          Consult|    98|
        # |         Pharmacy|   103|
        # | Case Management |   967|
        # |      Social Work|  2670|
        # |   Rehab Services|  5431|
        # |          General|  8301|
        # |        Nutrition|  9418|
        # |     Respiratory | 31739|
        # |             Echo| 45794|
        # |Discharge summary| 59652|
        # |       Physician |141624|
        # |              ECG|209051|
        # |          Nursing|223556|
        # |        Radiology|522279|
        # |    Nursing/other|822497|
        # +-----------------+------+
        # ***Readmissions count:  2093
        # ***Total admissions considered count:  52726
        # ***Readmission rate:  0.039695785760345936

        readmission_count = dataset_labeled.where(col('LABEL') == 1).count()
        total_admission_count = dataset.count()

        # admissions
        print('ADMISSIONS schema')
        admissions.printSchema()
        print('ADMISSIONS sample data')
        admissions.show()

        # next_admit
        print('='*150)
        print('NEXT_ADMIT schema')
        next_admit.printSchema()
        print('='*150)
        print('NEXT_ADMIT sample data')
        next_admit.show()

        # noteevents
        print('=' * 150)
        print('NOTEEVENTS schema')
        noteevents.printSchema()
        print('=' * 150)
        print('NOTEVENTS sample data')
        noteevents.show()

        print('***admissions total records: ', admissions.count())
        print('***next_admit total records: ', next_admit.count())
        print('***noteevents total records: ', noteevents.count())
        print('***noteevents TEXT column null records: ', noteevents.where(col('TEXT').isNull()).count())
        print('***noteevents discharge summary records: ', noteevents_discharge.count())
        print('***noteevents category counts')
        noteevents.groupBy('CATEGORY').agg(count('*').alias('counts')).orderBy('counts').show(100)

        print('***Readmissions count: ', readmission_count)
        print('***Total admissions considered count: ', total_admission_count)
        print('***Readmission rate: ', float(readmission_count) / total_admission_count)

    return dataset_labeled

@timeit
def add_next_admission(admissions):
    """
    Adds the columns NEXT_ADMITTIME and NEXT_ADMISSION_TYPE to the given admissions df
    Returns df with columns SUBJECT_ID, ADMITTIME, NEXT_ADMITTIME, NEXT_ADMISSION_TYPE (may be useful to keep more columns in future)
    """
    # Example of calculating value for column based on the previous row: https://stackoverflow.com/a/34296063/5486210
    w = Window.partitionBy('SUBJECT_ID').orderBy('ADMITTIME')
    admissions = admissions.select('SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                                   lead('ADMITTIME').over(w).alias('NEXT_ADMITTIME'),
                                   lead('ADMISSION_TYPE').over(w).alias('NEXT_ADMISSION_TYPE'),
                                   ).withColumn('NEXT_DAYS_ADMIT', datediff(col('NEXT_ADMITTIME'), col('ADMITTIME')))

    return admissions

@timeit
def label_readmissions(admissions, days):
    """
    Adds a label to each admission indicating whether the next admission is a readmission within 'days' days. 
    1=next admission is readmission, 0=no next admission, or next admission after 'days' days
    @param admissions : should contain ADMITTIME and NEXT_ADMITTIME columns (NEXT_ADMITTIME may be null if there was no previous admission)
    @param days : the max number of days between admissions during which the latter is counted as a readmission
    @return readmissions df
    """
    readmissions = admissions.withColumn('LABEL',
         when(col('NEXT_ADMITTIME').isNull(), 0)  # if there is no next admission, 0
        .when(col('NEXT_ADMISSION_TYPE') != 'EMERGENCY', 0)  # if next admission is not an emergency, don't count as an unplanned readmission
        .when(col('NEXT_DAYS_ADMIT') < days, 1)  # if next admission date < 'days' days after this admission, 1
        .otherwise(0)  # otherwise (next admission more than 'days' days after this admission), 0
        )
    return readmissions 

# @timeit
# def do_lr(train, test):
#     # https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c
#
#     if lr_class_weights:  # balance class weights
#         weight_col = 'classWeights'
#         neg_labels = train.where(col('LABEL') == 0).count()
#         train_count = train.count()
#         balancing_ratio = neg_labels / train_count
#         print('balancing ratio: {:.2f}'.format(balancing_ratio))
#         train = train.withColumn(weight_col, when(col('LABEL') == 1, balancing_ratio).otherwise(1 - balancing_ratio))
#         lr = LogisticRegression(featuresCol='FEATURES', labelCol='LABEL', maxIter=lr_iterations, weightCol=weight_col)
#         # train.select(weight_col).show(5)
#     else:
#         lr = LogisticRegression(featuresCol='FEATURES', labelCol='LABEL', maxIter=5)
#
#     evaluator = BinaryClassificationEvaluator().setLabelCol('LABEL')
#     param_grid = ParamGridBuilder()\
#         .addGrid(lr.fitIntercept, [False, True]) \
#         .addGrid(lr.regParam, [0.01, 0.5, 2.0]) \
#         .build()
#         # .addGrid(lr.aggregationDepth, [2, 5, 10])\
#         # .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
#         # .build()
#
#     model = lr.fit(train)
#     # Create 3-fold CrossValidator
#     # cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
#     # model = cv.fit(train)
#
#     predict_train = model.transform(train)
#     predictions = model.transform(test)
#
#     # evaluator = BinaryClassificationEvaluator().setLabelCol('LABEL')
#     print('Train Area Under ROC', evaluator.evaluate(predict_train))
#     print('Test Area Under ROC', evaluator.evaluate(predictions))


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


if __name__ == '__main__':

    t_start = time.time()

    admissions, noteevents = load_data()

    labeled_dataset = preprocess_data(admissions, noteevents)

    if config.sample_run:  # run on a smaller sized dataset
        labeled_dataset = labeled_dataset.limit(config.sample_size)

    elif config.balance_dataset_negatives:  # sub sample negatives according to ratio
        pos_labels = labeled_dataset.where(col('LABEL') == 1)
        neg_labels = labeled_dataset.where(col('LABEL') == 0)
        neg_percent_subsample = int(pos_labels.count() * config.balance_dataset_ratio) / neg_labels.count()
        neg_labels = neg_labels.sample(withReplacement=False, fraction=neg_percent_subsample, seed=SEED)
        labeled_dataset = pos_labels.union(neg_labels).orderBy(rand(seed=SEED))

        # df = spark.createDataFrame([{"a": "x", "b": "y", "c": "3"}])

    labeled_dataset.cache() # HUGE performance improvement by caching!
    #labeled_dataset.persist(StorageLevel.MEMORY_AND_DISK) 
    #labeled_dataset.groupby('HADM_ID').count().where(col('count') != 1).show() # checked that after preprocessing, HADM_ID is unique for all rows

    train_ids, test_ids = labeled_dataset.select(col('HADM_ID').alias('HADM_ID_SPLIT')).randomSplit([0.8, 0.2], seed=SEED) # the alias is to make joining to features smoother

    labeled_dataset_count = labeled_dataset.count()
    pos_count = labeled_dataset.where(col('LABEL') == 1).count()
    neg_count = labeled_dataset.where(col('LABEL') == 0).count()

    print('dataset count:{} positive:{} negative:{}'.format(labeled_dataset_count, pos_count, neg_count))
    print('splitting dataset into train & test')

    features_builders = [
        #TfIdf,
        #BagOfWords,
        BasicWord2Vec,
        #GloveWordEmbeddings,
        Words2Matrix,
        ]

    for features_builder in features_builders:
        f_start = time.time()
        print('-'*50)
        print('running feature builder: {}'.format(features_builder.__name__))
        save_model_path = config.save_model_paths.get(features_builder.__name__)
        extra_args = prepare_features_builder(features_builder)

        dataset_w_features = (add_features(labeled_dataset, features_builder, save_model_path, **extra_args)
            .select('HADM_ID', 'FEATURES', 'LABEL')
        )

        if features_builder.__name__ == 'Words2Matrix':
            dataset_w_features.printSchema()
            dataset_w_features.show()
            continue # for now, just testing if we get the matrix for each document, so skip logistic-regression

        dataset_w_features.cache()
        #dataset_w_features.persist(StorageLevel.MEMORY_AND_DISK)

        # logistic regression: mpatel364 - memory errors when running logistic regression locally
        train = train_ids.join(dataset_w_features, train_ids['HADM_ID_SPLIT'] == dataset_w_features['HADM_ID'])
        test = test_ids.join(dataset_w_features, test_ids['HADM_ID_SPLIT'] == dataset_w_features['HADM_ID'])
        print('starting logistic regression...')
        lr_start = time.time()
        # do_lr(train, test)
        ml_model = Model(algorithm='LogisticRegression', train=train, test=test, features_col='FEATURES',
                         label_col='LABEL')

        ml_model.train_model()
        ml_model.evaluate()
        print('logistic regression completed in {:.2f} minutes'.format((time.time()-lr_start) / 60.))
        print('feature run completed in {:.2f} minutes'.format((time.time()-f_start) / 60.))

    print('total run completed in {:.2f} minutes'.format((time.time()-t_start)/60.))
    print('-'*50)

        # print('training dataset count: ', train.count())
        # print('test dataset count: ', test.count())
        # training dataset count:  42178
        # test dataset count:  10548

