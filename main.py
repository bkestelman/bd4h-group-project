"""Loads ADMISSIONS and NOTEEVENTS from S3"""
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lag, lead, datediff
from pyspark.sql.window import Window

import config
from bag_of_words import BagOfWords 
from word2vec import BasicWord2Vec

from nlp_preprocessing_tools import NoPuncTokenizer, StopWordsRemover
from pyspark.ml.feature import CountVectorizer, RegexTokenizer

sc = SparkContext(master='yarn', appName='LoadData') #TODO: make a better app name xD
### Add local modules here (TODO: move this to a separate config file)
sc.addPyFile('bag_of_words.py')
sc.addPyFile('nlp_preprocessing_tools.py')
sc.addPyFile('word2vec.py')
spark = SparkSession.builder.appName("LoadData").getOrCreate()

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

def add_next_admission(admissions):
    """
    Adds the columns NEXT_ADMITTIME and NEXT_ADMISSION_TYPE to the given admissions df
    Returns df with columns SUBJECT_ID, ADMITTIME, NEXT_ADMITTIME, NEXT_ADMISSION_TYPE (may be useful to keep more columns in future)
    """
    # Example of calculating value for column based on the previous row: https://stackoverflow.com/a/34296063/5486210
    w = Window.partitionBy('SUBJECT_ID').orderBy('ADMITTIME')
    admissions = admissions.select('SUBJECT_ID', 'ADMITTIME', 
            lead('ADMITTIME').over(w).alias('NEXT_ADMITTIME'),
            lead('ADMISSION_TYPE').over(w).alias('NEXT_ADMISSION_TYPE'),
            )
    return admissions

def label_readmissions(admissions, days):
    """
    Adds a label to each admission indicating whether the next admission is a readmission within 'days' days. 
    1=next admission is readmission, 0=no next admission, or next admission after 'days' days
    @param admissions : should contain ADMITTIME and NEXT_ADMITTIME columns (NEXT_ADMITTIME may be null if there was no previous admission)
    @param days : the max number of days between admissions during which the latter is counted as a readmission
    @return readmissions df
    """
    readmissions = admissions.withColumn('LABEL',
        when(col('NEXT_ADMITTIME').isNull(), 0) # if there is no next admission, 0
        .when(col('NEXT_ADMISSION_TYPE') != 'EMERGENCY', 0) # if next admission is not an emergency, don't count as an unplanned readmission
        .when(datediff(col('NEXT_ADMITTIME'), col('ADMITTIME')) < days, 1) # if next admission date < 'days' days after this admission, 1
        .otherwise(0) # otherwise (next admission more than 'days' days after this admission), 0
        )
    return readmissions 

if __name__ == '__main__':
    admissions, noteevents = load_data()
    admissions.printSchema()
    noteevents.printSchema()
    #admissions.show()
    #noteevents.show()
    sample_noteevents = noteevents.limit(1000)
#    next_admit = add_next_admission(admissions)
#    next_admit.show()
#    readmissions = label_readmissions(next_admit, days=30)
#    readmissions.where(~col('NEXT_ADMITTIME').isNull()).show()
#
#    readmission_count = readmissions.where(col('LABEL') == 1).count()
#    total_admission_count = admissions.count() 
#    print('***Readmissions count:', readmission_count) 
#    print('***Total admissions count:', total_admission_count)
#    print('***Readmission rate:', float(readmission_count) / total_admission_count) 

#    noteevents = bag_of_words.tokenize(noteevents, 'TEXT')
#    noteevents.select('SUBJECT_ID', 'TEXT', 'TEXT_TOKENIZED').show()
#    noteevents = bag_of_words.add_bag_of_words(noteevents, 'TEXT_TOKENIZED')
#    noteevents.select('SUBJECT_ID', 'TEXT', 'TEXT_TOKENIZED', 'FEATURES').show()

    bagOfWords = BagOfWords(inputCol='TEXT', outputCol='FEATURES').fit(sample_noteevents)
    bagOfWordsResults = bagOfWords.transform(sample_noteevents)
    print('***Bag of Words***')
    bagOfWordsResults.show()

    vectorSize=50
    word2vec = BasicWord2Vec(inputCol='TEXT', outputCol='FEATURES', minCount=0, vectorSize=vectorSize)
    word2vecModel = word2vec.fit(sample_noteevents)
    word2vecResults = word2vecModel.transform(sample_noteevents)
    print('***Word2Vec***')
    word2vecResults.show()
    print('***Word Vectors***')
    word2vecModel.stages[2].getVectors().show(truncate=False)
