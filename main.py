"""Loads ADMISSIONS and NOTEEVENTS from S3"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, col, datediff
from pyspark.sql.window import Window

import config

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
    )

    return admissions, noteevents

def add_prev_admittime(admissions):
    """
    Adds the column PREV_ADMITTIME to the given admissions df
    Returns df with columns SUBJECT_ID, ADMITTIME, PREV_ADMITTIME (may be useful to keep more columns)
    Decided to calculate PREV_ADMITTIME instead of NEXT_ADMITTIME (as in the blog) because this way each row directly represents a readmission
    """
    # Example of calculating value for column based on the previous row: https://stackoverflow.com/a/34296063/5486210
    w = Window.partitionBy('SUBJECT_ID').orderBy('ADMITTIME')
    admissions = admissions.select('SUBJECT_ID', 'ADMITTIME', lag('ADMITTIME').over(w).alias('PREV_ADMITTIME'))
    return admissions

def count_readmissions(admissions, days):
    """
    Counts the number of readmissions in the dataset 
    @param admissions : should contain ADMITTIME and PREV_ADMITTIME columns (PREV_ADMITTIME may be null if there was no previous admission)
    @param days : the max number of days between admissions during which the latter is counted as a readmission
    """
    admissions = admissions.dropna()
    admissions = admissions.withColumn('days_between_admissions', datediff(col('ADMITTIME'), col('PREV_ADMITTIME')))
    count = admissions.where(col('days_between_admissions') < days).count()
    return count

if __name__ == '__main__':
    admissions, noteevents = load_data()
    admissions.printSchema()
    admissions.show()
    noteevents.show()
    prev_admit = add_prev_admittime(admissions)
    prev_admit.show()
    readmission_count = count_readmissions(prev_admit, days=30)
    total_admission_count = admissions.count() 
    total_admission_count2 = prev_admit.count() 
    print('***Readmissions count:', readmission_count)
    print('***Total admissions count:', total_admission_count, total_admission_count2)
    print('***Readmission rate:', float(readmission_count) / total_admission_count) 
