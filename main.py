"""Loads ADMISSIONS and NOTEEVENTS from S3"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lag, lead, datediff
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

def add_next_admittime(admissions):
    """
    Adds the column NEXT_ADMITTIME to the given admissions df
    Returns df with columns SUBJECT_ID, ADMITTIME, NEXT_ADMITTIME (may be useful to keep more columns)
    """
    # Example of calculating value for column based on the previous row: https://stackoverflow.com/a/34296063/5486210
    w = Window.partitionBy('SUBJECT_ID').orderBy('ADMITTIME')
    admissions = admissions.select('SUBJECT_ID', 'ADMITTIME', lead('ADMITTIME').over(w).alias('NEXT_ADMITTIME'))
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
        .when(datediff(col('NEXT_ADMITTIME'), col('ADMITTIME')) < days, 1) # if next admission date < 'days' days after this admission, 1
        .otherwise(0) # otherwise (next admission more than 'days' days after this admission), 0
    # TODO: take ADMISSION_TYPE into account
    return readmissions 

if __name__ == '__main__':
    admissions, noteevents = load_data()
    admissions.printSchema()
    admissions.show()
    noteevents.show()
    next_admit = add_next_admittime(admissions)
    next_admit.show()
    readmissions = label_readmissions(next_admit, days=30)

    readmission_count = readmissions.where(col('LABEL') == 1).count()
    total_admission_count = admissions.count() 
    print('***Readmissions count:', readmission_count) 
    print('***Total admissions count:', total_admission_count)
    print('***Readmission rate:', float(readmission_count) / total_admission_count) 
