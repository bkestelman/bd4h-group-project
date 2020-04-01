"""Loads ADMISSIONS and NOTEEVENTS from S3"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, col
from pyspark.sql.window import Window

import config

spark = SparkSession.builder.appName("LoadData").getOrCreate()

def load_data():
    admissions = (spark.read.format('csv')
        .option('header', 'true')
        .load(config.path_to_mimic + 'ADMISSIONS.csv.gz')
    )

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

if __name__ == '__main__':
    admissions, noteevents = load_data()
    admissions.show()
    noteevents.show()
    prev_admit = add_prev_admittime(admissions)
    prev_admit.show()
