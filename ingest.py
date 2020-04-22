"""Data Ingestion"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from utils.utils import timeit
import conf.config as config

@timeit
def load_data():
    spark = SparkSession.builder.getOrCreate()

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

