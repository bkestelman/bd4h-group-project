"""Loads ADMISSIONS and NOTEEVENTS from S3"""
from pyspark.sql import SparkSession
import config

spark = SparkSession.builder.appName("LoadData").getOrCreate()

admissions = (spark.read.format('csv')
    .option('header', 'true')
    .load(config.path_to_mimic + 'ADMISSIONS.csv.gz')
)

noteevents = (spark.read.format('csv')
    .option('header', 'true')
    .option('multiline', 'true') # the text for the discharge summaries spans multiple lines. This option prevents Spark from treating each line as a new row
    .option('escape', '"') # The discharge summaries use a double quote as the escape character (i.e. quotes within a discharge summary are escaped as "" instead of \")
    .load(config.path_to_mimic + 'NOTEEVENTS.csv.gz')
)

admissions.show()
noteevents.show()
