"""
Contains functions to setup Spark, mainly to create SparkSession and SparkContext
Gets configuration options from config.py
"""
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT

import conf.config as config
import helper_udfs

def setup_spark():
    """
    Creates SparkSession and SparkContext and does other essential Spark setup:
    - Sets log level
    - Adds local python modules that need to be distributed (addPyFile)
    - Registers UDF's
    @return SparkSession, SparkContext
    """
    sc = SparkContext(master=config.spark_master, appName=config.spark_app_name)
    sc.setLogLevel(config.spark_log_level)

    spark = SparkSession.builder.appName(config.spark_app_name).getOrCreate()

    add_py_files(sc)
    register_udfs(spark)

    return spark, sc

def add_py_files(sc):
    """Adds local modules that need to be distributed"""
    for sc_py_file in config.sc_py_files:
        sc.addPyFile(sc_py_file)

def register_udfs(spark):
    spark.udf.register('list_to_vector_udf', helper_udfs.list_to_vector, VectorUDT())
