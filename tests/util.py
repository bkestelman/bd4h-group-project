from pyspark.sql import SparkSession
from functools import lru_cache

#https://medium.com/@mrpowers/creating-a-pyspark-project-with-pytest-pyenv-and-egg-files-d2709eb1604c
@lru_cache(maxsize=None)
def get_spark(): 
    '''Helper function for craeting spark session'''
    return (SparkSession.builder
                .master("local")
                .appName("ptests")
                .getOrCreate())