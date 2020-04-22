from pyspark.sql.functions import regexp_replace

def write_labeled_readmissions_csv(labeled_dataset, dirname):
    """
    @param labeled_dataset
    @param dirname : output directory name (Spark may create many .csv files in this directory, depending on the size of the data). If None, do nothing
    """
    if dirname is not None:
        (labeled_dataset.select('TEXT', 'LABEL')
            .withColumn('TEXT', regexp_replace('TEXT', '\n', ' ')) # remove newlines from text
            #.withColumnRenamed('TEXT', 'text') # rename for consistency with pytorch schema
            #.withColumnRenamed('LABEL', 'label')
            #.write.option('header', 'true') # include header for consistency with pytorch
            .write.option('header', 'false') # header will be added in bash script after concatenating the partial files   
            .csv(dirname) # Note: this will create a directory with one or more .csv files in it
            )
