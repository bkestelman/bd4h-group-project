from pyspark.sql.functions import regexp_replace, expr, udf, col, lit, concat_ws
from pyspark.sql.types import FloatType
import helper_udfs

def write_labeled_readmissions_csv(labeled_dataset, dirname):
    """
    @param labeled_dataset
    @param dirname : output directory name (Spark may create many .csv files in this directory, depending on the size of the data). If None, do nothing
    """
    tokens_col = 'TOKENS'
    text_col = 'PROCESSED_TEXT'
    label_col = 'LABEL'
    # create string from the processed tokens, restoring the format of the original raw text, but cleaned
    labeled_dataset = labeled_dataset.withColumn(text_col, concat_ws(' ', tokens_col)) 
    if dirname is not None:
        (labeled_dataset.select(text_col, label_col)
            .withColumn(text_col, regexp_replace(text_col, '\n', ' ')) # remove newlines from text
            .write.option('header', 'false') # header will be added in bash script after concatenating the partial files   
            .csv(dirname) # Note: this will create a directory with one or more .csv files in it
            )

def write_vectors_csv(vectors_df, path):
    """
    @param vectors_df : the result of Word2VecModel.getVectors() (schema is 'word', 'vector' where 'vector' is an array)
    """
    # Can't write an array to csv directly, so we have to create a df column for each vector dimension
    out_df = vectors_df
    vector_get_udf = udf(helper_udfs.vector_get, FloatType())
    for i in range(len(vectors_df.first()['vector'])):
        #out_df = out_df.withColumn(str(i), expr('vector[' + str(i) + ']'))
        out_df = out_df.withColumn(str(i), vector_get_udf(col('vector'), lit(i)))
    out_df.drop('vector').repartition(1).write.option('sep', ' ').csv(path)

