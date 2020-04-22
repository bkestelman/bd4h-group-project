"""
1. Ingest NOTEEVENTS and ADMISSIONS data from mimic
2. Do ETL/preprocessing to label data (1=readmission within 30 days, 0 else)
3. Export the labeled data as a csv for use by other programs (e.g. pytorch model)
4. Run SparkML/sparknlp pipelines to train and test models on the data
"""
from pyspark import StorageLevel
from pyspark.sql.functions import col
from utils.utils import timeit
import time 

# Local imports
# Trying to keep the imports organized to roughly show the order of what main is going to call
import conf.config as config
from spark_setup import setup_spark 
from ingest import load_data
from etl import preprocess_data
from sampling import get_sample
from export_data import write_labeled_readmissions_csv
from pipelines.pipelines import run_spark_pipelines

def main():
    SEED = config.SEED

    spark, sc = setup_spark()

    t_start = time.time()

    admissions, noteevents = load_data()

    labeled_dataset = preprocess_data(admissions, noteevents)

    labeled_dataset = get_sample(labeled_dataset, balance=config.balance_dataset_negatives)

    labeled_dataset.cache() # HUGE performance improvement by caching!
    #labeled_dataset.persist(StorageLevel.MEMORY_AND_DISK) 
    #labeled_dataset.groupby('HADM_ID').count().where(col('count') != 1).show() # checked that after preprocessing, HADM_ID is unique for all rows

    # Check counts to verify csv is written correctly
    #print(labeled_dataset.where('LABEL = 1').count())
    #print(labeled_dataset.where('LABEL = 0').count())
    write_labeled_readmissions_csv(labeled_dataset, config.readmissions_csv_out)
    print('Wrote labeled data to csv at ', config.readmissions_csv_out)

    print('splitting dataset into train & test')
    train_ids, test_ids = labeled_dataset.select(col('HADM_ID').alias('HADM_ID_SPLIT')).randomSplit([0.8, 0.2], seed=SEED) # the alias is to make joining to features smoother

    # Check if counts are balanced
    labeled_dataset_count = labeled_dataset.count()
    pos_count = labeled_dataset.where(col('LABEL') == 1).count()
    neg_count = labeled_dataset.where(col('LABEL') == 0).count()
    print('dataset count:{} positive:{} negative:{}'.format(labeled_dataset_count, pos_count, neg_count))

    # Run pipelines that use SparkML and sparknlp (run pytorchmain.py to run the pytorch model)
    run_spark_pipelines(labeled_dataset, train_ids, test_ids)

    print('total run completed in {:.2f} minutes'.format((time.time()-t_start)/60.))
    print('-'*50)

if __name__ == '__main__':
    main()
