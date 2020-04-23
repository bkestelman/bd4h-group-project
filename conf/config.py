import os

spark_app_name = 'Predicting Readmissions'
spark_log_level = 'WARN'
debug_print = False
dump_dataset = False

plots_dir = os.path.join(os.path.dirname(__file__), 'plots')

# first preference (if set, run on a random dataset with sample_size = 3000)
sample_run = False # use only a sample of the data (applies after preprocessing, otherwise we can have a situation where very few of the samples are actually discharge summaries)
sample_size = 3000

# second preference (if set, subsample negative labels according to balance_dataset_ratio)
balance_dataset_negatives = True
balance_dataset_ratio = 1  # num negatives:num_positives

# to run locally, set env var RUN_ON_AWS=False, defaults to true
run_on_aws = True if os.getenv('RUN_ON_AWS', 'true').lower() in ['true', 'yes'] else False

if run_on_aws:
    path_to_mimic = 's3://cse6250-bucket/mimic-iii-physionet/'
    #path_to_mimic = 's3://mpatel364-bd4h/mimic-iii-clinical-database-1.4/'
    spark_master = 'yarn'
else:  # local run
    path_to_mimic = '../mimic-iii-clinical-database-1.4/'  # update this to path to MIMIC locally
    # spark_master = None
    spark_master = 'local[*]'

sc_py_files = ['helper_udfs.py'] # get import error when this is not added here. Probably only needed for local modules whose functions get distributed to workers (e.g. UDF's)

save_model_paths = {
    'BasicWord2Vec': 'BasicWord2Vec.model',
}
readmissions_csv_out = 'readmissions' # set to None if you want to skip writing the csv because you are using a csv created from a previous run

SEED = 40**3 # random seed
