import os

spark_app_name = 'LoadData'  # TODO: make a better app name xD
debug_print = False
dump_dataset = False
sample_run = True # use only a sample of the data (applies after preprocessing, otherwise we can have a situation where very few of the samples are actually discharge summaries)
sample_size = 1000

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

#sc_py_files = ['bag_of_words.py', 'word2vec.py', 'nlp_preprocessing_tools.py']
sc_py_files = [] # for some reason everything is working now without adding py_files...? 
