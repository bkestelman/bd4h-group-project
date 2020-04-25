"""ETL Preprocessing (Label Extraction)"""
from pyspark.sql.functions import col, when, lag, lead, datediff, concat_ws, collect_list, count, lower, regexp_replace
from pyspark.sql.window import Window
from pyspark.ml import Pipeline

from utils.utils import timeit
import conf.config as config

from nlp_preprocessing_tools import SeparatePuncTokenizer, DocAssembler, Finisher, SpellChecker

@timeit
def preprocess_data(admissions, noteevents):
    # add_next_admission date and days between admissions for each admission record
    next_admit = add_next_admission(admissions)
    # filter noteevents to only look for discharge summary
    noteevents_discharge = noteevents.where(col('CATEGORY') == 'Discharge summary')

    # concatenate discharge summaries, if multiple found, for a given SUBJECT_ID, HADM_ID
    # noteevents_discharge cols: SUBJECT_ID, HADM_ID, TEXT
    # https://stackoverflow.com/questions/41788919/concatenating-string-by-rows-in-pyspark
    noteevents_discharge = noteevents_discharge.\
        groupBy('SUBJECT_ID', 'HADM_ID').\
        agg(concat_ws(',', collect_list(noteevents['TEXT'])).alias('TEXT'))

    # merge df: next_admit & noteevents_discharge (inner join)
    # https://stackoverflow.com/questions/33745964/how-to-join-on-multiple-columns-in-pyspark
    dataset = next_admit.join(noteevents_discharge, ['SUBJECT_ID', 'HADM_ID'])

    # keep what we need
    dataset = dataset['SUBJECT_ID', 'HADM_ID', 'NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE', 'NEXT_DAYS_ADMIT', 'TEXT']

    # label dataset
    dataset_labeled = label_readmissions(dataset, days=30)
    # dataset_labeled.where(~col('NEXT_ADMITTIME').isNull()).show()

    ### Extra cleaning, spell checking, and tokenizing
    # New approach to the code is to do all cleaning and tokenizing here
    # ML Algorithms no longer need to include NLP preprocessing as part of their pipelines
    # They should now use 'TOKENS' column as input
    # That said, algorithms which want to do tokenization differently from here can use the original raw 'TEXT' column
    # (but for the most part, we should phase that out)
    dataset_labeled = (dataset_labeled
        .withColumn('TEXT', regexp_replace(col('TEXT'), '[0-9]', '')) # remove numbers
        .withColumn('TEXT', lower(col('TEXT'))) # make all text lowercase
        )
    doc_assembler = DocAssembler(inputCol='TEXT', outputCol='DOC')
    tokenizer = SeparatePuncTokenizer(inputCol='DOC', outputCol='TOK_TEMP')
    if config.spellchecking:
        spell_checker = SpellChecker(inputCol='TOK_TEMP', outputCol='SPELL_CHECKED')
        finisher = Finisher(inputCol='SPELL_CHECKED', outputCol='TOKENS')
    else:
        finisher = Finisher(inputCol='TOK_TEMP', outputCol='TOKENS')
    nlp_preprocessing_pipeline = Pipeline(stages=[
        doc_assembler,
        tokenizer,
        spell_checker,
        finisher,
        ])
    dataset_labeled = nlp_preprocessing_pipeline.fit(dataset_labeled).transform(dataset_labeled)
    #dataset_labeled.printSchema()

    #counts = dataset_labeled.where('LABEL = 1').groupby('NEXT_DAYS_ADMIT').count()
    #counts.show()
    #counts.write.csv('counts')
    #print('wrote counts csv')

    if config.debug_print:
        # ***admissions total records:  58976
        # ***next_admit total records:  58976
        # ***noteevents total records:  2083180
        # ***noteevents TEXT column null records:  0
        # ***noteevents discharge summary records:  52726
        # ***noteevents category counts
        # +-----------------+------+
        # |         CATEGORY|counts|
        # +-----------------+------+
        # |          Consult|    98|
        # |         Pharmacy|   103|
        # | Case Management |   967|
        # |      Social Work|  2670|
        # |   Rehab Services|  5431|
        # |          General|  8301|
        # |        Nutrition|  9418|
        # |     Respiratory | 31739|
        # |             Echo| 45794|
        # |Discharge summary| 59652|
        # |       Physician |141624|
        # |              ECG|209051|
        # |          Nursing|223556|
        # |        Radiology|522279|
        # |    Nursing/other|822497|
        # +-----------------+------+
        # ***Readmissions count:  2093
        # ***Total admissions considered count:  52726
        # ***Readmission rate:  0.039695785760345936

        readmission_count = dataset_labeled.where(col('LABEL') == 1).count()
        total_admission_count = dataset.count()

        # admissions
        print('ADMISSIONS schema')
        admissions.printSchema()
        print('ADMISSIONS sample data')
        admissions.show()

        # next_admit
        print('='*150)
        print('NEXT_ADMIT schema')
        next_admit.printSchema()
        print('='*150)
        print('NEXT_ADMIT sample data')
        next_admit.show()

        # noteevents
        print('=' * 150)
        print('NOTEEVENTS schema')
        noteevents.printSchema()
        print('=' * 150)
        print('NOTEVENTS sample data')
        noteevents.show()

        print('***admissions total records: ', admissions.count())
        print('***next_admit total records: ', next_admit.count())
        print('***noteevents total records: ', noteevents.count())
        print('***noteevents TEXT column null records: ', noteevents.where(col('TEXT').isNull()).count())
        print('***noteevents discharge summary records: ', noteevents_discharge.count())
        print('***noteevents category counts')
        noteevents.groupBy('CATEGORY').agg(count('*').alias('counts')).orderBy('counts').show(100)

        print('***Readmissions count: ', readmission_count)
        print('***Total admissions considered count: ', total_admission_count)
        print('***Readmission rate: ', float(readmission_count) / total_admission_count)

    return dataset_labeled

@timeit
def add_next_admission(admissions):
    """
    Adds the columns NEXT_ADMITTIME and NEXT_ADMISSION_TYPE to the given admissions df
    Returns df with columns SUBJECT_ID, ADMITTIME, NEXT_ADMITTIME, NEXT_ADMISSION_TYPE (may be useful to keep more columns in future)
    """
    # Example of calculating value for column based on the previous row: https://stackoverflow.com/a/34296063/5486210
    w = Window.partitionBy('SUBJECT_ID').orderBy('ADMITTIME')
    admissions = admissions.select('SUBJECT_ID', 'HADM_ID', 'ADMITTIME',
                                   lead('ADMITTIME').over(w).alias('NEXT_ADMITTIME'),
                                   lead('ADMISSION_TYPE').over(w).alias('NEXT_ADMISSION_TYPE'),
                                   ).withColumn('NEXT_DAYS_ADMIT', datediff(col('NEXT_ADMITTIME'), col('ADMITTIME')))

    return admissions

@timeit
def label_readmissions(admissions, days):
    """
    Adds a label to each admission indicating whether the next admission is a readmission within 'days' days. 
    1=next admission is readmission, 0=no next admission, or next admission after 'days' days
    @param admissions : should contain ADMITTIME and NEXT_ADMITTIME columns (NEXT_ADMITTIME may be null if there was no previous admission)
    @param days : the max number of days between admissions during which the latter is counted as a readmission
    @return readmissions df
    """
    readmissions = admissions.withColumn('LABEL',
         when(col('NEXT_ADMITTIME').isNull(), 0)  # if there is no next admission, 0
        .when(col('NEXT_ADMISSION_TYPE') != 'EMERGENCY', 0)  # if next admission is not an emergency, don't count as an unplanned readmission
        .when(col('NEXT_DAYS_ADMIT') < days, 1)  # if next admission date < 'days' days after this admission, 1
        .otherwise(0)  # otherwise (next admission more than 'days' days after this admission), 0
        )
    return readmissions 
