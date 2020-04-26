# Different pipelines now have the option to choose a specific column as input
# Currently the choice is between using the raw text (actually almost raw - has 
# numbers removed; see etl.py) and using further cleaned and spell checked 
# tokenized text. 
# The column names for the above choice are 'TEXT' and 'TOKENS'
default_pipeline_input_col = 'TOKENS'
pipeline_input_cols = {
    #'BasicWord2Vec': 'TOKENS',
}
