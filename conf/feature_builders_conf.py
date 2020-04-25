# Different pipelines now have the option to choose a specific column as input
# Currently the choice is between using the raw text (actually almost raw - has 
# numbers removed; see etl.py) and using further cleaned and spell checked 
# tokenized text. 
# 'TEXT' (raw text) is the default
# 'TOKENS' is the cleaned and tokenized text
pipeline_input_cols = {
    'BasicWord2Vec': 'TOKENS'
}
