"""
Pull random excerpts from each summary
To use, set the path pointing to the original data below, then run this scripting and redirect the output:
python randomize_excerpts.py > readmissions_excerpts.csv
"""
import csv
from random import randint

path = 'readmissions_balance_1_1.csv'
N_EXCERPTS = 100 # number of random excerpts to pull from each row
EXCERPT_SIZE = 100 # how many chars in an excerpt (TODO: may make more sense to make this number of words)
SEP = ' ... ' # separator to insert between excerpts
QUOTCHAR = '"'
DELIM = ','
ESCAPECHAR = '\\'

HEADER = 'text,label'
print(HEADER)

with open(path) as csvfile:
    reader = csv.DictReader(csvfile, escapechar='\\')
    #next(reader) # skip header (not needed for DictReader)
    for row in reader:
        excerpts = QUOTCHAR 
        for i in range(N_EXCERPTS):
            start = randint(0, len(row['text']) - EXCERPT_SIZE - 1)
            excerpt = row['text'][start:start + EXCERPT_SIZE]
            excerpt = excerpt.replace(ESCAPECHAR, '') # remove any loose escape chars
            excerpt = excerpt.replace(QUOTCHAR, ESCAPECHAR + QUOTCHAR) # escape quotes
            excerpts += excerpt + SEP
        excerpts += QUOTCHAR + DELIM + row['label']
        print(excerpts)

