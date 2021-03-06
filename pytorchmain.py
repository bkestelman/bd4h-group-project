import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors
import torchtext
from torchtext.data import Dataset,TabularDataset
from torch.utils.data import DataLoader
from pycnn import CNNHelper,CNNMUL,metrics

# TODO: move these options to a separate config file
SEED = 1234
N_EPOCHS = 3
# VECTORS can be one of the prebaked options available in torchtext, or a csv (passed to a Vectors object as below):
VECTORS_CSV = '/home/data/vectors_spellchecked.csv' 
VECTORS, EMBEDDING_DIM = Vectors(VECTORS_CSV), 50
#VECTORS, EMBEDDING_DIM = "glove.6B.100d", 100
OUTPUT_DIM = 1
if OUTPUT_DIM == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss() # includes softmax, better for one-hot 2d output
MULTI_LAYER = True # If True, use multi-layered CNN

#ref:https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', batch_first = True)
LABEL = data.LabelField(dtype = torch.float)

tst_datafields = [("text", TEXT),
    ("label", LABEL) 
    ]

csv_file = '/home/data/readmissions_1_1_60days_spellchecked.csv'
csv_reader_params = {
    'escapechar': '\\', # python's default is None, but our data has escaped quotes as '\"'
    }
train_data = TabularDataset(
    path=csv_file, 
    format='csv',
    skip_header=True, 
    fields=tst_datafields,
    csv_reader_params=csv_reader_params)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))
train_data, test_data = train_data.split(random_state = random.seed(SEED))

MAX_VOCAB_SIZE = 50_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = VECTORS, 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: len(x.text),
    sort_within_batch = False,
    repeat=False,
    shuffle=True,
    device = device)

INPUT_DIM = len(TEXT.vocab)
N_FILTERS = 100
FILTER_SIZES = [1,2,3,4,5]
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNNMUL(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX, MULTI_LAYER)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
model = model.to(device)
criterion = criterion.to(device)

CNNHelper(model,eval_func=metrics.binary_auc)\
    .start(train_iterator,valid_iterator,optimizer,criterion,N_EPOCHS)\
    .test(test_iterator,criterion)\
    .plot()
