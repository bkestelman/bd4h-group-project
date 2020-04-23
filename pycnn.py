import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from torchtext import data
from torchtext import datasets
import os
import torchtext
from torchtext.data import Dataset,TabularDataset
from torch.utils.data import DataLoader
import string
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score

class Utility(object):
    @staticmethod
    def string_cleaner(in_filename,out_filename):
        '''Clean out punctuations from sentences'''
        
        df =pd.read_csv(in_filename,usecols=['text','label'])
        df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df.to_csv(out_filename, index=False,header=['text','label'])

class MovieDataset(Dataset):
    def __init__(self, path):
        self.samples = []

        for filename in os.listdir(path):
            if 'ascii' in filename:
                label = 'pos' if 'pos' in filename else 'neg'
                f_path = os.path.join(path, filename)
                with open(f_path, 'r') as sent_file:
                    for text in sent_file.read().splitlines():
                        self.samples.append((text,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def save(self,filename,headers):
        dataloader = DataLoader(self, batch_size=1, shuffle=True)
        data = [(str(batch[0][0]),str(batch[1][0])) for _, batch in enumerate(dataloader)]
        df = pd.DataFrame(data, columns=[headers[0],headers[1]])
        df.to_csv(filename, index=False)

class CNNMUL(nn.Module):
    ''' Define network architecture and forward path. '''
    def __init__(self, vocab_size, 
                 vector_size, n_filters, 
                 filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        self.output_dim = output_dim
        # Create word embeddings from the input words     
        self.embedding = nn.Embedding(vocab_size, vector_size, 
                                      padding_idx = pad_idx)
        
        # Specify convolutions with filters of different sizes (fs)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, vector_size)) 
                                    for fs in filter_sizes])
        
        # Add a fully connected layer for final predicitons
        self.linear = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        # Drop some of the nodes to increase robustness in training
        self.dropout = nn.Dropout(dropout)
        
        
        
    def forward(self, text):
        '''Forward path of the network.'''       
        # Get word embeddings and formt them for convolutions
        embedded = self.embedding(text).unsqueeze(1)
        
        # Perform convolutions and apply activation functions
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        # Pooling layer to reduce dimensionality    
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Dropout layer
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.linear(cat)

class metrics(object):
    metric_map={
        'binary_accuracy':'Accuracy',
        'binary_roc_auc_score': 'ROC AUC',
        'binary_auc': 'AUC'
    }
    @staticmethod
    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        correct = (preds == y).float()
        acc = correct.sum() / len(correct)
        return acc.item()

    @staticmethod
    def binary_roc_auc_score(preds, y):
        """
        Returns ROC AUC: When Only one class present in y, ROC AUC is underfined
        """
        Y = y.cpu().detach().numpy()
        pred = preds.cpu().detach().numpy()
        return roc_auc_score(Y,pred)

    @staticmethod
    def binary_auc(preds, y):
        """
        Returns AUC: When Only one class present in y, AUC is underfined
        """
        Y = y.cpu().detach().numpy()
        pred = preds.cpu().detach().numpy()
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y, pred)
        return auc(false_positive_rate, true_positive_rate)

class CNNHelper(object):

    def __init__(self,model,eval_func):
        super().__init__()
        self.model = model
        self.eval_function = eval_func
        self.metric_name = f'{metrics.metric_map[self.eval_function.__name__]}'

    def count_parameters(self):
        '''Count number of parameters  in model'''

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def skip_batch(self,label):
        '''Batches with all labels same should be skipped, as can't be used for AUC calculation'''
        y = label.cpu().detach().numpy()
        return np.all(y == y[0])

    def cast_label(self, label, criterion):
        '''Some criterion functions require label to be a certain type'''
        if type(criterion).__name__ == 'CrossEntropyLoss':
            return label.long()
        else:
            return label

    def train(self, iterator, optimizer, criterion):
        ''' Training function'''

        epoch_loss = 0
        epoch_acc = 0   
        self.model.train()
        from tqdm import tqdm 

        skipped = 0
        for batch in tqdm(iterator):
            
            if self.skip_batch(batch.label):
                skipped += 1
                continue

            optimizer.zero_grad()           
            predictions = self.model(batch.text).squeeze(1)          
            batch.label = self.cast_label(batch.label, criterion)
            loss = criterion(predictions, batch.label)    
            if self.model.output_dim == 1:
                #round predictions to the closest integer
                predictions = torch.round(torch.sigmoid(predictions))
            else:
                _, predictions = torch.max(predictions, 1) # get max index from one-hot predictions 
            acc = self.eval_function(predictions, batch.label)
            loss.backward()     
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
        
        length = (len(iterator) -skipped)
        return epoch_loss / length, epoch_acc / length

    def evaluate(self, iterator, criterion):
        ''' Evaluation funtion'''

        epoch_loss = 0
        epoch_acc = 0       
        self.model.eval()
        
        with torch.no_grad():
            skipped = 0
            for batch in iterator:

                if self.skip_batch(batch.label):
                    skipped += 1
                    continue

                predictions = self.model(batch.text).squeeze(1)
                batch.label = self.cast_label(batch.label, criterion)
                loss = criterion(predictions, batch.label)             
                if self.model.output_dim == 1:
                    #round predictions to the closest integer
                    predictions = torch.round(torch.sigmoid(predictions))
                else:
                    _, predictions = torch.max(predictions, 1) # get max index from one-hot predictions
                acc = self.eval_function(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc
            
        length = (len(iterator) -skipped)
        return epoch_loss / length, epoch_acc / length

    def epoch_time(self,start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def start(self, train_iterator, valid_iterator,optimizer, criterion,N_EPOCHS = 5):
        ''' Start training and Evaluation for all epochs'''

        self.best_valid_loss = float('inf')
        self.val_loss = []
        self.val_metric = []
        self.tr_loss = []
        self.tr_metric = []
        print(f'The model has {self.count_parameters():,} trainable parameters')
        print(f'Using {self.metric_name} metric')

        for epoch in range(N_EPOCHS):
            
            # Calculate training time
            start_time = time.time()
            print("epoch #{}".format(epoch))
            # Get epoch losses and accuracies 
            train_loss, train_metric = self.train(train_iterator, optimizer, criterion)
            valid_loss, valid_metric = self.evaluate(valid_iterator, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            # Save training metrics
            self.val_loss.append(valid_loss)
            self.val_metric.append(valid_metric)
            self.tr_loss.append(train_loss)
            self.tr_metric.append(train_metric)
            
            if valid_loss < self.best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'CNN-model.pt')
                    
                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train {self.metric_name}: {train_metric*100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. {self.metric_name}: {valid_metric*100:.2f}%')
        
        return self

    def plot(self):
        ''' Plot accuracy metric and loss'''

        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].plot(self.val_loss, label='Validation loss')
        ax[0].plot(self.tr_loss, label='Training loss')
        ax[0].set_title('Losses')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[1].plot(self.val_metric, label=f'Validation {self.metric_name}')
        ax[1].plot(self.tr_metric, label=f'Training {self.metric_name}')
        ax[1].set_title(f'{self.metric_name}')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel(f'{self.metric_name}')
        plt.legend()
        plt.show()
    
    def test(self,test_iterator,criterion):
        '''Evaluate model on test data'''

        self.model.load_state_dict(torch.load('CNN-model.pt'))
        test_loss, test_acc = self.evaluate(test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test {self.metric_name}: {test_acc*100:.2f}%')

        return self
