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
        import pandas as pd
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

class CNNHelper(object):

    def __init__(self,model):
        super().__init__()
        self.model = model

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def binary_accuracy(self,preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc

    def train(self, iterator, optimizer, criterion):
    
        epoch_loss = 0
        epoch_acc = 0   
        self.model.train()
        from tqdm import tqdm 

        for batch in tqdm(iterator):
            
            optimizer.zero_grad()           
            predictions = self.model(batch.text).squeeze(1)          
            loss = criterion(predictions, batch.label)    
            acc = self.binary_accuracy(predictions, batch.label)     
            loss.backward()     
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, iterator, criterion):
    
        epoch_loss = 0
        epoch_acc = 0       
        self.model.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                predictions = self.model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)             
                acc = self.binary_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(self,start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def start(self, train_iterator, valid_iterator,optimizer, criterion,N_EPOCHS = 5):

        self.best_valid_loss = float('inf')
        self.val_loss = []
        self.val_acc = []
        self.tr_loss = []
        self.tr_acc = []
        print(f'The model has {self.count_parameters():,} trainable parameters')

        for epoch in range(N_EPOCHS):
            
            # Calculate training time
            start_time = time.time()
            print("epoch #{}".format(epoch))
            # Get epoch losses and accuracies 
            train_loss, train_acc = self.train(train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(valid_iterator, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            # Save training metrics
            self.val_loss.append(valid_loss)
            self.val_acc.append(valid_acc)
            self.tr_loss.append(train_loss)
            self.tr_acc.append(train_acc)
            
            if valid_loss < self.best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'CNN-model.pt')
                    
                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
        return self

    def plot(self):

        # Plot accuracy and loss
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].plot(self.val_loss, label='Validation loss')
        ax[0].plot(self.tr_loss, label='Training loss')
        ax[0].set_title('Losses')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[1].plot(self.val_acc, label='Validation accuracy')
        ax[1].plot(self.tr_acc, label='Training accuracy')
        ax[1].set_title('Accuracies')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        plt.legend()
        plt.show()
    
    def test(self,test_iterator,criterion):
        # Evaluate model on test data
        self.model.load_state_dict(torch.load('CNN-model.pt'))
        test_loss, test_acc = self.evaluate(test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

        return self