


import pandas as pd


import random
import torch, torchtext
from torchtext import data
import os, pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
import en_core_web_sm




class classifier(nn.Module):
    
    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        
        super().__init__()          
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.encoder = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout,
                           batch_first=True)
        # try using nn.GRU or nn.RNN here and compare their performances
        # try bidirectional and compare their performances
        
        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):
        
        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
      
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
    
        # Hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)   
        
        # Final activation function softmax
        output = F.softmax(dense_outputs[0], dim=1)
            
        return output
		
		
# define metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    _, predictions = torch.max(preds, 1)
    
    correct = (predictions == y).float() 
    acc = correct.sum() / len(correct)
    return acc
	
def train_model(model, iterator, optimizer, criterion):
    
    # initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    # set the model in training phase
    model.train()  
    
    for batch in iterator:
        
        # resets the gradients after every batch
        optimizer.zero_grad()   
        
        # retrieve text and no. of words
        tweet, tweet_lengths = batch.tweets   
        
        # convert to 1D tensor
        predictions = model(tweet, tweet_lengths).squeeze()  
        
        # compute the loss
        loss = criterion(predictions, batch.labels)        
        
        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.labels)   
        
        # backpropage the loss and compute the gradients
        loss.backward()       
        
        # update the weights
        optimizer.step()      
        
        # loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_model(model, iterator, criterion):
    
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()
    
    # deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            # retrieve text and no. of words
            tweet, tweet_lengths = batch.tweets
            
            # convert to 1d tensor
            predictions = model(tweet, tweet_lengths).squeeze()
            
            # compute loss and accuracy
            loss = criterion(predictions, batch.labels)
            acc = binary_accuracy(predictions, batch.labels)
            
            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
	
def perform_training(dataset_name, dataset_path,  N_EPOCHS=10, lr=2e-4):
	
    df = pd.read_csv(dataset_path)
    df.columns = ['tweets', 'labels']
    unique_value_list = df.labels.unique()
    num_output_nodes = len(unique_value_list)	

    # Manual Seed
    SEED = 43
    torch.manual_seed(SEED)
    spacy.load('en')
    #nlp = en_core_web_sm.load()

    #spacy.load('en')

    Tweet = data.Field(sequential = True, tokenize = 'spacy', batch_first =True, include_lengths=True)
    Label = data.LabelField(tokenize ='spacy', is_target=True, batch_first =True, sequential =False)

    fields = [('tweets', Tweet),('labels',Label)]

    example = [data.Example.fromlist([df.tweets[i],df.labels[i]], fields) for i in range(df.shape[0])] 

    twitterDataset = data.Dataset(example, fields)

    (train, valid) = twitterDataset.split(split_ratio=[0.80, 0.20], random_state=random.seed(SEED))

    Tweet.build_vocab(train)
    Label.build_vocab(train)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iterator, valid_iterator = data.BucketIterator.splits((train, valid), batch_size = 32, 
                                                            sort_key = lambda x: len(x.tweets),
                                                            sort_within_batch=True, device = device)
															

    vocab_path = dataset_name + '_sa.pkl'
    with open("vocab_files/" + vocab_path, 'wb') as tokens: 
        pickle.dump(Tweet.vocab.stoi, tokens)


    # Define hyperparameters
    size_of_vocab = len(Tweet.vocab)
    embedding_dim = 300
    num_hidden_nodes = 100
    num_layers = 2
    dropout = 0.2

    # Instantiate the model
    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout = dropout)


    # define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # push to cuda if available
    model = model.to(device)
    criterion = criterion.to(device)



    best_valid_acc = float('-inf')
    best_train_acc = float('-inf')
    model_path = dataset_name + '_sa.pt'

    for epoch in range(N_EPOCHS):
     
        # train the model
        train_loss, train_acc = train_model(model, train_iterator, optimizer, criterion)
    
        # evaluate the model
        valid_loss, valid_acc = evaluate_model(model, valid_iterator, criterion)
    
        # save the best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_train_acc = train_acc
            torch.save(model.state_dict(), "model_files/" + model_path)
    
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% \n')

    print("Best validation accuracy: ", best_valid_acc)
    print("Best training accuracy: ", best_train_acc)
    result_dict = {}
    result_dict['training_status'] = 'Training Successful'
    result_dict['training_accuracy'] = str(best_train_acc)
    result_dict['validation_accuracy'] = str(best_valid_acc)
    result_dict['model_path'] = model_path
    result_dict['vocab_path'] = vocab_path
    result_dict['size_of_vocab'] = str(size_of_vocab)
    result_dict['num_output_nodes'] = str(size_of_vocab)

    return result_dict



#inference

import spacy
nlp = spacy.load('en')

def predict_sentiment_analysis(dataset_name,input_text, model_path, vocab_path, size_of_vocab, num_output_nodes ) :
    try:
        tokenizer_file = open(vocab_path, 'rb')
        tokenizer = pickle.load(vocab_path)
        embedding_dim = 300
        num_hidden_nodes = 100
        num_layers = 2
        dropout = 0.2

        # Instantiate the model
        model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout = dropout)
        model.load_state_dict(torch.load(model_path));
        model.eval();

        # tokenize the tweet
        tokenized = [tok.text for tok in nlp.tokenizer(tweet)]
        # convert to integer sequence using predefined tokenizer dictionary
        indexed = [tokenizer[t] for t in tokenized]
        # compute no. of words
        length = [len(indexed)]
        # convert to tensor
        tensor = torch.LongTensor(indexed).to(device)
        # reshape in form of batch, no. of words
        tensor = tensor.unsqueeze(1).T
        # convert to tensor
        length_tensor = torch.LongTensor(length)
        # Get the model prediction
        prediction = model(tensor, length_tensor)

        _, pred = torch.max(prediction, 1)
       resp = {}
       resp['prediction_status'] = 'Prediction Successful'
       resp['prediction_value'] = pred.item()
       return resp
   except Exception:
       resp = {}
       resp['prediction_status'] = 'Prediction failed because of an internal error'
    #return categories[pred.item()]

#perform_training('tweets', 'tweets1.csv')
