# -*- coding: utf-8 -*-
# config.py
import nltk
import torch
from nltk.corpus import stopwords
from nltk import SnowballStemmer

class Config(object):    
    # TEXT CLENAING
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    
    # Checking if GPU is available or not
    is_cuda = torch.cuda.is_available()
    if is_cuda: device = torch.device("cuda")
    else: device = torch.device("cpu")

    # Stop words
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    # Model params
    batch_size = 32            # Batch size 
    embed_size = 300           # Word2Vec Embedding size       
    hidden_layers = 2          # Number of Hidden layers for Bi-directional LSTM
    hidden_size = 100          # Size of each Hidden layer in LSTM
    output_size = 2            # Output size 
    hidden_size_linear = 128   # Fully connected layers
    dropout_keep = 0.51        # Dropout layer probability
    lr= 0.05                 # Learning rate
    epochs = 100               # Number of Epochs
    
    # Directories path
    model_path = ""                                                     # Trained model path state_dict.pt file
    embedding_path = "./Dataset/embedding_matrix.npz"                   # Embedding matrix path .npz file
    train_path = "./Dataset/trainset.npz"                               # Training data file path .npz
    test_path = "./Dataset/validset.npz"                                # Testing data file path .npz
    tokenizer_path = ""                                                 # Tokenizer file path which you can use during inference
    path = "./results/RCNN/0.0005"    # directory path to save results
