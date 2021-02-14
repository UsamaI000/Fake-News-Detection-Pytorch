# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from src.evals import Test_Eval
from src.model import RCNNSentimentNet, LSTMSentimentNet
from config import Config
from src.utils import Dataloader

# LOADING MODEL CONFIGURATION FROM CONFIG FILE
CONFIG = Config()
TEST_PATH = CONFIG.test_path
EMBEDDING_PATH = CONFIG.embedding_path
BATCH_SIZE = CONFIG.batch_size
MODEL_PATH = CONFIG.model_path
DEVICE = CONFIG.device


if __name__ == "__main__":
    # LOADING TEST AND EMBEDDING MATRIX .NPZ FILES
    test_data = np.load(TEST_PATH, allow_pickle=True)
    embed = np.load(EMBEDDING_PATH, allow_pickle=True)
    test_x = test_data['arr_4']
    test_y = test_data['arr_5']
    embedding_matrix = embed['arr_0']
    
    # LOAD TEST DATA LOADER
    test_loader = Dataloader(test_x, test_y, BATCH_SIZE)
    
    # MODEL PARAMS AND INSTANTIATION
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    model = RCNNSentimentNet(CONFIG, vocab_size, torch.Tensor(embedding_matrix))
    model.to(DEVICE)
    
    # LOSS FUNCTION AND OPTIMIZATION FUNCTION
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # LOADING SAVED MODEL STATE_DICT
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # EVALUATION ON TEST SET
    loss, acc, f1 = Test_Eval(model, test_loader, criterion)
    print("The loss of test data is: {:.5f}".format(loss))
    print("The accuracy of test data is: {:.5f}".format(acc*100))
    print("The F1-score of test data is: {:.5f}".format(f1))