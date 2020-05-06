import torch
import torch.nn as nn

from numpy.random import uniform
from numpy.linalg import pinv

class NMFNet(nn.Module):
    """NMF implemented as a neural network"""
    
    def __init__(self, X_height, k):
        """Params
        X_height: TODO INSERT DESC HERE
        k: TODO INSERT DESC HERE
        """
        super(NMFNet, self).__init__()
        
        self.k = k
        
        W_numpy = uniform(0, 1, (X_height, k))
        W_numpy = W_numpy / W_numpy.sum(0)[None,:]
        self.W = nn.Parameter(torch.FloatTensor(W_numpy))
        self.W_inv = nn.Parameter(torch.FloatTensor(pinv(W_numpy)))
        
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=0)
        
    def forward(self, X):
        H = self.get_H(X) # this is the encoder
        
        X_hat = self.sm(self.W) @ H # this is the decoder
        
        return X_hat
    
    
    def get_H(self, X):
        return self.relu(self.W_inv @ X)
    
    
    def get_W(self):
        return self.sm(self.W)