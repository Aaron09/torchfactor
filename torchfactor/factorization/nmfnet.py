import torch
import torch.nn as nn

class NMFNet(nn.Module):
    """NMF implemented as a neural network"""
    
    def __init__(self, X_height, k):
        """Params
        X_height: TODO INSERT DESC HERE
        k: TODO INSERT DESC HERE
        """
        super(NMFNet, self).__init__()
        
        self.k = k
        self.W = nn.Parameter(torch.Tensor(X_height, k))
        self.W_inv = nn.Parameter(torch.Tensor(k, X_height))
        
        self.relu = nn.ReLU()
        
        # initialize W and W_inv
        nn.init.uniform_(self.W)
        nn.init.uniform_(self.W_inv)
        
        
    def forward(self, X):
        H = self.get_H(X) # this is the encoder
        
        X_hat = self.relu(self.W) @ H # this is the decoder
        return X_hat
    
    
    def get_H(self, X):
        return self.relu(self.W_inv @ X)
    
    
    def get_W(self):
        return self.W.data