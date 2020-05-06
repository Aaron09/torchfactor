import torch 
import torch.nn as nn
import numpy as np

def torch_kron(A, B):
    return torch.einsum('ij, kl->ikjl', A, B).view((A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]))


class AutoEncoderTucker(nn.Module):
    def __init__(self, target_dims, latent_dims):
        super(AutoEncoderTucker, self).__init__()    
        B = np.random.uniform(size=(target_dims[1], latent_dims[1]))
        C = np.random.uniform(size=(target_dims[2], latent_dims[2]))
        G = np.random.uniform(size=(latent_dims[0], latent_dims[1] * latent_dims[2]))
        
        self.B = nn.Parameter(torch.Tensor(B))
        self.C = nn.Parameter(torch.Tensor(C))
        self.G = nn.Parameter(torch.Tensor(G))
        
        
        self.B_inv = nn.Parameter(torch.Tensor(np.linalg.pinv(B)))
        self.C_inv = nn.Parameter(torch.Tensor(np.linalg.pinv(C)))
        self.G_inv = nn.Parameter(torch.Tensor(np.linalg.pinv(G)))
        
        self.sm = nn.Softmax(-1)
        self.relu = nn.ReLU()

    def get_A(self, X):
        return self.encode(X)
    
    def get_BCG(self):
        return self.B.data, self.C.data, self.G.data
    
    def encode(self, X):
        X_T = X.transpose(0, 1)
        C_KR_B_INV = torch_kron(self.C_inv, self.B_inv)
        G_inv_T = self.G_inv.transpose(-1, -2)
        A_T = G_inv_T @ C_KR_B_INV @ X_T
        return A_T.transpose(-1, -2)
    
    def decode(self, A):
        C_KR_B = torch_kron(self.sm(self.C), self.sm(self.B))
        
        return self.sm(A) @ self.relu(self.G) @ C_KR_B.transpose(-1, -2)
    
    def forward(self, X):
        return self.decode(self.encode(X))