import torch 
import torch.nn as nn

def torch_kron(A, B):
    return torch.einsum('ij, kl->ikjl', A, B).view((A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]))

class DirectTuckerNet(nn.Module):
    def __init__(self, target_dims, latent_dims, nonnegative=False):
        super(DirectTuckerNet, self).__init__()
        self.A = nn.Parameter(torch.Tensor(target_dims[0], latent_dims[0]))
        self.B = nn.Parameter(torch.Tensor(target_dims[1], latent_dims[1]))
        self.C = nn.Parameter(torch.Tensor(target_dims[2], latent_dims[2]))

        self.G = nn.Parameter(torch.Tensor(latent_dims[0], latent_dims[1] * latent_dims[2]))
        
        nn.init.uniform_(self.A)
        nn.init.uniform_(self.B)
        nn.init.uniform_(self.C)
        nn.init.uniform_(self.G)

        if nonnegative:
            self.sm = nn.Softmax(-1)
            self.relu = nn.ReLU()
        else:
            self.sm = nn.Identity()
            self.relu = nn.Identity()
        
    def get_ABCG(self):
        return self.A.data, self.B.data, self.C.data, self.G.data
    
    def forward(self):
        C_KR_B = torch_kron(self.sm(self.C), self.sm(self.B))
        return self.sm(self.A) @ self.relu(self.G ) @ C_KR_B.transpose(-1, -2)