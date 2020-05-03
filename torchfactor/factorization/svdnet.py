import numpy as np
import torch
import torch.nn as nn

class SVDNet(nn.Module):
    """SVD implemented as a neural network in PyTorch"""

    class HouseHolderLayer(nn.Module):
        """Householder layer for SVD network"""

        def __init__(self, n, num_hhrs):
            super(SVDNet.HouseHolderLayer, self).__init__()
            
            self.hhrs = nn.ParameterList([])
            for i in range(num_hhrs):
                self.hhrs.append(nn.Parameter(torch.Tensor(n)))

        def mult_by_hhr(self, x, h):
            hT = h.unsqueeze(0)
            h = h.unsqueeze(1)
            beta = 2 / (h**2).sum()

            return x - beta * (h @ (hT @ x)) 
        
        def forward(self, x):
            res = self.mult_by_hhr(x, self.hhrs[0])
            for i in range(1, len(self.hhrs)):
                res = self.mult_by_hhr(res, self.hhrs[i])
            return res


    def __init__(self, n, num_hhrs):
        """Params
        n: TODO INSERT HERE
        num_hhrs: TODO INSERT HERE
        """
        super(SVDNet, self).__init__()
        self.sigma = nn.Parameter(torch.ones(n,1))
        
        self.U = SVDNet.HouseHolderLayer(n, num_hhrs)
        self.VT = SVDNet.HouseHolderLayer(n, num_hhrs)
        
        for p in self.U.parameters():
            torch.nn.init.normal_(p)
        for p in self.VT.parameters():
            torch.nn.init.normal_(p)
    

    def get_USVT(self, device):
        eye = torch.eye(self.sigma.shape[0]).to(device)
        U = self.U(eye).detach().cpu().numpy()
        
        S = np.diag(self.sigma.data.cpu().numpy()[:,0])

        VT = self.VT(eye).detach().cpu().numpy()
        return U, S, VT
    

    def forward(self, x):
        return self.U(self.sigma * self.VT(x))
