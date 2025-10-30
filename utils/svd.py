# -*- coding: utf-8 -*-
import torch
from torch import nn

class SVDSECON(nn.Module):
    '''
    Reference: Vipin Vijayan (2022). Fast SVD and PCA (https://www.mathworks.com/matlabcentral/fileexchange/47132-fast-svd-and-pca), MATLAB Central File Exchange. Retrieved April 15, 2022.
    '''
    def __init__(self, k='full', max_iter=8):
        super(SVDSECON, self).__init__()
        self.k = k
        self.max_iter = max_iter
        
    
    def EigsOrthIter(self, A):
        batch, n, _ = A.shape
        if self.k == 'half':
            k = int(n*0.5)
        elif self.k == 'full':
            k = n
        else:
            k = self.k

        eig_vec = torch.stack([torch.eye(n, k, device=A.device, dtype=A.dtype) for i in range(batch)])
        for i in range(self.max_iter):
            z = A.bmm(eig_vec)
            eig_vec, eig_val = torch.linalg.qr(z, mode='reduced')
            eig_val = eig_val.abs()
            
        eig_val = torch.stack([torch.diag(eig_val[i]) for i in range(batch)])
        
        return eig_vec, eig_val

    def forward(self, X):
        batch, m, n = X.shape
        
        if m <= n:
            U, S = self.EigsOrthIter(X.bmm(X.transpose(-1,-2)))
            V = X.transpose(-1,-2).bmm(U)
            S.sqrt_()
            V = V/(S[:,None,:]+1e-10)
        else:
            V, S = self.EigsOrthIter(X.transpose(-1,-2).bmm(X))
            U = X.bmm(V)
            S.sqrt_()
            U = U/(S[:,None,:]+1e-10)
        
        self.U = U
        self.S = S
        self.V = V
        return U, S, V
            
    def recon(self):
        return self.U @ torch.diag_embed(self.S) @ self.V.transpose(-1,-2)
    
def svdsecon(X, k='full', max_iter=8):
    U,S,V = SVDSECON(k,max_iter)(X)
    return U, S, V

def svd_recon(U,S,V):
    return U @ torch.diag_embed(S) @ V.transpose(-1,-2)