import numpy as np
import torch
from scipy.io import loadmat
import os
class BaseADMMLoss(torch.nn.Module):
    '''Base class for the loss functions'''
    def __init__(self) -> None:
        super().__init__()
        self.weights= [1,1]
        self.name_self()
        return
    def _step(self):
        return
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Sub-classes that depend on the base gradient-loss
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

class DPCmegaDUAL(BaseADMMLoss):
    '''
    Dual cost function that sums up the error over all optimization steps
    '''
    def __init__(self,n,m,N):
        super().__init__()
        # Read the DPC parameter from mmatlab-files
        # Produce a cost function from that
        to_load = 'QP_parameters_SS.mat'
        own_dic = os.path.dirname(os.path.realpath(__file__))
        dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{N}',to_load))
        HINV = np.linalg.inv(dpc_parameters['H'])
        self.register_buffer('H',torch.tensor(dpc_parameters['H']))
        self.register_buffer('F',torch.tensor(dpc_parameters['F']))
        self.register_buffer('d',torch.tensor(dpc_parameters['d']))
        self.register_buffer('G',torch.tensor(dpc_parameters['G']))
        self.register_buffer('E',torch.tensor(dpc_parameters['E']))
        self.register_buffer('GHG',torch.tensor(dpc_parameters['G']@HINV@dpc_parameters['G'].T))
        self.register_buffer('EGHF',torch.tensor(dpc_parameters['E']+dpc_parameters['G']@HINV@dpc_parameters['F']))
        return
    def name_self(self)->None:
        self.name = 'dpc mega dual'
        return
    def forward(self,
                z:torch.Tensor,
                z_hat:torch.Tensor,
                z_opt:torch.Tensor,
                mu:torch.Tensor,
                x:torch.Tensor)->torch.Tensor:
        # prepare the dual cost function
        d = self.d.to(device= x.device,dtype = x.dtype)
        GHG = self.GHG.to(device= x.device,dtype = x.dtype)
        EGHF = self.EGHF.to(device= x.device,dtype = x.dtype)
        # Cost function as the dpc
        full_loss = 0
        for i in range(1, z.shape[2]):
            mu_i = mu[:,:,i]
            quad = 0.5 * torch.einsum('bi,ij,bj->b', mu_i,GHG, mu_i)
            q = x @ EGHF.T + d.T
            lin = torch.sum(mu_i * q, dim=1)
            full_loss+=(quad+lin).mean()
        return full_loss

class DPCmegaDUALLast(BaseADMMLoss):
    '''
    Dual cost function that uses the last of the optimization steps to calculate the error
    '''
    def __init__(self,n,m,N):
        super().__init__()
        # Read the DPC parameter from matlab-files
        # Produce a cost function from that
        to_load = 'QP_parameters_SS.mat'
        own_dic = os.path.dirname(os.path.realpath(__file__))
        dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{N}',to_load))
        HINV = np.linalg.inv(dpc_parameters['H'])
        self.register_buffer('H',torch.tensor(dpc_parameters['H']))
        self.register_buffer('F',torch.tensor(dpc_parameters['F']))
        self.register_buffer('d',torch.tensor(dpc_parameters['d']))
        self.register_buffer('G',torch.tensor(dpc_parameters['G']))
        self.register_buffer('E',torch.tensor(dpc_parameters['E']))
        self.register_buffer('GHG',torch.tensor(dpc_parameters['G']@HINV@dpc_parameters['G'].T))
        self.register_buffer('EGHF',torch.tensor(dpc_parameters['E']+dpc_parameters['G']@HINV@dpc_parameters['F']))
        return
    def name_self(self)->None:
        self.name = 'dpc mega dual last'
        return
    def forward(self,
                z:torch.Tensor,
                z_hat:torch.Tensor,
                z_opt:torch.Tensor,
                mu:torch.Tensor,
                x:torch.Tensor)->torch.Tensor:
        # prepare the dual cost function
        d = self.d.to(device= x.device,dtype = x.dtype)
        GHG = self.GHG.to(device= x.device,dtype = x.dtype)
        EGHF = self.EGHF.to(device= x.device,dtype = x.dtype)
        # only ask for the last dual estimate
        mu_i = mu[:,:,-1]
        quad = 0.5 * torch.einsum('bi,ij,bj->b', mu_i,GHG, mu_i)
        q = x @ EGHF.T + d.T
        lin = torch.sum(mu_i * q, dim=1)
        return (quad+lin).mean()

class DPCmega(BaseADMMLoss):
    '''
    Primal cost function that sums up the error over all optimization steps
    '''
    def __init__(self,n,m,N):
        super().__init__()
        # Read the DPC parameter from matlab-files
        # Produce a cost function from that
        to_load = 'QP_parameters_SS.mat'
        own_dic = os.path.dirname(os.path.realpath(__file__))
        dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{N}',to_load))
        self.register_buffer('H',torch.tensor(dpc_parameters['H']))
        self.register_buffer('F',torch.tensor(dpc_parameters['F']))
        return
    def name_self(self)->None:
        self.name = 'dpc mega'
        return
    def forward(self,
                z:torch.Tensor,
                z_hat:torch.Tensor,
                z_opt:torch.Tensor,
                mu:torch.Tensor,
                x:torch.Tensor)->torch.Tensor:
        # prepare the primal cost function
        H = self.H.to(device= x.device,dtype = x.dtype)
        F = self.F.to(device= x.device,dtype = x.dtype)
        # Cost function as the dpc
        full_loss = 0
        # loop over all predictions
        for i in range(1, z.shape[2]):
            z_i = z[:,:,i]
            quad = 0.5 * torch.einsum('bi,ij,bj->b', z_i,H,z_i)
            q = x @ F.T
            lin = torch.sum(z_i * q, dim=1)
            full_loss+=(quad+lin).mean()
        return full_loss

class DPCmegaLast(BaseADMMLoss):
    '''
    Primal cost function that uses the last of the optimization steps to calculate the error
    '''
    def __init__(self,n,m,N):
        super().__init__()
        # Read the DPC parameter from the matlab files
        # Produce a cost function from that
        to_load = 'QP_parameters_SS.mat'
        own_dic = os.path.dirname(os.path.realpath(__file__))
        dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{N}',to_load))
        self.register_buffer('H',torch.tensor(dpc_parameters['H']))
        self.register_buffer('F',torch.tensor(dpc_parameters['F']))
        return
    def name_self(self)->None:
        self.name = 'dpc mega last'
        return
    def forward(self,
                z:torch.Tensor,
                z_hat:torch.Tensor,
                z_opt:torch.Tensor,
                mu:torch.Tensor,
                x:torch.Tensor)->torch.Tensor:
        # prepare the primal cost function
        H = self.H.to(device= x.device,dtype = x.dtype)
        F = self.F.to(device= x.device,dtype = x.dtype)
        # Only ask for the last prediction
        z_i = z[:,:,-1]
        quad = 0.5 * torch.einsum('bi,ij,bj->b', z_i,H,z_i)
        q = x @ F.T
        lin = torch.sum(z_i * q, dim=1)
        return (quad+lin).mean()

class DPCmegaFirstLast(BaseADMMLoss):
    def __init__(self,n,m,N):
        super().__init__()
        # Read the DPC parameter from the matlab files
        # Produce a cost function from that
        to_load = 'QP_parameters_SS.mat'
        own_dic = os.path.dirname(os.path.realpath(__file__))
        dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{N}',to_load))
        HINV = np.linalg.inv(dpc_parameters['H'])
        self.register_buffer('H',torch.tensor(dpc_parameters['H']))
        self.register_buffer('F',torch.tensor(dpc_parameters['F']))
        self.register_buffer('d',torch.tensor(dpc_parameters['d']))
        self.register_buffer('G',torch.tensor(dpc_parameters['G']))
        self.register_buffer('E',torch.tensor(dpc_parameters['E']))
        self.register_buffer('GHG',torch.tensor(dpc_parameters['G']@HINV@dpc_parameters['G'].T))
        self.register_buffer('EGHF',torch.tensor(dpc_parameters['E']+dpc_parameters['G']@HINV@dpc_parameters['F']))
        return
    def name_self(self)->None:
        self.name = 'dpc mega first last'
        return
    def forward(self,
                z:torch.Tensor,
                z_hat:torch.Tensor,
                z_opt:torch.Tensor,
                mu:torch.Tensor,
                x:torch.Tensor)->torch.Tensor:
        # prepare the primal cost function
        H = self.H.to(device= x.device,dtype = x.dtype)
        F = self.F.to(device= x.device,dtype = x.dtype)
        # Cost function as the dpc
        full_loss = 0
        for i in [0,-1]:
            z_i = z[:,:,i]
            quad = 0.5 * torch.einsum('bi,ij,bj->b', z_i,H,z_i)
            q = x @ F.T
            lin = torch.sum(z_i * q, dim=1)
            full_loss = quad+lin
        return full_loss.mean()

class DPCmegaHat(BaseADMMLoss):
    def __init__(self,n,m,N):
        super().__init__()
        # Read the DPC parameter from Manuel's file
        # Produce a cost function from that
        to_load = 'QP_parameters_SS.mat'
        own_dic = os.path.dirname(os.path.realpath(__file__))
        dpc_parameters = loadmat(os.path.join(own_dic,f'mat_files_{N}',to_load))
        HINV = np.linalg.inv(dpc_parameters['H'])
        self.register_buffer('H',torch.tensor(dpc_parameters['H']))
        self.register_buffer('F',torch.tensor(dpc_parameters['F']))
        return
    def name_self(self)->None:
        self.name = 'dpc mega hat'
        return
    def forward(self,
                z:torch.Tensor,
                z_hat:torch.Tensor,
                z_opt:torch.Tensor,
                mu:torch.Tensor,
                x:torch.Tensor)->torch.Tensor:
        H = self.H.to(device= x.device,dtype = x.dtype)
        F = self.F.to(device= x.device,dtype = x.dtype)
        # Cost function as the dpc
        full_loss = 0
        for i in range(1, z.shape[2]):
            z_i = z_hat[:,:,i]
            quad = 0.5 * torch.einsum('bi,ij,bj->b', z_i,H,z_i)
            q = x @ F.T
            lin = torch.sum(z_i * q, dim=1)
            full_loss+=(quad+lin).mean()
        return full_loss