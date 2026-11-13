from torch import nn
from models.admm_base import ADMMBase
from models.admm_dual import ADMMDual
from models.gd_prim import GDPRIM
from models.gd_dual import GDDUAL
class NET_CLASS(nn.Module):
    '''
    Base MLP class. Produces the trainable net and decides about the structure depending on the input
    '''
    def __init__(self,name:str="linear",input_dim:int=2,output_dim:int=100,problem_parameters:dict={},num_pars:int=100):
        super().__init__()
        self.name = name
        if num_pars is None:
            self.num_pars = 1
        else:
           self.num_pars = num_pars
        # IMPORTANT! MAKES THAT ALL NUMBERS ARE IN THE SAME DATA TYPE
        self.float()

        self.input_dim = input_dim
        self.output_dim= output_dim
        match self.name:
            case "gd dpc dual":
                self.net = GDDUAL(mat_dim=output_dim,
                                    input_dim=input_dim,
                                    num_steps=num_pars,
                                    problem_parameters=problem_parameters)
            case "gd dpc prim":
                self.net = GDPRIM(mat_dim=output_dim,
                                    input_dim=input_dim,
                                    num_steps=num_pars,
                                    problem_parameters=problem_parameters)
            case "admm base":
                self.net = ADMMBase(mat_dim=output_dim,
                                    input_dim=input_dim,
                                    num_steps=num_pars,
                                    problem_parameters=problem_parameters)
            case "admm dual":
                self.net = ADMMDual(mat_dim=output_dim,
                                    input_dim=input_dim,
                                    num_steps=num_pars,
                                    problem_parameters=problem_parameters)
            case _ :
                print("No valid model selected")
                
    def __str__(self):
        return self.name
    
    def forward(self,xb):
        return self.net(xb)