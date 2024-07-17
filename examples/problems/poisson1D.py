import numpy as np
import torch
from pina.problem import SpatialProblem
from pina.operators import grad
from pina import Condition, Span
from pina.label_tensor import LabelTensor


class Poisson1D(SpatialProblem):

    output_variables = ['u']
    spatial_domain = Span({'x': [0, torch.pi]})

    def laplace_equation(input_, output_):
        f = (torch.sin(input_.extract(['x'])))
        du = grad(output_, input_)
        Coeff = 0.5*(torch.sin(input_.extract(['x'])*torch.pi*16))
        p = LabelTensor(torch.mul(Coeff,du),'p')
        dp = grad(p, input_)
        return 0.1*(dp - f)

    def nil_dirichlet(input_, output_):
        value = 0.0
        return (output_.extract(['u']) - value)

    conditions = {
        'gamma1': Condition(Span({'x': 0}), nil_dirichlet),
        'gamma2': Condition(Span({'x': torch.pi}), nil_dirichlet),
        'D': Condition(Span({'x': [0, torch.pi]}), laplace_equation),
    }

    
