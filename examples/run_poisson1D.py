import argparse
import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
from problems.poisson1D import Poisson1D


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x'])*torch.pi))
        return LabelTensor(t, ['sin(x)'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature()] if args.features else []

    poisson_problem = Poisson1D()
    model = FeedForward(
        layers=[20, 20],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = PINN(
        poisson_problem,
        model,
        lr=0.03,
        error_norm='mse',
        regularizer=1e-8)

    if args.s:

        pinn.span_pts(100, 'grid', locations=['gamma1', 'gamma2'])
        pinn.span_pts(100, 'grid', locations=['D'])
        pinn.train(15000, 100)
        pinn.save_state('pina.poisson1D')

    else:
        pinn.load_state('pina.poisson1D')
        plotter = Plotter()
        plotter.plot(pinn)
