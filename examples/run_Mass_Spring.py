import argparse
import torch
from torch.nn import Softplus
from pina import PINN, Plotter, LabelTensor
from pina.model import FeedForward
from problems.Mass_Spring_Dis1 import MassSpring1D
import numpy as np


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """
    def __init__(self, idx):
        super(myFeature, self).__init__()
        self.idx = idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature(0)] if args.features else []

    # Define Structural Properties
    
    M_2dof = np.zeros((2,2))
    K_2dof = np.zeros((2,2))
    M_2dof[0,0] = 1
    M_2dof[0,1] = 1.8 
    M_2dof[1,0] = 1.8
    M_2dof[1,1] = 3.48
    K_2dof[0,0] = 1.0
    K_2dof[0,1] = 0.0
    K_2dof[1,0] = 0.0
    K_2dof[1,1] = 3.48
    vf  = 0.8410
    nt = 1000
    
    MassSpring_problem = MassSpring1D(M_2dof,K_2dof,vf,nt)
    
    model = FeedForward(
        layers=[30, 20, 10, 5],
        output_variables=MassSpring_problem.output_variables,
        input_variables=MassSpring_problem.input_variables,
        func=Softplus,
        extra_features=feat,
    )

    pinn = PINN(
        MassSpring_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 1000,'variables': 'all'},
            locations=['D','gamma'])
        pinn.train(1, 1)
        pinn.save_state('pina.MassSpring.{}.{}'.format(args.id_run, args.features))
    else:
        pinn.load_state('pina.MassSpring.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()
        plotter.plot(pinn)
        plotter.plot_loss(pinn)

