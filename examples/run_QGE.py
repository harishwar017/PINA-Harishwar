
from torch.nn import Softplus
from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from problems.QGE import QGE

#args
id_run = 0
save = False

qge_problem = QGE()
model = FeedForward(
    layers=[10, 10, 10, 10],
    output_variables=qge_problem.output_variables,
    input_variables=qge_problem.input_variables,
    func=Softplus,
)
pinn = PINN(
    qge_problem,
    model,
    lr=0.006,
    error_norm='mse',
    regularizer=1e-8)

if save:
    pinn.span_pts(
            {'n': 100, 'mode': 'grid', 'variables': 't'},
            {'n': 4, 'mode': 'grid', 'variables': 'x'},
            {'n': 8, 'mode': 'grid', 'variables': 'y'},
            locations=['t0', 'upper','fixedWall1','fixedWall2','fixedWall3', 'D'])
    
    pinn.train(100, 10)
    with open('qge_history_{}.txt'.format(id_run), 'w') as file_:
        for i, losses in pinn.history_loss.items():
            file_.write('{} {}\n'.format(i, sum(losses)))
    pinn.save_state('pina.qge')
else:
    pinn.load_state('pina.qge')
    plotter = Plotter()
    plotter.plot(pinn, components='si')
    plotter.plot_loss(pinn)
