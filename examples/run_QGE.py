import argparse
from torch.nn import Softplus
from pina import PINN, Plotter
from pina.model import FeedForward
from problems.QGE import QGE

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()
    
    qge_problem = QGE()
    
    model = FeedForward(
        layers=[256, 128, 64, 32],
        output_variables=qge_problem.output_variables,
        input_variables=qge_problem.input_variables,
        # func=Softplus,
    )
    pinn = PINN(
        qge_problem,
        model,
        lr=0.01,
        error_norm='mse',
        regularizer=1e-8
        )

    if args.s:
        pinn.span_pts(
                {'n': 25, 'mode': 'grid', 'variables': 't'},
                {'n': 10, 'mode': 'grid', 'variables': 'x'},
                {'n': 20, 'mode': 'grid', 'variables': 'y'},
                # locations=['t0', 'upper','fixedWall1','fixedWall2','fixedWall3', 'D'])
                locations=['D'])

        # pinn.train(5000, 500)
        pinn.train(1, 1)
        with open('problems/QGE/qge_files/qge_history_{}.txt'.format(args.id_run), 'w') as file_:
            for i, losses in pinn.history_loss.items():
                file_.write('{} {}\n'.format(i, sum(losses)))
        pinn.save_state('problems/QGE/qge_files/pina_data_trial.qge')
    else:
        pinn.load_state('problems/QGE/qge_files/pina_data_trial.qge')
        # pinn.load_state('problems/QGE/qge_results_good/pina_data_trial.qge')
        plotting = Plotter.plot_with_existing_points
        plotting(pinn, components='si')
        # plotting.plot_loss(pinn)
