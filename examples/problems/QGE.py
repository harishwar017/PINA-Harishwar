
import torch

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operators import nabla, grad, div, curl, advection
from pina import Condition, Span, LabelTensor


class QGE(SpatialProblem, TimeDependentProblem):

    output_variables = ['q', 'si']
    spatial_domain = Span({'x': [0, 1], 'y': [-1, 1]})
    temporal_domain = Span({'t': [0, 100]})
    
    def rand_choice_integer_Data(self):
        pass 
    
    def eq1(input_, output_):
        nu = 0.0022
        Re = 100 / nu
        
        #convective term
        si_curl = curl(output_.extract(['si']), input_, d = ['x', 'y'])
        convective_ = advection(output_.extract(['q']), input_, velocity_field = si_curl, d = ['x', 'y'])
        
        #diffusive term
        diffusive_ = nabla(output_.extract(['q']), input_, d = ['x', 'y'])   
        
        #transient term 
        du = grad(output_.extract(['q']), input_)
        transient_ = du.extract('dqdt')
        
        #forcing term
        force_ = torch.sin(torch.pi * input_.extract(['y']))
        
        return transient_ + convective_ - (1/Re) * diffusive_ - force_
    
    def eq2(input_, output_):
        r0 = 0.0036
        
        #second equation
        output = output_.extract(['q']) + r0 * nabla(output_.extract(['si']), input_, d = ['x', 'y']) - input_.extract(['y'])
        return output
        
    def continuity(input_, output_):
        si_curl = curl(output_.extract(['si']), input_, d = ['x', 'y'])
        return div(si_curl, input_, d = ['x', 'y'])

    def initial(input_, output_):
        value = 0.0
        return output_.extract(['si']) - value
    
    def zeta(input_, output_):
        value = input_.extract(['y'])
        return output_.extract(['q']) - value
    
    def si(input_, output_):
        si_expected = 0.0
        return output_.extract(['si']) - si_expected
        
    
    conditions = {
        't0': Condition(Span({'x': [0, 1], 'y': [-1, 1], 't' : 0}), initial),
        
        'upper': Condition(Span({'x':  [0,1], 'y': 1, 't': [0,100]}), [si, zeta]),
        'fixedWall1': Condition(Span({'x':  0, 'y': [-1,1], 't': [0,100]}), [si, zeta]),
        'fixedWall2': Condition(Span({'x':  1, 'y': [-1,1], 't': [0,100]}), [si, zeta]),
        'fixedWall3': Condition(Span({'x':  [0,1], 'y': -1, 't': [0,100]}), [si, zeta]),
        
        'D': Condition(Span({'x': [0, 1], 'y': [-1, 1], 't': [0, 100]}), [eq1, eq2]),
    }
