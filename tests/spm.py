# Date: 15/02/2022
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'..'); import fatDAE

# Dolfin package
from dolfin import *

# User defined
from fatDAE.dolfin_interface.class_problem import UFL_Problem
from fatDAE.dolfin_interface.class_problem import UFL_Control

class SPM_Problem(UFL_Problem):
    
    def __init__(self):
        '''
        SPM
        '''
        v_form = self.get_v_form()
        UFL_Problem.__init__(self, v_form, x_0, t_0, t_f)
    
    def get_v_form(self):
        pass

class SPM_Control(UFL_Control, SPM_Problem):

    def __init__(self):
        '''
        '''
        v_form = self.get_v_form()
        UFL_Control.__init__(self, control, J_form, g_form, \
                                   v_form, x_0, t_0, t_f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('json', help = "Butcher's tables")

    parser.add_argument('-adv', help = "Advancing method, choose 1 or 2.", action = 'store', required = True)
    parser.add_argument('-est', help = "Estimator method, choose 1 or 2.", action = 'store')

    args = parser.parse_args()

    with open(args.json) as data_file:
        butcher_json = json.load(data_file)

    if args.adv == '1':
        embedded_1 = False
        embedded_2 = True
    else:
        if args.adv == '2':
            embedded_1 = True
            embedded_2 = False
        else:
            raise NameError('Choose between 1 or 2 for advancing method...')

    solver = fatDAE.class_solvers.build(butcher_json, embedded_1, embedded_2)

    '''
        Dahlquist problem

            y_1'(t) = - lmb * y_1(t)
            y_2'(t) = - dlt * y_2(t)

        with initial condition:

            y_1(0) = 1.
            y_2(0) = 1.

    '''

    lmb = 1.
    dlt = 5.

    def M(t, x):

        A = numpy.array([[1., 0.], \
                         [0., 1.]])

        return scipy.sparse.csc_matrix(A)

    def f(t, x):

        b = numpy.array([- lmb * x[0], \
                         - dlt * x[1]])

        return b

    def dfdx(t, x):

        A = numpy.array([[- lmb, 0.], \
                         [0., -dlt]])

        return scipy.sparse.csc_matrix(A)

    def dfdt(t, x):

        b = numpy.array([0., \
                         0.])

        return b

    x_0 = numpy.array([1., \
                       1.])

    t_0 = 0.
    t_f = 1.

    derivatives = {'dfdx': dfdx, \
                   'dfdt': dfdt}

    problem = Problem(M(t_0, x_0), f, x_0, t_0, t_f, derivatives)

    def x(t):

        b = numpy.array([x_0[0] * numpy.exp(-lmb * t), \
                         x_0[1] * numpy.exp(-dlt * t)])

        return b

    problem.exact = x

    problem.solve(solver, h=5.e-2, tlm=False); problem.plot()
