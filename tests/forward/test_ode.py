import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import fatdae.solvers
import fatdae.problem
import json
import scipy.sparse
import numpy

with open(sys.path[0]+'/fatdae/json_butcher/DIRK/SDIRK1.json') as data_file:
        butcher_json = json.load(data_file)

solver = fatdae.solvers.build(butcher_json,False,True)

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

problem = fatdae.problem.Problem(M(t_0, x_0), f, x_0, t_0, t_f, derivatives)

def x(t):

    b = numpy.array([x_0[0] * numpy.exp(-lmb * t), \
                     x_0[1] * numpy.exp(-dlt * t)])

    return b

problem.exact = x

problem.solve(solver, h=5.e-2, tlm=False); problem.plot()